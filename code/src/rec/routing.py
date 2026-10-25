"""Small listwise-trained routers over learned historical experts.

The optional base score must be generated out of time for training rows. At inference
the router sees expert predictions, historical context and the frozen base model only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

CONTEXT = ("user_count_all", "user_age_min", "user_age_mean", "user_velocity_7", "user_velocity_30")


def query_layout(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, unique = pd.factorize(frame.query_id, sort=False)
    counts = np.bincount(ids)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    if not np.array_equal(ids, np.repeat(np.arange(len(unique)), counts)):
        raise ValueError("Query rows must be contiguous")
    return ids, starts, counts


def query_standardize(values: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        values = values[:, None]
    means = np.add.reduceat(values, starts, axis=0) / counts[:, None]
    centered = values - np.repeat(means, counts, axis=0)
    variances = np.add.reduceat(centered**2, starts, axis=0) / counts[:, None]
    normalized = centered / np.repeat(np.maximum(np.sqrt(variances), 1e-6), counts, axis=0)
    return np.clip(normalized, -5, 5)


def router_inputs(frame: pd.DataFrame, experts: list[str]) -> tuple:
    ids, starts, counts = query_layout(frame)
    raw = frame[experts].to_numpy(dtype=np.float64)
    scores = query_standardize(raw, starts, counts)
    context = frame.iloc[starts][list(CONTEXT)].to_numpy(dtype=np.float64)
    context[:, :3] = np.log1p(np.maximum(context[:, :3], 0))
    disagreement = scores.std(axis=1)
    context = np.column_stack([context, np.add.reduceat(disagreement, starts) / counts])
    return scores, context, ids, starts, counts


def listwise_objective(
    parameters: np.ndarray,
    context: np.ndarray,
    experts: np.ndarray,
    query_ids: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    labels: np.ndarray,
    offset: np.ndarray,
    scale: float,
    l2: float,
) -> tuple[float, np.ndarray]:
    weights = parameters.reshape(context.shape[1], experts.shape[1])
    gates = softmax(context @ weights, axis=1)
    logits = 3 * (offset + scale * np.sum(gates[query_ids] * experts, axis=1))
    maxima = np.maximum.reduceat(logits, starts)
    exp_scores = np.exp(logits - maxima[query_ids])
    denominators = np.add.reduceat(exp_scores, starts)
    positive_counts = np.add.reduceat(labels, starts)
    targets = labels / positive_counts[query_ids]
    probabilities = exp_scores / denominators[query_ids]
    loss = -np.sum(targets * (logits - maxima[query_ids] - np.log(denominators[query_ids]))) / len(
        starts
    )
    row_gradient = 3 * scale * (probabilities - targets) / len(starts)
    gate_gradient = np.add.reduceat(row_gradient[:, None] * experts, starts, axis=0)
    logit_gradient = gates * (gate_gradient - (gate_gradient * gates).sum(axis=1, keepdims=True))
    gradient = context.T @ logit_gradient + l2 * weights
    loss += 0.5 * l2 * np.sum(weights**2)
    return float(loss), gradient.ravel()


@dataclass
class TemporalRouter:
    experts: list[str]
    adaptive: bool = True
    l2: float = 0.01
    residual_scale: float = 1
    offset_column: str | None = None
    max_iterations: int = 160

    def fit(self, frame: pd.DataFrame) -> TemporalRouter:
        positive = frame.groupby("query_id").label.transform("sum") > 0
        data = frame.loc[positive].reset_index(drop=True)
        experts, raw_context, ids, starts, counts = router_inputs(data, self.experts)
        self.mean_ = raw_context.mean(axis=0)
        self.std_ = np.maximum(raw_context.std(axis=0), 0.1)
        context = self._context(raw_context)
        offset = (
            np.zeros(len(data))
            if self.offset_column is None
            else query_standardize(data[self.offset_column].to_numpy(), starts, counts).ravel()
        )
        result = minimize(
            listwise_objective,
            np.zeros(context.shape[1] * experts.shape[1]),
            args=(
                context,
                experts,
                ids,
                starts,
                counts,
                data.label.to_numpy(),
                offset,
                self.residual_scale,
                self.l2,
            ),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iterations, "ftol": 1e-9},
        )
        if not np.isfinite(result.fun):
            raise ValueError("Non-finite router objective")
        self.weights_ = result.x.reshape(context.shape[1], experts.shape[1])
        self.optimization_ = {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": result.nit,
            "loss": float(result.fun),
        }
        return self

    def _context(self, values: np.ndarray) -> np.ndarray:
        normalized = (values - self.mean_) / self.std_
        return (
            np.column_stack([np.ones(len(values)), normalized])
            if self.adaptive
            else np.ones((len(values), 1))
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        experts, context, ids, starts, counts = router_inputs(frame, self.experts)
        gates = softmax(self._context(context) @ self.weights_, axis=1)
        offset = (
            np.zeros(len(frame))
            if self.offset_column is None
            else query_standardize(frame[self.offset_column].to_numpy(), starts, counts).ravel()
        )
        return offset + self.residual_scale * np.sum(gates[ids] * experts, axis=1)

    def state(self) -> dict:
        return {
            "experts": self.experts,
            "adaptive": self.adaptive,
            "l2": self.l2,
            "residual_scale": self.residual_scale,
            "offset_column": self.offset_column,
            "context_mean": self.mean_.tolist(),
            "context_std": self.std_.tolist(),
            "weights": self.weights_.tolist(),
            "optimization": self.optimization_,
        }

    @classmethod
    def from_state(cls, state: dict) -> TemporalRouter:
        model = cls(
            state["experts"],
            adaptive=state["adaptive"],
            l2=state["l2"],
            residual_scale=state["residual_scale"],
            offset_column=state["offset_column"],
        )
        model.mean_ = np.asarray(state["context_mean"], dtype=np.float64)
        model.std_ = np.asarray(state["context_std"], dtype=np.float64)
        model.weights_ = np.asarray(state["weights"], dtype=np.float64)
        model.optimization_ = state["optimization"]
        return model
