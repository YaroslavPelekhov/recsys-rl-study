"""NumPy-only metrics/replay; no original data, GPU, or reward oracle needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DIAGNOSTICS = (
    "predicted_ratio",
    "rm_std",
    "audit_ratio",
    "predicted_return24",
    "kl",
    "entropy",
    "recent_repeat",
    "duration",
)


def ranking_rows(top, target):
    match = np.asarray(top) == np.asarray(target)[:, None]
    rank = np.where(match.any(1), match.argmax(1) + 1, 0)
    return pd.DataFrame(
        {
            "ndcg": np.where(rank > 0, 1 / np.log2(np.maximum(rank, 1) + 1), 0.0),
            "recall": (rank > 0).astype(float),
            "map": np.where(rank > 0, 1 / np.maximum(rank, 1), 0.0),
        }
    )


def user_macro(values, users):
    return pd.DataFrame(values).assign(uid=np.asarray(users)).groupby("uid").mean().mean().to_dict()


def reward_metrics(truth, predictions):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    result = {"n": len(truth)}
    for j, name in enumerate(("ratio", "return24")):
        error = predictions[:, j] - truth[:, j]
        result[f"{name}_mse"] = float(np.mean(error**2))
        result[f"{name}_mae"] = float(np.mean(np.abs(error)))
        result[f"{name}_bias"] = float(np.mean(error))
        bins = np.minimum((predictions[:, j] * 10).astype(int), 9)
        result[f"{name}_ece10"] = float(
            sum(abs(error[bins == k].sum()) for k in range(10)) / len(error)
        )
    result["return24_auc"] = (
        float(roc_auc_score(truth[:, 1], predictions[:, 1]))
        if len(np.unique(truth[:, 1])) > 1
        else None
    )
    return result


def paired_interval(values, baseline, users, seed=718283):
    delta = (
        pd.DataFrame({"uid": users, "delta": np.asarray(values) - np.asarray(baseline)})
        .groupby("uid")
        .delta.mean()
        .to_numpy()
    )
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(3000)])
    return {
        "delta": float(delta.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "users": len(delta),
    }


def behavior_summary(top, diagnostics, users, catalog_size):
    counts = np.unique(top[:, 0], return_counts=True)[1]
    probabilities = counts / counts.sum()
    per_user = pd.DataFrame(diagnostics).assign(uid=users).groupby("uid").mean()
    result = per_user.mean().to_dict()
    result.update(
        {
            "catalog_coverage10": float(len(np.unique(top)) / catalog_size),
            "top1_item_share": float(probabilities.max()),
            "top1_hhi": float(np.sum(probabilities**2)),
            "distinct_top1": int(len(counts)),
        }
    )
    return result
