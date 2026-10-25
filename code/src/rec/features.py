from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rec.protocol import Config
from rec.retrieval import tokens

RELATIONS = ("genres", "directors", "countries", "content_type")
CAT_FEATURES = ("age", "income", "sex", "kids_flg", "content_type", "age_rating", "for_kids")


@dataclass(frozen=True)
class Kernel:
    kind: str
    days: int | None = None

    @property
    def name(self) -> str:
        return "all" if self.days is None else f"{self.kind}{self.days}"

    def weights(self, ages: np.ndarray) -> np.ndarray:
        if self.kind == "window":
            return np.ones(len(ages)) if self.days is None else (ages <= self.days).astype(float)
        if self.kind == "decay" and self.days is not None:
            return np.exp2(-ages / self.days)
        raise ValueError(f"Unknown kernel {self}")


def search_kernels(config: Config) -> list[Kernel]:
    return [Kernel("window", w) for w in config.search_windows if w not in (7, 30)] + [
        Kernel("decay", w) for w in config.search_half_lives
    ]


class FeatureBuilder:
    """A bounded temporal relational grammar; feature groups are selected without test access."""

    def __init__(
        self, history: pd.DataFrame, items: pd.DataFrame, users: pd.DataFrame, cutoff: pd.Timestamp
    ):
        if history.empty or not (history.timestamp < cutoff).all():
            raise ValueError("FeatureBuilder accepts strictly historical records only")
        self.history = history.copy()
        self.history["age_days"] = (cutoff - history.timestamp).dt.total_seconds() / 86400
        self.items = items
        self.users = users
        self.cutoff = cutoff
        self.expanded = {}
        for relation in RELATIONS:
            metadata = items[[relation]].copy()
            metadata[relation] = metadata[relation].map(tokens)
            expanded = self.history.merge(metadata, left_on="item_id", right_index=True, how="left")
            expanded = expanded.explode(relation).dropna(subset=[relation])
            # An event contributes one unit in total, even for multi-valued categories.
            multiplicity = expanded.groupby(["user_id", "item_id", "timestamp"])[
                relation
            ].transform("size")
            expanded["fraction"] = 1 / multiplicity
            self.expanded[relation] = expanded

    def static(self, pairs: pd.DataFrame) -> pd.DataFrame:
        result = pairs.copy()
        for name in ("age", "income", "sex", "kids_flg"):
            result[name] = pairs.user_id.map(self.users[name]).fillna("__missing__").astype(str)
        for name in ("content_type", "age_rating", "for_kids"):
            result[name] = pairs.item_id.map(self.items[name]).fillna("__missing__").astype(str)
        result["release_year"] = (
            pairs.item_id.map(self.items.release_year).fillna(0).astype(np.float32)
        )
        for entity in ("user_id", "item_id"):
            stats = self.history.groupby(entity).age_days.agg(["min", "max", "mean"])
            prefix = "user" if entity == "user_id" else "item"
            for stat in stats:
                result[f"{prefix}_age_{stat}"] = (
                    pairs[entity].map(stats[stat]).fillna(-1).astype(np.float32)
                )
        return result

    def aggregate(self, pairs: pd.DataFrame, kernel: Kernel) -> pd.DataFrame:
        h = self.history
        weights = kernel.weights(h.age_days.to_numpy())
        weighted = h.assign(weight=weights)
        output = pd.DataFrame(index=pairs.index)
        suffix = kernel.name
        for entity, prefix in (("user_id", "user"), ("item_id", "item")):
            grouped = weighted.groupby(entity).weight.sum()
            output[f"{prefix}_count_{suffix}"] = (
                pairs[entity].map(grouped).fillna(0).astype(np.float32)
            )
        # Direct user-item counts are zero for novel-item recommendation by construction.
        # Do not add constant features merely to pad a baseline.
        total = max(float(weights.sum()), 1)
        output[f"item_share_{suffix}"] = output[f"item_count_{suffix}"] / total
        for relation, expanded in self.expanded.items():
            e = expanded.copy()
            e["weight"] = kernel.weights(e.age_days.to_numpy()) * e.fraction
            counts = e.groupby(["user_id", relation]).weight.sum()
            candidate_values = pairs[["user_id", "item_id"]].copy()
            candidate_values[relation] = pairs.item_id.map(self.items[relation]).map(tokens)
            candidate_values["pair_index"] = pairs.index
            candidate_values = candidate_values.explode(relation).dropna(subset=[relation])
            keys = pd.MultiIndex.from_frame(candidate_values[["user_id", relation]])
            candidate_values["value"] = counts.reindex(keys, fill_value=0).to_numpy()
            aggregated = candidate_values.groupby("pair_index").value.agg(["sum", "max"])
            for stat in ("sum", "max"):
                output[f"ui_{relation}_{stat}_{suffix}"] = (
                    aggregated[stat].reindex(pairs.index, fill_value=0).astype(np.float32)
                )
            output[f"ui_{relation}_share_{suffix}"] = output[
                f"ui_{relation}_sum_{suffix}"
            ] / output[f"user_count_{suffix}"].clip(lower=1)
        return output

    def build(
        self, pairs: pd.DataFrame, config: Config
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        result = self.static(pairs)
        groups = {"static": [c for c in result if c not in ("user_id", "item_id")]}
        for kernel in [
            Kernel("window", None),
            Kernel("window", 7),
            Kernel("window", 30),
            *search_kernels(config),
        ]:
            block = self.aggregate(pairs, kernel)
            groups[kernel.name] = block.columns.tolist()
            result = pd.concat([result, block], axis=1)
        for window in (7, 30):
            result[f"item_velocity_{window}"] = result[f"item_count_window{window}"] / result[
                "item_count_all"
            ].clip(lower=1)
            result[f"user_velocity_{window}"] = result[f"user_count_window{window}"] / result[
                "user_count_all"
            ].clip(lower=1)
        groups["static"].extend(
            [f"{entity}_velocity_{w}" for w in (7, 30) for entity in ("item", "user")]
        )
        manual = groups["static"] + groups["all"] + groups["window7"] + groups["window30"]
        groups["manual"] = manual
        return result, groups
