from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pandas as pd


@dataclass(frozen=True)
class Config:
    n_users: int = 10000
    sample_seed: int = 2026
    model_seeds: tuple[int, ...] = (17, 42, 73)
    horizon_days: int = 7
    train_snapshots: int = 4
    candidates: int = 120
    retrieval_per_source: int = 80
    neighbors: int = 50
    max_history: int = 50
    iterations: int = 350
    depths: tuple[int, ...] = (5, 7)
    learning_rate: float = 0.05
    threads: int = 12
    search_steps: int = 2
    search_windows: tuple[int, ...] = (2, 3, 5, 10, 14, 21, 45, 60)
    search_half_lives: tuple[int, ...] = (3, 10, 30)
    min_gain: float = 0.0005
    bootstrap_samples: int = 3000
    k: int = 10

    def __post_init__(self) -> None:
        for name in (
            "n_users",
            "horizon_days",
            "train_snapshots",
            "candidates",
            "retrieval_per_source",
            "neighbors",
            "max_history",
            "iterations",
            "threads",
            "bootstrap_samples",
            "k",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not self.model_seeds or not self.depths:
            raise ValueError("Specify model seeds and depths")
        if self.search_steps < 0 or self.min_gain < 0:
            raise ValueError("Invalid search budget")
        if any(w <= 0 for w in (*self.search_windows, *self.search_half_lives)):
            raise ValueError("Temporal kernels must be positive")
        if self.k > self.candidates:
            raise ValueError("k exceeds candidate budget")

    @classmethod
    def load(cls, path: Path) -> Config:
        return cls(**json.loads(path.read_text("utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Fold:
    name: str
    cutoff: pd.Timestamp
    end: pd.Timestamp


def temporal_folds(end_exclusive: pd.Timestamp, config: Config) -> list[Fold]:
    """Four training weeks, search week, untouched confirmation week, final test week."""
    step = pd.Timedelta(days=config.horizon_days)
    total = config.train_snapshots + 3
    names = [f"train_{i}" for i in range(config.train_snapshots)] + ["search", "confirm", "test"]
    return [
        Fold(name, end_exclusive - (total - i) * step, end_exclusive - (total - i - 1) * step)
        for i, name in enumerate(names)
    ]


def history_before(events: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return events.loc[events.timestamp < cutoff].copy()


def future_truth(events: pd.DataFrame, fold: Fold, history: pd.DataFrame) -> pd.DataFrame:
    """All future novel items, including ones absent from the historical catalog."""
    future = events.loc[
        (events.timestamp >= fold.cutoff) & (events.timestamp < fold.end), ["user_id", "item_id"]
    ].drop_duplicates()
    seen = history[["user_id", "item_id"]].drop_duplicates()
    return (
        future.merge(seen.assign(_seen=1), how="left", on=["user_id", "item_id"])
        .query("_seen.isna()")
        .drop(columns="_seen")
        .reset_index(drop=True)
    )
