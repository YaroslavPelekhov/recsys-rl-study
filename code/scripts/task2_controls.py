"""Compute the three fixed non-neural controls on Task 2 development queries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_task2 import simple_forecasts
from rec.io import write_json
from rec.task2_metrics import ranking_rows, user_macro

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/task2_final"))
    args = parser.parse_args()
    root = args.root
    (root / "search_predictions").mkdir(exist_ok=True)
    simple_forecasts(root, "search", np.load(root / "search_x.npy"), np.load(root / "catalog.npz"))
    labels = pd.read_parquet(root / "search_labels.parquet")
    meta = pd.read_parquet(root / "search_queries.parquet")
    result = {
        name: user_macro(
            ranking_rows(np.load(root / f"search_predictions/{name}.npz")["top"], labels.token),
            meta.uid,
        )
        for name in ("popularity", "repeat", "markov")
    }
    write_json(root / "search_controls.json", result)
    print(result)
