"""Matched-parameter candidate-policy control for the second study."""

from pathlib import Path

import numpy as np
import pandas as pd

from rec.experiment import fit_model, score_model
from rec.metrics import summarize, user_metrics
from rec.protocol import Config
from rec.io import write_json


def original_pool(frame: pd.DataFrame, budget: int = 120) -> pd.DataFrame:
    # Positive RRF score means nomination by an original source. Because extras are
    # sorted by the same fusion and tie break, its first budget rows recover old RRF.
    mask = (frame.retrieval_rank <= budget) & (frame.retrieval_rrf > 0)
    return frame.loc[mask].reset_index(drop=True)


def score_original_pool(
    training: list[pd.DataFrame],
    frame: pd.DataFrame,
    recipe: dict,
    config: Config,
    output: Path,
    truth: pd.DataFrame | None = None,
):
    output.mkdir(parents=True, exist_ok=True)
    train = pd.concat([original_pool(block) for block in training], ignore_index=True)
    slate = original_pool(frame)
    metrics = []
    for name in ("manual", "hybrid"):
        spec = recipe[name]
        model = fit_model(train, spec["columns"], config, spec["depth"], 17, spec["iterations"])
        scores = score_model(model, slate, spec["columns"])
        method = f"{name}_original_pool"
        model.save_model(str(output / f"{method}_seed17.cbm"))
        slate[["user_id", "item_id"]].assign(score=scores).to_parquet(
            output / f"{method}_seed17_predictions.parquet", index=False
        )
        if truth is not None:
            per_user = user_metrics(slate, scores, truth, np.sort(truth.user_id.unique()))
            per_user.to_parquet(output / f"{method}_users.parquet", index=False)
            metrics.append({"method": method, "seed": 17, **summarize(per_user)})
    write_json(
        output / "policy.json",
        {
            "policy": "original RRF <=120",
            "seed": 17,
            "parameters": "held identical to corresponding union-pool recipe",
            "metrics": metrics,
        },
    )
