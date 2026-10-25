from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rec.io import file_hash
from rec.metrics import paired_bootstrap


def verify_evidence(report: Path) -> dict:
    """Recompute published averages and paired intervals without fitting or reading raw data."""
    evidence = report / "evidence"
    if (evidence / "study_v4.json").exists():
        from rec.verification_v4 import verify_v4_evidence

        return verify_v4_evidence(report)
    if (evidence / "study_v3_support.json").exists():
        from rec.verification_support import verify_support_evidence

        return verify_support_evidence(report)
    if (evidence / "study_v3.json").exists():
        from rec.verification_v3 import verify_v3_evidence

        return verify_v3_evidence(report)
    if (evidence / "study_v2.json").exists():
        from rec.verification_v2 import verify_v2_evidence

        return verify_v2_evidence(report)
    checksums = json.loads((evidence / "checksums.json").read_text("utf-8"))
    for name, expected in checksums.items():
        path = (evidence / name).resolve()
        if not path.is_relative_to(evidence.resolve()):
            raise ValueError(f"Invalid evidence path: {name}")
        if file_hash(path) != expected:
            raise ValueError(f"Evidence checksum mismatch: {name}")
    result = json.loads((evidence / "results.json").read_text("utf-8"))
    config = json.loads((evidence / "config.json").read_text("utf-8"))
    complete = json.loads((evidence / "complete.json").read_text("utf-8"))
    if file_hash(evidence / "results.json") != complete["results_sha256"]:
        raise ValueError("Completion manifest does not match results")
    if file_hash(evidence / "frozen_recipe.json") != complete["recipe_sha256"]:
        raise ValueError("Completion manifest does not match frozen recipe")
    results = pd.DataFrame(result["test"])
    means = {}
    for method, rows in results.groupby("method"):
        frame = pd.read_parquet(evidence / f"{method}_users.parquet")
        if frame.user_id.duplicated().any():
            raise ValueError(f"Duplicate metric users: {method}")
        if not frame.ndcg.between(0, 1 + 1e-12).all():
            raise ValueError(f"Out-of-bounds NDCG: {method}")
        for metric in ("ndcg", "recall", "map", "hit_rate"):
            expected = rows[f"{metric}@{config['k']}"].mean()
            if not np.isclose(frame[metric].mean(), expected, rtol=0, atol=1e-12):
                raise ValueError(f"Metric mismatch: {method}/{metric}")
        means[method] = float(frame.ndcg.mean())
    manual = pd.read_parquet(evidence / "manual_users.parquet")
    for method, expected in result["paired_comparisons_vs_manual"].items():
        frame = pd.read_parquet(evidence / f"{method}_users.parquet")
        actual = paired_bootstrap(manual, frame, config["sample_seed"], config["bootstrap_samples"])
        if actual != expected:
            raise ValueError(f"Paired comparison mismatch: {method}")
    return {"status": "verified", "files": len(checksums), "ndcg": means}
