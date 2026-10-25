"""Independent, raw-data-free verification of the second study's published evidence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rec.io import file_hash
from rec.metrics import paired_bootstrap


def verify_v2_evidence(report: Path) -> dict:
    evidence = report / "evidence"

    def read(name):
        return json.loads((evidence / name).read_text("utf-8"))

    checksums = read("checksums.json")
    for name, expected in checksums.items():
        path = (evidence / name).resolve()
        if not path.is_relative_to(evidence.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Evidence checksum mismatch: {name}")
    complete = read("complete.json")
    for name, key in (
        ("results.json", "results_sha256"),
        ("frozen_recipe.json", "frozen_recipe_sha256"),
    ):
        if file_hash(evidence / name) != complete[key]:
            raise ValueError(f"Completion mismatch: {name}")
    recipe = read("frozen_recipe.json")
    if (
        recipe["external_labels_used"]
        or file_hash(evidence / "reserved_users.json") != recipe["reserved_users_sha256"]
    ):
        raise ValueError("Frozen cohort/recipe mismatch")
    reserved = set(read("reserved_users.json")["user_ids"])
    for name in ("development_sample.json", "replication_sample.json"):
        if reserved & set(read(name)["sampled_user_ids"]):
            raise ValueError("External users overlap with development")
    audits = pd.read_csv(evidence / "fold_audit.csv")
    if not (pd.to_datetime(audits.history_max_timestamp) < pd.to_datetime(audits.cutoff)).all():
        raise ValueError("Non-historical feature data")
    external_test = audits[(audits.block == "external_cache") & (audits.cutoff == "2021-06-21")]
    if len(external_test) != 1 or external_test.iloc[0].future_labels_loaded:
        raise ValueError("External candidate frame was constructed with targets")
    result = read("results.json")
    scores = pd.DataFrame(result["metrics"])
    manual = pd.read_parquet(evidence / "manual_users.parquet")
    means = {}
    for method, rows in scores.groupby("method"):
        table = pd.read_parquet(evidence / f"{method}_users.parquet")
        if table.user_id.duplicated().any() or len(table) != result["target_users"]:
            raise ValueError(f"Invalid metric population: {method}")
        for metric in ("ndcg", "recall", "map", "hit_rate"):
            if not np.isclose(
                table[metric].mean(), rows[f"{metric}@10"].mean(), atol=1e-12, rtol=0
            ):
                raise ValueError(f"Metric mismatch: {method}/{metric}")
        if method != "manual":
            if paired_bootstrap(manual, table, 314159) != result["comparisons"][method]:
                raise ValueError(f"Bootstrap mismatch: {method}")
        means[method] = float(table.ndcg.mean())
    if result["primary"] != recipe["selected"]:
        raise ValueError("Primary method changed after freeze")
    for contrast in read("descriptive_comparisons.json"):
        first = pd.read_parquet(evidence / f"{contrast['first']}_users.parquet")
        second = pd.read_parquet(evidence / f"{contrast['second']}_users.parquet")
        if paired_bootstrap(first, second, 314159) != contrast["result"]:
            raise ValueError("Descriptive comparison mismatch")
    replay = read("replay_verification.json")
    if replay["status"] != "verified" or any(
        c["max_absolute_error"] > 1e-12 for c in replay["checks"]
    ):
        raise ValueError("Prediction replay not verified")
    return {
        "status": "verified",
        "files": len(checksums),
        "primary": result["primary"],
        "ndcg": means,
    }
