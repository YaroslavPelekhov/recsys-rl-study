"""Independent compact-evidence reconstruction for the final factorial study."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rec.io import file_hash
from rec.metrics import paired_bootstrap, summarize


def verify_v4_evidence(report):
    evidence = Path(report) / "evidence"

    def read(name):
        return json.loads((evidence / name).read_text("utf-8"))

    checksums = read("checksums.json")
    for name, digest in checksums.items():
        path = (evidence / name).resolve()
        if not path.is_relative_to(evidence.resolve()) or file_hash(path) != digest:
            raise ValueError(f"V4 checksum mismatch: {name}")
    complete, protocol, recipe = (
        read("complete.json"),
        read("protocol.json"),
        read("frozen_recipe.json"),
    )
    for name, key in (("results.json", "results_sha256"), ("frozen_recipe.json", "recipe_sha256")):
        if file_hash(evidence / name) != complete[key]:
            raise ValueError("Completion hash mismatch")
    if recipe["external_labels_used"] or recipe["protocol_sha256"] != file_hash(
        evidence / "protocol.json"
    ):
        raise ValueError("Invalid recipe freeze")
    for name, key in (
        ("protocol_frozen.md", "document_sha256"),
        ("sample.json", "sample_sha256"),
        ("source_manifest.json", "source_manifest_sha256"),
    ):
        if file_hash(evidence / name) != protocol[key]:
            raise ValueError("Frozen source input mismatch")
    sample, excluded = read("sample.json"), read("excluded_sample.json")
    if len(set(sample["sampled_user_ids"])) != 1000 or set(sample["sampled_user_ids"]) & set(
        excluded["sampled_user_ids"]
    ):
        raise ValueError("External cohort is not disjoint")
    if file_hash(evidence / "excluded_sample.json") != sample["excluded_sample_sha256"]:
        raise ValueError("Excluded cohort changed")
    ready = read("predictions_ready.json")
    if ready["targets_exist"] or ready["recipe_sha256"] != complete["recipe_sha256"]:
        raise ValueError("Invalid pre-target forecast seal")
    audit = pd.read_csv(evidence / "fold_audit.csv")
    if not (audit.history_max_timestamp < audit.history_cutoff_exclusive).all():
        raise ValueError("Future record in history")
    if audit[audit.day.isin([298, 299])].targets_constructed.any():
        raise ValueError("Test features used target sets")
    result = read("results.json")
    blocks = {}
    for row in result["metrics"]:
        path = evidence / f"{row['method']}_seed{row['seed']}_day{row['day']}_users.parquet"
        table = pd.read_parquet(path)
        if table.user_id.duplicated().any() or not table.ndcg.between(0, 1 + 1e-12).all():
            raise ValueError("Invalid metric population")
        for metric, actual in summarize(table).items():
            if not np.isclose(actual, row[metric], rtol=0, atol=1e-12):
                raise ValueError("Daily metric mismatch")
        blocks.setdefault(row["method"], []).append(table)
    means = {}
    for method, tables in blocks.items():
        user_day = (
            pd.concat(tables).groupby(["user_id", "day"], as_index=False).mean(numeric_only=True)
        )
        means[method] = (
            user_day.drop(columns="day").groupby("user_id", as_index=False).mean(numeric_only=True)
        )
        pd.testing.assert_frame_equal(
            means[method], pd.read_parquet(evidence / f"{method}_users.parquet"), check_exact=True
        )
        for metric, actual in summarize(means[method]).items():
            if not np.isclose(actual, result["overall"][method][metric], rtol=0, atol=1e-12):
                raise ValueError("Overall metric mismatch")
    if paired_bootstrap(means["G0"], means["A1"], 424242) != result["primary"]:
        raise ValueError("Primary CI mismatch")
    for name, expected in result["vs_manual"].items():
        if paired_bootstrap(means["manual"], means[name], 424242) != expected:
            raise ValueError("Manual comparison mismatch")
    for label, expected in result["factorial"].items():
        b, a = label.split("_minus_")
        if paired_bootstrap(means[a], means[b], 424242) != expected:
            raise ValueError("Factorial comparison mismatch")
    interaction = means["G0"][["user_id", "ndcg"]].copy()
    aligned = {name: table.set_index("user_id").ndcg for name, table in means.items()}
    interaction["ndcg"] = interaction.user_id.map(
        aligned["A1"] - aligned["A0"] - aligned["G1"] + aligned["G0"]
    )
    zero = interaction.copy()
    zero["ndcg"] = 0.0
    if paired_bootstrap(zero, interaction, 424242) != read("interaction.json"):
        raise ValueError("Factorial interaction mismatch")
    replay = read("replay_verification.json")
    if (
        replay["status"] != "verified"
        or len(replay["checks"]) != 66
        or len(ready["predictions"]) != 66
    ):
        raise ValueError("Incomplete forecast replay")
    for row in replay["checks"]:
        name = f"{row['method']}_seed{row['seed']}_day{row['day']}_predictions.parquet"
        if (
            row["max_absolute_error"] > 1e-12
            or ready["predictions"].get(name) != row["predictions_sha256"]
        ):
            raise ValueError("Forecast differs from pre-target seal")
    if len(replay["reconstruction_future_invariance"]) != 6 or any(
        c["max_absolute_error"] != 0 for c in replay["reconstruction_future_invariance"]
    ):
        raise ValueError("Reconstruction future invariant failed")
    if len(replay["full_candidate_future_invariance"]) != 2 or not all(
        c["all_columns_equal"] for c in replay["full_candidate_future_invariance"]
    ):
        raise ValueError("Candidate future invariant failed")
    return {
        "status": "verified",
        "files": len(checksums),
        "primary": result["primary"],
        "ndcg": {k: v["ndcg@10"] for k, v in result["overall"].items()},
    }
