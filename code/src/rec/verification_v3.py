"""Recompute seed/day aggregation and paired uncertainty from compact evidence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rec.io import file_hash
from rec.metrics import paired_bootstrap, summarize


def verify_v3_evidence(report):
    evidence = Path(report) / "evidence"

    def read(name):
        return json.loads((evidence / name).read_text("utf-8"))

    checksums = read("checksums.json")
    for name, expected in checksums.items():
        path = (evidence / name).resolve()
        if not path.is_relative_to(evidence.resolve()) or file_hash(path) != expected:
            raise ValueError(f"V3 checksum mismatch: {name}")
    complete, recipe, protocol = (
        read("complete.json"),
        read("frozen_recipe.json"),
        read("protocol.json"),
    )
    for name, key in (("results.json", "results_sha256"), ("frozen_recipe.json", "recipe_sha256")):
        if file_hash(evidence / name) != complete[key]:
            raise ValueError(f"Completion mismatch: {name}")
    if recipe["external_labels_used"] or recipe["protocol_sha256"] != file_hash(
        evidence / "protocol.json"
    ):
        raise ValueError("Frozen protocol changed")
    sealed = read("frozen_inputs.json")
    implementation = read("final_implementation_sha256.json")
    if sealed["recipe_sha256"] != complete["recipe_sha256"] or any(
        implementation.get(name) != digest for name, digest in sealed["code"].items()
    ):
        raise ValueError("Core implementation differs from its pre-test seal")
    if (evidence / "orchestration_amendment.json").exists():
        amendment = read("orchestration_amendment.json")
        initial = read("frozen_inputs.initial.json")
        if (
            amendment["test_targets_exist"]
            or amendment["numerical_model_and_training_inputs_changed"]
            or amendment["recipe_sha256"] != complete["recipe_sha256"]
            or initial["inputs"] != sealed["inputs"]
            or initial["code"]["scripts/parallel_v3.py"] != amendment["old_orchestrator_sha256"]
            or sealed["code"]["scripts/parallel_v3.py"] != amendment["new_orchestrator_sha256"]
        ):
            raise ValueError("Invalid pre-target orchestration amendment")
    for name, key in (
        ("sample.json", "sample_sha256"),
        ("source_manifest.json", "source_manifest_sha256"),
        ("protocol_frozen.md", "protocol_sha256"),
    ):
        if file_hash(evidence / name) != protocol[key]:
            raise ValueError(f"Protocol input mismatch: {name}")
    sample = read("sample.json")
    if len(set(sample["sampled_user_ids"])) != sample["n_users"]:
        raise ValueError("User sampling manifest contains duplicate IDs")
    audit = pd.read_csv(evidence / "fold_audit.csv")
    if not (audit.history_max_timestamp < audit.history_cutoff_exclusive).all():
        raise ValueError("Future event in historical feature table")
    if not (audit.label_cutoff - audit.history_cutoff_exclusive == 1800).all():
        raise ValueError("Incorrect temporal gap")
    test_audit = audit[audit.day.isin(protocol["test"])]
    if len(test_audit) != 2 or test_audit.targets_constructed.any():
        raise ValueError("Test features constructed with target item sets")
    result = read("results.json")
    by_method = {}
    for row in result["metrics"]:
        name = f"{row['method']}_seed{row['seed']}_day{row['day']}_users.parquet"
        table = pd.read_parquet(evidence / name)
        if table.user_id.duplicated().any() or not table.ndcg.between(0, 1 + 1e-12).all():
            raise ValueError(f"Invalid user-day metrics: {name}")
        for metric, actual in summarize(table).items():
            if not np.isclose(actual, row[metric], rtol=0, atol=1e-12):
                raise ValueError(f"Daily metric mismatch: {name}/{metric}")
        by_method.setdefault(row["method"], []).append(table)
    means = {}
    for method, blocks in by_method.items():
        user_day = (
            pd.concat(blocks).groupby(["user_id", "day"], as_index=False).mean(numeric_only=True)
        )
        calculated = (
            user_day.drop(columns="day").groupby("user_id", as_index=False).mean(numeric_only=True)
        )
        saved = pd.read_parquet(evidence / f"{method}_users.parquet")
        pd.testing.assert_frame_equal(calculated, saved, check_exact=True)
        means[method] = saved
    for method, expected in result["comparisons_vs_manual"].items():
        if paired_bootstrap(means["manual"], means[method], 2028) != expected:
            raise ValueError(f"Paired interval mismatch: {method}")
    primary = paired_bootstrap(means["global"], means["conditional"], 2028)
    if primary != result["primary_conditional_minus_global"]:
        raise ValueError("Primary interval mismatch")
    replay = read("replay_verification.json")
    expected_checks = len(result["metrics"]) + len(read("confirm_results.json")["metrics"])
    if replay["status"] != "verified" or len(replay["checks"]) != expected_checks:
        raise ValueError("Incomplete prediction replay")
    if any(c["max_absolute_error"] > 1e-12 for c in replay["checks"]):
        raise ValueError("Prediction replay failed")
    if len(replay["future_invariance"]) != 2 or not all(
        c["all_columns_equal"] for c in replay["future_invariance"]
    ):
        raise ValueError("Future-invariance verification failed")
    return {
        "status": "verified",
        "files": len(checksums),
        "users": len(means["manual"]),
        "primary": primary,
        "ndcg": {name: float(data.ndcg.mean()) for name, data in means.items()},
    }
