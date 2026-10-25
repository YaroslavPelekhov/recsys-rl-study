"""Raw-data-free verification of the explicitly secondary v3 support extension."""

import json
from pathlib import Path

import pandas as pd
import numpy as np

from rec.io import file_hash
from rec.metrics import paired_bootstrap, summarize


def verify_support_evidence(report):
    evidence = Path(report) / "evidence"

    def read(name):
        return json.loads((evidence / name).read_text("utf-8"))

    checksums = read("checksums.json")
    for name, digest in checksums.items():
        path = (evidence / name).resolve()
        if not path.is_relative_to(evidence.resolve()) or file_hash(path) != digest:
            raise ValueError(f"Support evidence checksum mismatch: {name}")
    complete = read("complete.json")
    if (
        file_hash(evidence / "results.json") != complete["results_sha256"]
        or file_hash(evidence / "frozen_recipe.json") != complete["recipe_sha256"]
    ):
        raise ValueError("Support completion mismatch")
    ready, protocol, recipe = (
        read("predictions_ready.json"),
        read("protocol.json"),
        read("frozen_recipe.json"),
    )
    if (
        ready["main_test_targets_exist"]
        or protocol["main_test_targets_exist"]
        or recipe["external_labels_used"]
    ):
        raise ValueError("Support pre-test target audit failed")
    if recipe["protocol_sha256"] != file_hash(evidence / "protocol.json"):
        raise ValueError("Support frozen protocol changed")
    if ready["recipe_sha256"] != complete["recipe_sha256"]:
        raise ValueError("Support predictions belong to another recipe")
    if protocol["sample_sha256"] != file_hash(evidence / "sample.json"):
        raise ValueError("Support user sample changed")
    if recipe["base_protocol_sha256"] != protocol["base_protocol_sha256"]:
        raise ValueError("Support parent protocol changed")
    if read("base_recipe.json")["protocol_sha256"] != protocol["base_protocol_sha256"]:
        raise ValueError("Support parent recipe uses another protocol")
    if ready["base_recipe_sha256"] != file_hash(evidence / "base_recipe.json"):
        raise ValueError("Parent recipe changed after support freeze")
    if protocol["amendment_sha256"] != file_hash(evidence / "amendment_frozen.md"):
        raise ValueError("Support amendment changed")
    manual = pd.read_parquet(evidence / "manual_users.parquet")
    corrected = pd.read_parquet(evidence / "support_users.parquet")
    result = read("results.json")
    blocks = []
    test_days = read("base_results.json")["days"]
    for row in result["metrics"]:
        table = pd.read_parquet(
            evidence / f"support_seed{row['seed']}_day{row['day']}_users.parquet"
        )
        for metric, value in summarize(table).items():
            if not np.isclose(value, row[metric], rtol=0, atol=1e-12):
                raise ValueError("Support daily metric mismatch")
        if row["day"] in test_days:
            blocks.append(table)
    user_day = pd.concat(blocks).groupby(["user_id", "day"], as_index=False).mean(numeric_only=True)
    calculated = (
        user_day.drop(columns="day").groupby("user_id", as_index=False).mean(numeric_only=True)
    )
    pd.testing.assert_frame_equal(calculated, corrected, check_exact=True)
    if paired_bootstrap(manual, corrected, 2028) != result["paired_vs_manual"]:
        raise ValueError("Support paired interval mismatch")
    replay = read("replay_verification.json")
    if (
        replay["status"] != "verified"
        or len(replay["checks"]) != 7
        or any(row["max_absolute_error"] != 0 for row in replay["checks"])
    ):
        raise ValueError("Incomplete support prediction replay")
    if len(replay["feature_replay"]) != 2 or any(
        row["max_absolute_error"] != 0 for row in replay["feature_replay"]
    ):
        raise ValueError("Support feature replay failed")
    for row in replay["checks"]:
        name = f"support_seed{row['seed']}_day{row['day']}_predictions.parquet"
        if ready["predictions"].get(name) != row["predictions_sha256"]:
            raise ValueError("Support replay differs from its pre-target prediction seal")
    return {
        "status": "verified",
        "files": len(checksums),
        "secondary_difference": result["paired_vs_manual"],
    }
