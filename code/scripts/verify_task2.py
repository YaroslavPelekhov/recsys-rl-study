"""Recalculate published Task 2 evidence without source Yambda or PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rec.io import file_hash
from rec.task2_data import SPECIAL, read_json
from rec.task2_metrics import (
    DIAGNOSTICS,
    behavior_summary,
    paired_interval,
    ranking_rows,
    reward_metrics,
    user_macro,
)


def close(left, right, path="result"):
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"Keys differ: {path}")
        for key in left:
            close(left[key], right[key], f"{path}/{key}")
    elif left is None:
        assert right is None, path
    else:
        np.testing.assert_allclose(left, right, atol=1e-10, rtol=1e-7, err_msg=path)


def verify(directory):
    evidence = directory / "evidence"
    checksums = read_json(evidence / "checksums.json")
    for name, expected in checksums.items():
        if file_hash(evidence / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    seal = read_json(evidence / "prediction_seal.json")
    opened = read_json(evidence / "test_opened.json")
    assert file_hash(evidence / "frozen.json") == seal["frozen_sha256"]
    assert file_hash(evidence / "prediction_seal.json") == opened["prediction_seal_sha256"]
    assert file_hash(evidence / "labels.parquet") == opened["labels_sha256"]
    for name, expected_hash in seal["files"].items():
        source = Path(name)
        if source.suffix == ".npz":
            assert file_hash(evidence / "predictions" / source.name) == expected_hash
        elif source.name == "test_queries.parquet":
            assert file_hash(evidence / "queries.parquet") == expected_hash
        elif source.name == "test_x.npy":
            stream = io.BytesIO()
            np.save(stream, np.load(evidence / "contexts.npz")["x"])
            assert hashlib.sha256(stream.getvalue()).hexdigest() == expected_hash
    frozen = read_json(evidence / "frozen.json")
    project = Path(__file__).resolve().parents[1]
    for name, expected_hash in frozen["files"].items():
        if name.startswith(("src/", "scripts/", "docs/")):
            assert file_hash(project / name) == expected_hash, f"Frozen source changed: {name}"
    meta, labels = (
        pd.read_parquet(evidence / "queries.parquet"),
        pd.read_parquet(evidence / "labels.parquet"),
    )
    expected = read_json(evidence / "results.json")
    per_query = pd.read_parquet(evidence / "query_metrics.parquet")
    grouped, behavior, seeds = {}, {}, {}
    for path in sorted((evidence / "predictions").glob("*.npz")):
        arrays = np.load(path)
        if (arrays["top"] < SPECIAL).any():
            raise AssertionError("Noncatalog action recommended")
        values = ranking_rows(arrays["top"], labels.token)
        actual_rows = per_query.loc[per_query.model == path.stem].sort_values("query")
        np.testing.assert_allclose(values, actual_rows[["ndcg", "recall", "map"]], atol=1e-12)
        parts = path.stem.rsplit("_", 1)
        method = parts[0] if parts[-1].isdigit() else path.stem
        grouped.setdefault(method, []).append(values.to_numpy())
        seeds[path.stem] = user_macro(values, meta.uid)
        if "diagnostics" in arrays:
            behavior.setdefault(method, []).append(
                behavior_summary(
                    arrays["top"],
                    pd.DataFrame(arrays["diagnostics"], columns=DIAGNOSTICS),
                    meta.uid,
                    16384,
                )
            )
    assert len(grouped) == len(expected["methods"])
    means = {m: np.mean(v, 0) for m, v in grouped.items()}
    for method, values in means.items():
        report = expected["methods"][method]
        close(
            user_macro(pd.DataFrame(values, columns=["ndcg", "recall", "map"]), meta.uid),
            {m: report[m] for m in ("ndcg", "recall", "map")},
            method,
        )
        close(paired_interval(values[:, 0], means["sft"][:, 0], meta.uid), report["vs_sft"], method)
        for name, mask in (
            ("positive", labels.ratio >= 0.5),
            ("novel", ~labels.repeat_full),
            ("repeat", labels.repeat_full),
            ("oov", labels.token < SPECIAL),
        ):
            actual = {
                "queries": int(mask.sum()),
                **user_macro(
                    pd.DataFrame(values[mask], columns=["ndcg", "recall", "map"]), meta.uid[mask]
                ),
            }
            close(actual, report[name], f"{method}/{name}")
        if method in behavior:
            close(
                {k: float(np.mean([v[k] for v in behavior[method]])) for k in behavior[method][0]},
                report["behavior"],
                method,
            )
    for entry in expected["seeds"]:
        close(
            seeds[entry["model"]], {k: entry[k] for k in ("ndcg", "recall", "map")}, entry["model"]
        )
    close(
        paired_interval(means["anchored_b0.1_h5"][:, 0], means["ensemble_b0.1_h5"][:, 0], meta.uid),
        expected["primary_anchored_minus_ensemble"],
        "primary",
    )
    reward = np.load(evidence / "reward_predictions.npz")
    for row in read_json(evidence / "reward_metrics.json")["metrics"]:
        split, model = row["split"], row["model"]
        truth = reward[f"{split}_truth"]
        if model in ("ensemble_catalog", "ensemble_oov"):
            mask = reward[f"{split}_actions"] >= SPECIAL
            if model.endswith("oov"):
                mask = ~mask
            pred, truth = reward[f"{split}_ensemble"][mask], truth[mask]
        else:
            pred = reward[f"{split}_{model}"]
        close(
            reward_metrics(truth, pred),
            {k: v for k, v in row.items() if k not in ("split", "model")},
            model,
        )
    rollouts = pd.read_parquet(evidence / "rollouts.parquet")
    first = rollouts.loc[rollouts.step == 1].pivot(
        index=["method", "seed", "uid"], columns="mode", values="action"
    )
    assert (first.evolving == first.frozen).all()
    assert (rollouts.loc[rollouts["mode"] == "frozen", "context_shift"] == 0).all()
    assert (rollouts.loc[rollouts.step == 1, "context_shift"] == 0).all()
    assert (rollouts.loc[rollouts.step == 1, "duplicate"] == 0).all()
    audit = read_json(evidence / "audit.json")
    assert audit["retained_all_source_rows"] and audit["no_old_user_overlap"]
    assert all(row["exact_top10"] for row in audit["prediction_replays"])
    return {
        "status": "passed",
        "checksum_files": len(checksums),
        "prediction_arrays": len(seeds),
        "queries": len(meta),
        "users": meta.uid.nunique(),
        "reward_metric_rows": len(read_json(evidence / "reward_metrics.json")["metrics"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/task2"))
    args = parser.parse_args()
    print(json.dumps(verify(args.report_dir), indent=2))
