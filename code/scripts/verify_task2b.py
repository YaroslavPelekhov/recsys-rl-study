"""Replay new-cohort metrics, intervals and seals without data, weights or GPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rec.io import file_hash
from rec.task2_data import read_json
from rec.task2_metrics import paired_interval, ranking_rows, user_macro
from rec.task2b import item_indices


def close(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            close(a[key], b[key])
    elif isinstance(a, list):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    else:
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def verify(report=Path("reports/task2b")):
    evidence = report / "evidence"
    expected = read_json(evidence / "checksums.json")
    for name, digest in expected.items():
        assert file_hash(evidence / name) == digest, name
    seal, opened = read_json(evidence / "prediction_seal.json"), read_json(evidence / "opened.json")
    assert seal["labels_materialized"] is False
    assert opened["seal_sha256"] == file_hash(evidence / "prediction_seal.json")
    assert opened["labels_sha256"] == file_hash(evidence / "test_labels.parquet")
    assert seal["frozen_sha256"] == file_hash(evidence / "frozen.json")
    runtime = read_json(evidence / "runtime_fix.json")
    before = read_json(evidence / "frozen_before_runtime_fix.json")["files"]
    after = read_json(evidence / "frozen.json")["files"]
    assert set(before) == set(after)
    changed = [name.replace("\\", "/") for name in before if before[name] != after[name]]
    assert changed == ["scripts/evaluate_task2b.py"]
    previous_source = next(
        digest for name, digest in before.items() if name.replace("\\", "/") == changed[0]
    )
    assert file_hash(evidence / "evaluate_before_runtime_fix.py") == previous_source
    assert runtime["labels_opened"] is False and runtime["model_or_settings_changed"] is False
    for name, digest in runtime["prediction_sha_before"].items():
        assert file_hash(evidence / "test_predictions" / name) == digest
    for name, digest in seal["files"].items():
        relative = Path(name.replace("\\", "/"))
        candidate = (
            evidence
            / ("test_predictions" if "test_predictions" in relative.parts else "")
            / relative.name
        )
        assert file_hash(candidate) == digest, name
    # Published source retains the frozen implementation. Local model files are
    # intentionally not required for the compact CPU-only replay.
    for name, digest in read_json(evidence / "frozen.json")["files"].items():
        p = Path(name.replace("\\", "/"))
        if p.parts[0] in ("src", "scripts", "docs"):
            assert file_hash(p) == digest, name
    meta = pd.read_parquet(evidence / "test_queries.parquet")
    truth = pd.read_parquet(evidence / "test_labels.parquet")
    result = read_json(evidence / "results.json")
    cat = dict(np.load(evidence / "catalog.npz"))
    target = item_indices(truth.item_id.to_numpy(), cat["items"])
    np.testing.assert_array_equal(target, truth.item_index)
    data = dict(np.load(evidence / "test.npz"))
    assert np.all(data["target"] == -5)
    np.testing.assert_array_equal(data["users"], meta.uid)
    frames = []
    for path in sorted((evidence / "test_predictions").glob("*.npz")):
        pred = dict(np.load(path))
        assert pred["top"].shape == (len(meta), 10)
        assert (pred["top"] >= 0).all() and (pred["top"] < len(cat["items"])).all()
        assert all(len(np.unique(row)) == 10 for row in pred["top"])
        frame = ranking_rows(pred["top"], target)
        for key in ("reward", "repeat", "duration"):
            frame[key] = pred[key]
        frame["uid"], frame["query"] = meta.uid, np.arange(len(meta))
        parts = path.stem.rsplit("_", 1)
        frame["method"] = parts[0] if parts[-1] in ("17", "42", "73") else path.stem
        frame["seed"] = int(parts[-1]) if parts[-1] in ("17", "42", "73") else 0
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    published = pd.read_parquet(evidence / "query_metrics.parquet")
    keys = ["method", "seed", "query"]
    pd.testing.assert_frame_equal(
        rows.sort_values(keys).reset_index(drop=True)[published.columns],
        published.sort_values(keys).reset_index(drop=True),
        check_dtype=False,
        atol=1e-12,
        rtol=0,
    )
    means = rows.groupby(["method", "query", "uid"], as_index=False).mean(numeric_only=True)
    for method, frame in means.groupby("method"):
        close(
            user_macro(frame[["ndcg", "recall", "map", "reward", "repeat", "duration"]], frame.uid),
            result["methods"][method],
        )
    for name, published_ci in result["comparisons"].items():
        a, b = name.split("-minus-")
        av = means.loc[means.method == a].sort_values("query").ndcg
        bv = means.loc[means.method == b].sort_values("query").ndcg
        close(paired_interval(av, bv, meta.uid, seed=161804), published_ci)
    rm = dict(np.load(evidence / "rm_test.npz"))
    actual, pred = rm["truth"], rm["predictions"].mean(0)
    np.testing.assert_array_equal(actual, truth.ratio)
    close(float(np.mean((pred - actual) ** 2)), result["rm"]["mse"])
    close(float(np.mean((rm["constant"] - actual) ** 2)), result["rm"]["constant_mse"])
    close(float(np.abs(pred - actual).mean()), result["rm"]["mae"])
    for name, score in result["rm"]["controls"].items():
        close(float(np.mean((rm[name] - actual) ** 2)), score)
    for baseline, key in (
        (rm["constant"], "mse_difference_ci95"),
        (rm["state_only"], "minus_state_only_ci95"),
    ):
        errors = (
            pd.DataFrame(dict(uid=meta.uid, error=(pred - actual) ** 2 - (baseline - actual) ** 2))
            .groupby("uid")
            .error.agg(["sum", "size"])
        )
        rng = np.random.default_rng(161804)
        draw = rng.integers(len(errors), size=(3000, len(errors)))
        values = errors["sum"].to_numpy()[draw].sum(1) / errors["size"].to_numpy()[draw].sum(1)
        close(np.quantile(values, [0.025, 0.975]).tolist(), result["rm"][key])
    head = item_indices(truth.item_id.to_numpy(), cat["head"]) >= 0
    support = head | ((target >= 0) & (data["history"] == target[:, None]).any(1))
    close(float(head.mean()), result["head_coverage"])
    close(float(support.mean()), result["support_coverage"])
    close(float((target >= 0).mean()), result["full_catalog_coverage"])
    assert len(meta) == opened["queries"] == result["queries"]
    assert meta.uid.nunique() == opened["users"] == result["users"]
    sample = read_json(evidence / "sample.json")
    assert set(meta.uid) <= set(sample["external_users"])
    for name in ("v3", "v4", "task2"):
        old = read_json(Path("reports") / name / "evidence/sample.json")
        assert not set(sample["external_users"]) & set(old["sampled_user_ids"])
    audit = read_json(evidence / "audit.json")
    assert audit["retained_rows"] == sample["rows"] and audit["labels_opened"] is False
    from analyze_task2b import analyze

    assert analyze(evidence) == read_json(evidence / "descriptive.json")
    return dict(
        status="verified",
        queries=len(meta),
        users=meta.uid.nunique(),
        prediction_arrays=len(frames),
        evidence_files=len(expected),
        primary=result["comparisons"]["gated-minus-continue_ce"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/task2b"))
    print(verify(parser.parse_args().report_dir))
