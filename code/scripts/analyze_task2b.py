"""Descriptive post-test audit; never changes policies, rewards or candidates."""

from pathlib import Path

import numpy as np
import pandas as pd

from rec.io import file_hash, write_json
from rec.task2_metrics import user_macro


def analyze(evidence=Path("reports/task2b/evidence"), create_lengths=False):
    catalog = dict(np.load(evidence / "catalog.npz"))
    destination = evidence / "duration270.npz"
    if create_lengths:
        frame = pd.read_parquet(
            "runs/task2_final/sampled_events.parquet",
            columns=["timestamp", "item_id", "track_length_seconds"],
        )
        lengths = (
            frame.loc[frame.timestamp < 270 * 86400]
            .groupby("item_id")
            .track_length_seconds.median()
            .reindex(catalog["items"])
            .to_numpy(np.float32)
        )
        np.savez_compressed(destination, lengths=lengths)
    lengths = np.load(destination)["lengths"]
    assert np.isfinite(lengths).all() and (lengths > 0).all()
    meta = pd.read_parquet(evidence / "test_queries.parquet")
    rows = []
    for path in sorted((evidence / "test_predictions").glob("*.npz")):
        with np.load(path) as data:
            duration, unknown = lengths[data["top"][:, 0]], data["duration"] == 0
        parts = path.stem.rsplit("_", 1)
        name = parts[0] if parts[-1] in ("17", "42", "73") else path.stem
        rows.append(
            pd.DataFrame(
                dict(
                    method=name,
                    query=np.arange(len(meta)),
                    uid=meta.uid,
                    duration270=duration,
                    missing200=unknown.astype(float),
                )
            )
        )
    average = (
        pd.concat(rows).groupby(["method", "query", "uid"], as_index=False).mean(numeric_only=True)
    )
    result = {
        name: user_macro(frame[["duration270", "missing200"]], frame.uid)
        for name, frame in average.groupby("method")
    }
    return dict(
        status="descriptive_post_test_no_model_change",
        explanation="Frozen duration diagnostic used pre200 item metadata with unknown=0. This audit uses train-only median through269; no policy/RM/candidate change.",
        methods=result,
        length_sha256=file_hash(destination),
    )


if __name__ == "__main__":
    import sys

    evidence = Path("reports/task2b/evidence")
    result = analyze(evidence, create_lengths="--create-lengths" in sys.argv)
    write_json(evidence / "descriptive.json", result)
    files = [p for p in evidence.rglob("*") if p.is_file() and p.name != "checksums.json"]
    write_json(
        evidence / "checksums.json",
        {p.relative_to(evidence).as_posix(): file_hash(p) for p in files},
    )
    print(result)
