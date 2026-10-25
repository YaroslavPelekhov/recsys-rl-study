"""Full-row preservation and future-mutation checks before opening new test."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from rec.io import file_hash, write_json
from rec.task2_data import histories, read_json
from rec.task2b import record


def digest(path, users):
    result, rows = hashlib.sha256(), 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, use_threads=False):
        part = batch.to_pandas()
        part = part.loc[part.uid.isin(users)]
        result.update(np.ascontiguousarray(part.to_numpy(dtype=np.uint64)).tobytes())
        rows += len(part)
    return result.hexdigest(), rows


def run():
    root = Path("runs/task2b")
    sample = read_json(root / "sample.json")
    users = sample["external_users"]
    source = Path("data/yambda/listens.parquet")
    expected = read_json("runs/task2_final/sample.json")["source_sha256"]
    assert file_hash(source) == expected == sample["source_sha256"]
    original, n = digest(source, users)
    retained, m = digest(root / "sampled_events.parquet", users)
    assert original == retained and n == m == sample["rows"]
    excluded = [
        read_json(f"reports/{v}/evidence/sample.json")["sampled_user_ids"] for v in ("v3", "v4")
    ]
    excluded += [read_json("runs/task2_final/sample.json")["sampled_user_ids"]]
    assert not set(users) & set(np.concatenate(excluded))
    cat = dict(np.load(root / "catalog.npz"))
    frame = pd.read_parquet(root / "sampled_events.parquet")
    history = histories(frame, cat["head"])
    rng = np.random.default_rng(161805)
    tests = 0
    # Prefix examples in the new cohort, without reading evaluation targets.
    for uid in rng.choice(users, 64, replace=False):
        h = history[int(uid)]
        pos = int(np.searchsorted(h["t"], 280 * 86400))
        if pos == 0 or pos >= len(h["t"]):
            continue
        first = record(h, pos, cat["items"], cat["head"])
        end = np.searchsorted(h["t"], h["t"][pos], side="left")
        modified = {k: v.copy() for k, v in h.items()}
        modified["item"][end:] = 987654321
        modified["ratio"][end:] = 0
        modified["organic"][end:] = 1
        second = record(modified, pos, cat["items"], cat["head"])
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)
        tests += 1
    write_json(
        root / "audit.json",
        dict(
            source_sha256=expected,
            canonical_rows_sha256=original,
            retained_rows=m,
            raw_columns=6,
            disjoint_from_previous=True,
            future_mutations=tests,
            labels_opened=(root / "test_labels.parquet").exists(),
        ),
    )


if __name__ == "__main__":
    run()
