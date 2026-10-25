"""Raw-data retention, time invariance and replay of every frozen forecast."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import research_task2 as study
from rec.io import file_hash, write_json
from rec.task2_data import TRAIN_END, encode, prefix, read_json, vocabulary
from rec.task2_models import load_policy, seed_everything


def row_digest(path, users=None):
    digest = hashlib.sha256()
    count = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=131072, use_threads=False):
        frame = batch.to_pandas()
        if users is not None:
            frame = frame.loc[frame.uid.isin(users)]
        # Identical six-column, little-endian row representation independent of
        # Parquet row-group boundaries and compression settings.
        values = frame.to_numpy(dtype="<i8")
        digest.update(values.tobytes(order="C"))
        count += len(frame)
    return count, digest.hexdigest()


def audit(root, output, replay=True):
    sample = read_json(root / "sample.json")
    users = sample["sampled_user_ids"]
    original = row_digest(Path("data/yambda/listens.parquet"), users)
    retained = row_digest(root / "sampled_events.parquet")
    if original != retained or retained[0] != sample["rows"]:
        raise AssertionError("Selected users' complete source rows not retained")
    previous = set()
    for version in ("v3", "v4"):
        previous.update(read_json(f"reports/{version}/evidence/sample.json")["sampled_user_ids"])
    assert not previous.intersection(users)
    assert not set(sample["development_users"]).intersection(sample["external_users"])
    frame = pd.read_parquet(root / "sampled_events.parquet")
    catalog = np.load(root / "catalog.npz")
    mutated = frame.copy()
    future = mutated.timestamp >= TRAIN_END
    mutated.loc[future, "item_id"] = 4000000000
    mutated.loc[future, "track_length_seconds"] = 999999
    mutated.loc[future, "played_ratio_pct"] = 0
    calculated = vocabulary(mutated)
    for key, values in zip(("items", "lengths", "counts"), calculated, strict=True):
        np.testing.assert_array_equal(catalog[key], values)
    meta = pd.read_parquet(root / "test_queries.parquet")
    contexts = np.load(root / "test_x.npy")
    tested = 0
    for row in meta.drop_duplicates("uid").head(64).itertuples():
        user = frame.loc[frame.uid == row.uid].sort_values("timestamp", kind="stable")
        changed = user.copy()
        changed.loc[changed.timestamp >= row.timestamp, "item_id"] = 4000000000
        past = changed.loc[changed.timestamp < row.timestamp]
        reconstructed = prefix(encode(past.item_id, catalog["items"]), len(past))
        np.testing.assert_array_equal(contexts[row.Index], reconstructed)
        tested += 1
    del frame, mutated
    replays = []
    if replay:
        seed_everything(17)
        for path in sorted((root / "test_predictions").glob("*.npz")):
            if path.stem in ("popularity", "repeat", "markov"):
                continue
            model = load_policy(root / "models" / (path.stem + ".pt"), study.DEVICE)
            with study.amp(), torch.no_grad():
                top = study.recommend(model, contexts)
            np.testing.assert_array_equal(top, np.load(path)["top"])
            replays.append(
                {
                    "model": path.stem,
                    "rows": len(contexts),
                    "exact_top10": True,
                    "model_sha256": file_hash(root / "models" / (path.stem + ".pt")),
                }
            )
            del model
            torch.cuda.empty_cache()
    result = {
        "source_rows": original[0],
        "canonical_rows_sha256": original[1],
        "retained_all_source_rows": True,
        "no_old_user_overlap": True,
        "catalog_future_mutation_exact": True,
        "query_future_mutations_exact": tested,
        "prediction_replays": replays,
    }
    write_json(output, result)
    print(
        f"All {original[0]:,} source rows retained; {tested} time checks; {len(replays)} exact replays."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/task2_final"))
    parser.add_argument("--output", type=Path, default=Path("reports/task2/evidence/audit.json"))
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    audit(args.root, args.output, not args.no_replay)
