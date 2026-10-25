"""Strictly historical item-token data for the second, independent study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from rec.io import environment, file_hash, write_json

DAY = 86400
TRAIN_END = 270 * DAY
CONTEXT = 64
SPECIAL = 3


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def context_memmap(root):
    """Read-only file mapping avoids repeatedly committing a 209 MiB array."""
    path = root / "rm_contexts.npy"
    if not path.exists():
        with np.load(root / "train.npz") as source:
            np.save(path, source["rm_x"])
    return np.load(path, mmap_mode="r")


def reserve(root: Path, source=Path("data/yambda/listens.parquet")):
    if (root / "sample.json").exists():
        raise FileExistsError("Cohort already reserved")
    manifest = read_json("data/yambda/source_manifest.json")
    if file_hash(source) != manifest["files"]["listens.parquet"]["sha256"]:
        raise ValueError("Yambda source checksum mismatch")
    population = np.unique(pq.read_table(source, columns=["uid"]).column("uid").to_numpy())
    excluded = np.concatenate(
        [read_json(f"reports/{v}/evidence/sample.json")["sampled_user_ids"] for v in ("v3", "v4")]
    )
    available = np.setdiff1d(population, excluded)
    selected = np.sort(np.random.default_rng(718281).choice(available, 1200, replace=False))
    shuffled = np.random.default_rng(718282).permutation(selected)
    # Bounded scanner: Arrow's threaded whole-file filter can materialize many
    # row groups concurrently on a desktop with unrelated applications open.
    pieces = []
    for batch in pq.ParquetFile(source).iter_batches(batch_size=131072, use_threads=False):
        part = batch.to_pandas()
        pieces.append(part.loc[part.uid.isin(selected)])
    frame = pd.concat(pieces, ignore_index=True)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / "sampled_events.parquet", index=False, compression="zstd")
    write_json(
        root / "sample.json",
        {
            "seed": 718281,
            "sampled_user_ids": selected.tolist(),
            "development_users": np.sort(shuffled[:800]).tolist(),
            "external_users": np.sort(shuffled[800:]).tolist(),
            "source_users": len(population),
            "available_users": len(available),
            "rows": len(frame),
            "all_source_rows_retained": True,
            "source_sha256": file_hash(source),
            "events_sha256": file_hash(root / "sampled_events.parquet"),
            "excluded_manifests": {
                v: file_hash(Path(f"reports/{v}/evidence/sample.json")) for v in ("v3", "v4")
            },
        },
    )
    write_json(
        root / "protocol.json",
        {
            "document_sha256": file_hash(Path("docs/TASK2_PROTOCOL.md")),
            "sample_sha256": file_hash(root / "sample.json"),
            "test_opened": False,
            "train_end": TRAIN_END,
        },
    )
    write_json(root / "environment.json", environment())
    return frame


def vocabulary(frame, limit=16384, cutoff=TRAIN_END):
    """Only prefix counts/metadata determine the shared recommendable catalog."""
    past = frame.loc[frame.timestamp < cutoff]
    counts = past.groupby("item_id").size().rename("count").reset_index()
    ranked = counts.sort_values(["count", "item_id"], ascending=[False, True]).head(limit)
    items = np.sort(ranked.item_id.to_numpy(dtype=np.int64))
    lengths = past.groupby("item_id").track_length_seconds.median().reindex(items)
    return (
        items,
        lengths.to_numpy(np.float32),
        counts.set_index("item_id").reindex(items)["count"].to_numpy(np.float32),
    )


def encode(items, catalog):
    ids = np.asarray(items)
    positions = np.searchsorted(catalog, ids)
    safe = positions.clip(0, len(catalog) - 1)
    return np.where(
        (positions < len(catalog)) & (catalog[safe] == ids), positions + SPECIAL, 1
    ).astype(np.int32)


def histories(frame, catalog):
    # Source order within a timestamp is retained only as context; never a target.
    ordered = frame.sort_values(["uid", "timestamp"], kind="stable")
    result = {}
    for uid, group in ordered.groupby("uid", sort=True):
        t = group.timestamp.to_numpy(np.int64)
        tokens = encode(group.item_id, catalog)
        ratio = np.clip(group.played_ratio_pct.to_numpy(np.float32) / 100, 0, 1)
        result[int(uid)] = {
            "t": t,
            "tokens": tokens,
            "ratio": ratio,
            "organic": group.is_organic.to_numpy(),
            "item": group.item_id.to_numpy(np.int64),
            "length": group.track_length_seconds.to_numpy(np.float32),
        }
    return result


def prefix(tokens, end, context=CONTEXT):
    values = tokens[max(0, end - context) : end]
    out = np.zeros(context, np.int32)
    if len(values):
        out[-len(values) :] = values
    else:
        out[-1] = 2
    return out


def return_labels(times):
    # Full [t+1h,t+24h) is observed only for examples ending before TRAIN_END-DAY.
    lo = np.searchsorted(times, times + 3600, side="left")
    hi = np.searchsorted(times, times + DAY, side="left")
    return (hi > lo).astype(np.float32)


def make_queries(history, users, days, with_labels=False):
    rows, contexts, targets = [], [], []
    for uid in users:
        h = history[int(uid)]
        times = h["t"]
        returns = return_labels(times) if with_labels else None
        unique = np.r_[True, np.diff(times) != 0] & np.r_[np.diff(times) != 0, True]
        for day in days:
            eligible = np.flatnonzero(
                (times >= day * DAY) & (times < (day + 1) * DAY) & unique & (h["organic"] == 0)
            )
            for pos in eligible[:4]:
                start = int(np.searchsorted(times, times[pos], side="left"))
                if start == 0:
                    continue
                rows.append((int(uid), int(times[pos]), int(pos), day))
                contexts.append(prefix(h["tokens"], start))
                if with_labels:
                    targets.append(
                        (
                            int(h["item"][pos]),
                            int(h["tokens"][pos]),
                            float(h["ratio"][pos]),
                            float(returns[pos]),
                        )
                    )
    meta = pd.DataFrame(rows, columns=["uid", "timestamp", "position", "day"])
    values = np.stack(contexts) if contexts else np.empty((0, CONTEXT), np.int32)
    labels = (
        pd.DataFrame(targets, columns=["item_id", "token", "ratio", "return24"])
        if with_labels
        else None
    )
    return meta, values, labels


def training_arrays(history):
    chunks, targets, valid, rm_contexts, rm_actions, rm_y, rm_users = [], [], [], [], [], [], []
    for user_number, (uid, h) in enumerate(history.items()):
        n = np.searchsorted(h["t"], TRAIN_END, side="left")
        tokens, times = h["tokens"][:n], h["t"][:n]
        for start in range(0, n - 1, CONTEXT):
            width = min(CONTEXT, n - 1 - start)
            x, y = np.zeros(CONTEXT, np.int32), np.full(CONTEXT, -100, np.int64)
            x[:width] = tokens[start : start + width]
            y[:width] = tokens[start + 1 : start + width + 1]
            keep = (times[start + 1 : start + width + 1] > times[start : start + width]) & (
                y[:width] >= SPECIAL
            )
            y[:width][~keep] = -100
            if keep.any():
                chunks.append(x)
                targets.append(y)
        unique = np.r_[True, np.diff(h["t"]) != 0] & np.r_[np.diff(h["t"]) != 0, True]
        positions = np.flatnonzero((h["t"] < TRAIN_END - DAY) & unique & (h["organic"] == 0))
        # Training-example cap is uniform per user; raw histories are untouched.
        rng = np.random.default_rng(1000 + int(uid))
        if len(positions) > 1500:
            positions = np.sort(rng.choice(positions, 1500, replace=False))
        returns = return_labels(h["t"])
        for pos in positions:
            end = np.searchsorted(h["t"], h["t"][pos], side="left")
            if end == 0:
                continue
            rm_contexts.append(prefix(h["tokens"], end))
            rm_actions.append(h["tokens"][pos])
            rm_y.append((h["ratio"][pos], returns[pos]))
            rm_users.append(user_number)
        valid.append(int(n))
    return {
        "x": np.stack(chunks),
        "y": np.stack(targets),
        "rm_x": np.stack(rm_contexts),
        "rm_a": np.asarray(rm_actions, np.int32),
        "rm_y": np.asarray(rm_y, np.float32),
        "rm_u": np.asarray(rm_users, np.int32),
        "train_sizes": np.asarray(valid, np.int64),
    }


def prepare(root):
    frame = pd.read_parquet(root / "sampled_events.parquet")
    catalog, lengths, counts = vocabulary(frame)
    np.savez_compressed(root / "catalog.npz", items=catalog, lengths=lengths, counts=counts)
    history = histories(frame, catalog)
    train = training_arrays(history)
    np.savez_compressed(root / "train.npz", **train)
    np.save(root / "rm_contexts.npy", train["rm_x"])
    sample = read_json(root / "sample.json")
    meta, x, labels = make_queries(history, sample["development_users"], range(280, 287), True)
    meta.to_parquet(root / "search_queries.parquet", index=False)
    labels.to_parquet(root / "search_labels.parquet", index=False)
    np.save(root / "search_x.npy", x)
    write_json(
        root / "data_audit.json",
        {
            "rows": len(frame),
            "users": frame.uid.nunique(),
            "catalog": len(catalog),
            "train_rows": int((frame.timestamp < TRAIN_END).sum()),
            "train_chunks": len(train["x"]),
            "supervised_tokens": int((train["y"] >= SPECIAL).sum()),
            "rm_examples": len(train["rm_a"]),
            "search_queries": len(x),
            "search_users": meta.uid.nunique(),
            "search_catalog_coverage": float((labels.token >= SPECIAL).mean()),
            "ratio_over_100": int((frame.played_ratio_pct > 100).sum()),
            "train_npz_bytes": (root / "train.npz").stat().st_size,
        },
    )
    return history
