"""History-supported adapter study; original Task 2 modules stay immutable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from rec.io import file_hash, write_json
from rec.task2_data import DAY, encode, histories, make_queries, read_json

STATE_NAMES = (
    "log_history",
    "ratio5",
    "ratio20",
    "ratio64",
    "ratio_last",
    "completed64",
    "skip64",
    "head_share",
    "unique_share",
    "log_gap",
    "hour_sin",
    "hour_cos",
    "organic_share",
)


def reserve(root):
    if (root / "sample.json").exists():
        raise FileExistsError("External cohort already reserved")
    source = __import__("pathlib").Path("data/yambda/listens.parquet")
    old = read_json("runs/task2_final/sample.json")
    excluded = np.concatenate(
        [read_json(f"reports/{v}/evidence/sample.json")["sampled_user_ids"] for v in ("v3", "v4")]
        + [old["sampled_user_ids"]]
    )
    population = np.unique(pq.read_table(source, columns=["uid"]).column("uid").to_numpy())
    available = np.setdiff1d(population, excluded)
    users = np.sort(np.random.default_rng(161803).choice(available, 1000, replace=False))
    root.mkdir(parents=True, exist_ok=True)
    pieces = []
    for batch in pq.ParquetFile(source).iter_batches(batch_size=131072, use_threads=False):
        frame = batch.to_pandas()
        pieces.append(frame.loc[frame.uid.isin(users)])
    frame = pd.concat(pieces, ignore_index=True)
    frame.to_parquet(root / "sampled_events.parquet", index=False, compression="zstd")
    write_json(
        root / "sample.json",
        dict(
            seed=161803,
            external_users=users.tolist(),
            excluded_users=len(np.unique(excluded)),
            source_users=len(population),
            available_users=len(available),
            rows=len(frame),
            source_sha256=file_hash(source),
            events_sha256=file_hash(root / "sampled_events.parquet"),
            protocol_sha256=file_hash(__import__("pathlib").Path("docs/TASK2B_PROTOCOL.md")),
            all_source_rows_retained=True,
        ),
    )
    print(f"Reserved {len(users)} external users, {len(frame):,} complete rows", flush=True)


def catalog(root, frame):
    past = frame.loc[frame.timestamp < 270 * DAY]
    items = np.sort(past.item_id.unique().astype(np.int64))
    stats = (
        frame.loc[frame.timestamp < 200 * DAY]
        .assign(ratio=lambda d: np.clip(d.played_ratio_pct / 100, 0, 1))
        .groupby("item_id")
        .agg(
            count=("uid", "size"),
            ratio=("ratio", "mean"),
            length=("track_length_seconds", "median"),
        )
    )
    global_ratio = float(
        np.clip(frame.loc[frame.timestamp < 200 * DAY].played_ratio_pct / 100, 0, 1).mean()
    )
    aligned = stats.reindex(items)
    count = aligned["count"].fillna(0).to_numpy(np.float32)
    ratio = (aligned.ratio.fillna(global_ratio).to_numpy() * count + 20 * global_ratio) / (
        count + 20
    )
    head = np.load("runs/task2_final/catalog.npz")["items"]
    np.savez_compressed(
        root / "catalog.npz",
        items=items,
        head=head,
        counts=past.groupby("item_id").size().reindex(items).to_numpy(np.float32),
        count200=count,
        ratio200=ratio.astype(np.float32),
        lengths=aligned.length.fillna(0).to_numpy(np.float32),
        global_ratio=global_ratio,
        constant=float(np.clip(past.played_ratio_pct / 100, 0, 1).mean()),
    )


def record(h, pos, items, head):
    end = int(np.searchsorted(h["t"], h["t"][pos], side="left"))
    start = max(0, end - 64)
    raw, ratios = h["item"][start:end], h["ratio"][start:end]
    idx = np.searchsorted(items, raw)
    valid = idx < len(items)
    valid[valid] &= items[idx[valid]] == raw[valid]
    ids = np.full(64, -1, np.int32)
    outcome = np.zeros(64, np.float32)
    ids[-len(raw) :] = np.where(valid, idx, -1)
    outcome[-len(raw) :] = ratios
    hour = (h["t"][pos] % DAY) / DAY * 2 * np.pi
    state = np.asarray(
        [
            np.log1p(end) / 10,
            ratios[-5:].mean(),
            ratios[-20:].mean(),
            ratios.mean(),
            ratios[-1],
            (ratios >= 0.99).mean(),
            (ratios < 0.1).mean(),
            (encode(raw, head) >= 3).mean(),
            len(np.unique(raw)) / len(raw),
            np.log1p(h["t"][pos] - h["t"][end - 1]) / 12,
            np.sin(hour),
            np.cos(hour),
            h["organic"][start:end].mean(),
        ],
        np.float32,
    )
    return ids, outcome, state


def make_records(history, positions, cat):
    items, head = cat["items"], cat["head"]
    records = [record(history[int(u)], int(p), items, head) for u, p in positions]
    return {
        name: np.stack([r[j] for r in records])
        for j, name in enumerate(("history", "outcome", "state"))
    }


def prepare(root):
    frame = pd.read_parquet("runs/task2_final/sampled_events.parquet")
    catalog(root, frame)
    with np.load(root / "catalog.npz") as archive:
        cat = dict(archive)
    h = histories(frame, cat["head"])
    del frame
    rng = np.random.default_rng(271828)
    train, calibration, policy = [], [], []
    for uid, value in h.items():
        t = value["t"]
        unique = np.r_[True, np.diff(t) != 0] & np.r_[np.diff(t) != 0, True]
        for name, lo, hi, cap in (
            (train, 200, 260, 150),
            (calibration, 260, 270, 30),
            (policy, 240, 270, 40),
        ):
            eligible = np.flatnonzero(
                (t >= lo * DAY) & (t < hi * DAY) & unique & (value["organic"] == 0)
            )
            eligible = eligible[eligible > 0]
            if len(eligible) > cap:
                eligible = np.sort(rng.choice(eligible, cap, replace=False))
            name.extend((uid, p) for p in eligible)
    for name, positions in (
        ("rm_train", train),
        ("rm_calibration", calibration),
        ("policy", policy),
    ):
        arrays = make_records(h, positions, cat)
        arrays["users"] = np.asarray([u for u, _ in positions], np.int64)
        arrays["x"] = np.stack(
            [
                __import__("rec.task2_data", fromlist=["prefix"]).prefix(
                    h[u]["tokens"], np.searchsorted(h[u]["t"], h[u]["t"][p])
                )
                for u, p in positions
            ]
        )
        arrays["target"] = np.asarray([h[u]["item"][p] for u, p in positions], np.int64)
        arrays["ratio"] = np.asarray([h[u]["ratio"][p] for u, p in positions], np.float32)
        np.savez_compressed(root / f"{name}.npz", **arrays)
        print(f"Prepared {name}: {len(positions):,}", flush=True)
    users = read_json("runs/task2_final/sample.json")["development_users"]
    meta, x, labels = make_queries(h, users, range(280, 287), True)
    arrays = make_records(h, zip(meta.uid, meta.position, strict=True), cat)
    np.savez_compressed(
        root / "development.npz",
        x=x,
        target=labels.item_id,
        ratio=labels.ratio,
        users=meta.uid,
        **arrays,
    )
    meta.to_parquet(root / "development_queries.parquet", index=False)
    write_json(
        root / "prepared.json",
        dict(
            train=len(train),
            calibration=len(calibration),
            policy=len(policy),
            development=len(meta),
            catalog=len(cat["items"]),
            head=len(cat["head"]),
            state_features=STATE_NAMES,
        ),
    )


def reward_features(data, actions, cat):
    """B x K actions -> historical features; never inspect current target/outcome."""
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[:, None]
    b, k = actions.shape
    valid = (actions >= 0) & (actions < len(cat["items"]))
    safe = np.clip(actions, 0, len(cat["items"]) - 1)
    same = (data["history"][:, :, None] == actions[:, None, :]) & valid[:, None, :]
    count = same.sum(1)
    recency = np.where(same, np.arange(64)[None, :, None], -1).max(1)
    prior_ratio = (same * data["outcome"][:, :, None]).sum(1) / np.maximum(count, 1)
    candidate = np.stack(
        [
            np.log1p(np.where(valid, cat["count200"][safe], 0)) / 10,
            np.where(valid, cat["ratio200"][safe], cat["global_ratio"]),
            np.where(valid, cat["lengths"][safe], 0) / 600,
            np.log1p(count),
            np.where(recency >= 0, (64 - recency) / 64, 2),
            prior_ratio,
            same[:, -1].astype(float),
        ],
        -1,
    )
    return (
        np.concatenate(
            [np.broadcast_to(data["state"][:, None, :], (b, k, data["state"].shape[1])), candidate],
            -1,
        )
        .reshape(b * k, -1)
        .astype(np.float32)
    )


def item_indices(items, catalog):
    idx = np.searchsorted(catalog, items)
    valid = idx < len(catalog)
    valid[valid] &= catalog[idx[valid]] == np.asarray(items)[valid]
    return np.where(valid, idx, -1).astype(np.int32)
