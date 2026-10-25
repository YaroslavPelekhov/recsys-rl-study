"""Point-in-time candidate tables for the Yambda study; all sampled rows are retained."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from rec.io import write_json
from rec.retrieval import top_indices
from rec.transition import (
    DAY,
    HALF_LIVES,
    TransitionEmbedding,
    graph_operators,
    historical_profiles,
    transition_pairs,
)

MUSIC_CATS = ["user_id_cat", "item_id_cat", "artist_cat"]


def bounded_top(scores, ids, count):
    """Exact stable top-k without sorting a 280k-item catalog for every query."""
    if len(scores) <= count * 4:
        return top_indices(scores, ids, count)
    threshold = np.partition(scores, len(scores) - count)[-count]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    needed = count - len(above)
    if len(tied) > needed:
        tied = tied[np.argsort(ids[tied], kind="stable")[:needed]]
    chosen = np.r_[above, tied[:needed]]
    return chosen[np.lexsort((ids[chosen], -scores[chosen]))]


def load_music(root: Path):
    events = pd.read_parquet(root / "sampled_events.parquet").rename(columns={"uid": "user_id"})
    mapping = pd.read_parquet("data/yambda/artist_item_mapping.parquet")
    mapping = mapping[mapping.item_id.isin(events.item_id.unique())]
    artist = mapping.groupby("item_id").artist_id.min()
    events["artist"] = events.item_id.map(artist).fillna(-1).astype(np.int64)
    return events, artist


def music_truth(events: pd.DataFrame, day: int) -> pd.DataFrame:
    cutoff = day * DAY
    return (
        events[
            (events.timestamp >= cutoff)
            & (events.timestamp < cutoff + DAY)
            & (events.played_ratio_pct >= 50)
        ][["user_id", "item_id"]]
        .drop_duplicates()
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )


def _matrix(history, users, items, values):
    return sparse.csr_matrix(
        (
            np.asarray(values, np.float32),
            (users.get_indexer(history.user_id), items.get_indexer(history.item_id)),
        ),
        shape=(len(users), len(items)),
    )


def _sparse_pairs(matrix, rows, cols):
    return np.asarray(matrix[rows, cols]).ravel().astype(np.float32)


def manual_music_features(
    pairs: pd.DataFrame, history: pd.DataFrame, artists: pd.Series, cutoff: int
):
    if not (history.timestamp < cutoff).all():
        raise ValueError("Future record supplied to manual features")
    frame = pairs.copy()
    frame["artist"] = frame.item_id.map(artists).fillna(-1).astype(np.int64)
    h = history.copy()
    h["positive"] = (h.played_ratio_pct >= 50).astype(np.int32)
    axes = {
        "u": ["user_id"],
        "i": ["item_id"],
        "ui": ["user_id", "item_id"],
        "ua": ["user_id", "artist"],
    }
    for days in (1, 7, 30, None):
        window = "all" if days is None else str(days)
        selected = h if days is None else h[h.timestamp >= cutoff - days * DAY]
        for name, keys in axes.items():
            aggregate = selected.groupby(keys, sort=False).agg(
                total=("positive", "size"), positive=("positive", "sum")
            )
            index = (
                pd.MultiIndex.from_frame(frame[keys]) if len(keys) > 1 else pd.Index(frame[keys[0]])
            )
            aligned = aggregate.reindex(index).fillna(0)
            for measure in ("total", "positive"):
                frame[f"{name}_{measure}_{window}"] = aligned[measure].to_numpy(dtype=np.float32)
        for name in ("u", "i", "ui", "ua"):
            frame[f"{name}_positive_rate_{window}"] = (frame[f"{name}_positive_{window}"] + 1) / (
                frame[f"{name}_total_{window}"] + 2
            )
        for name in ("ui", "ua"):
            frame[f"{name}_share_{window}"] = frame[f"{name}_positive_{window}"] / np.maximum(
                frame[f"u_positive_{window}"], 1
            )
    for name, keys in axes.items():
        last = h.groupby(keys).timestamp.max()
        index = pd.MultiIndex.from_frame(frame[keys]) if len(keys) > 1 else pd.Index(frame[keys[0]])
        frame[f"{name}_age_hours"] = (
            cutoff - last.reindex(index).fillna(-30 * DAY).to_numpy(dtype=np.int64)
        ) / 3600
    for key in ("user_id", "item_id", "artist"):
        frame[f"{key}_cat"] = frame[key].astype(str)
    columns = [c for c in frame if c not in ("user_id", "item_id", "artist")]
    return frame, columns


def prepare_music(events: pd.DataFrame, artists: pd.Series, day: int, root: Path, targets: bool):
    cache = root / "cache"
    cache.mkdir(exist_ok=True, parents=True)
    path = cache / f"day{day}.parquet"
    if path.exists():
        audit = json.loads((cache / f"day{day}_audit.json").read_text())
        if audit["targets_constructed"] != targets:
            raise ValueError("Target availability differs from cache")
        return pd.read_parquet(path)
    cutoff = day * DAY - 1800
    history = events[events.timestamp < cutoff].copy()
    users = pd.Index(np.sort(history.user_id.unique()))
    items = pd.Index(np.sort(history.item_id.unique()))
    positive = history[history.played_ratio_pct >= 50]
    truth = music_truth(events, day) if targets else None
    queries = np.intersect1d(users, truth.user_id.unique()) if targets else users.to_numpy()
    query_rows = users.get_indexer(queries)
    profiles = historical_profiles(history, users, items, cutoff)
    chronological = transition_pairs(history, cutoff)
    operators = graph_operators(chronological, items)
    graph_scores = {
        f"{name}_h{half}": (profiles[half] @ operator).tocsr()
        for name, operator in operators.items()
        for half in HALF_LIVES
    }
    del operators
    # UserKNN avoids materializing the quadratic item-item Gram matrix.
    binary = _matrix(positive, users, items, np.ones(len(positive)))
    binary.data[:] = 1
    inverse_frequency = np.log1p(len(users) / np.maximum(np.asarray(binary.sum(axis=0)).ravel(), 1))
    normalized = binary @ sparse.diags(inverse_frequency)
    norms = np.sqrt(np.asarray(normalized.multiply(normalized).sum(axis=1)).ravel())
    normalized = sparse.diags(1 / np.maximum(norms, 1e-8)) @ normalized
    similarities = (normalized @ normalized.T).toarray()
    np.fill_diagonal(similarities, 0)
    for row in range(len(users)):
        keep = top_indices(similarities[row], users.to_numpy(), 20)
        mask = np.ones(len(users), bool)
        mask[keep] = False
        similarities[row, mask] = 0
    collaborative = (sparse.csr_matrix(similarities) @ binary).tocsr()
    ages = cutoff - positive.timestamp.to_numpy(dtype=np.int64)
    weights = np.exp2(-ages / (7 * DAY)).astype(np.float32)
    repeats = _matrix(positive, users, items, weights)
    popularity = np.bincount(
        items.get_indexer(positive.item_id), weights=weights, minlength=len(items)
    ).astype(np.float32)
    ua = positive.assign(weight=weights).groupby(["user_id", "artist"]).weight.sum()
    item_artist = items.map(artists).fillna(-1).to_numpy(dtype=np.int64)
    item_pop_factor = np.log1p(popularity) / max(float(np.log1p(popularity).max()), 1.0)
    encoders, vectors, logs = {}, {}, []
    retrieval_nn = {}
    models_dir = root / "encoders" / f"day{day}"
    models_dir.mkdir(parents=True, exist_ok=True)
    for seed in (17, 42, 73):
        for name, conditional in (("global", False), ("conditional", True)):
            stored = models_dir / f"{name}_seed{seed}.npz"
            start = time.perf_counter()
            if stored.exists() and stored.with_suffix(".json").exists():
                encoder = TransitionEmbedding.load(stored)
                information = json.loads(stored.with_suffix(".json").read_text())
            else:
                encoder = TransitionEmbedding.fit(chronological, conditional, seed)
                encoder.save(stored)
                information = {
                    "metadata": encoder.metadata,
                    "losses": encoder.losses,
                    "seconds": time.perf_counter() - start,
                }
                write_json(stored.with_suffix(".json"), information)
            logs.append({"method": name, "seed": seed, **information})
            encoders[name, seed] = encoder
            for half in HALF_LIVES:
                vectors[name, seed, half] = encoder.user_vectors(profiles[half], items)
            if seed == 17:
                scores = encoder.retrieval_scores(vectors[name, seed, 24][query_rows])
                retrieval_nn[name] = (items.get_indexer(encoder.items), scores)
            print(
                f"day{day} {name}/{seed}: {information['seconds']:.1f}s, loss {information['losses'][-1]['loss']:.4f}",
                flush=True,
            )
    rows = []
    for query_position, (uid, user_row) in enumerate(zip(queries, query_rows)):
        artist_scores = (
            ua.loc[uid].reindex(item_artist).fillna(0).to_numpy(dtype=np.float32) * item_pop_factor
            if uid in ua.index.get_level_values(0)
            else np.zeros(len(items), np.float32)
        )
        original = [
            repeats.getrow(user_row).toarray().ravel(),
            popularity,
            collaborative.getrow(user_row).toarray().ravel(),
            artist_scores,
        ]
        source_lists = []
        for source, values in enumerate(original):
            top = bounded_top(values, items.to_numpy(), 120 if source == 0 else 80)
            source_lists.append(top[values[top] > 0])
        for name in ("raw", "signed"):
            vector = graph_scores[f"{name}_h24"].getrow(user_row)
            top = top_indices(vector.data, items.to_numpy()[vector.indices], 80)
            source_lists.append(vector.indices[top[vector.data[top] > 0]])
        for name in ("global", "conditional"):
            indices, scores = retrieval_nn[name]
            values = scores[query_position]
            top = bounded_top(values, items.to_numpy()[indices], 80)
            source_lists.append(indices[top[values[top] > 0]])
        union = np.unique(np.concatenate(source_lists))
        fusion = np.zeros(len(items), np.float32)
        for nominations in source_lists:
            fusion[nominations] += 1 / (20 + np.arange(1, len(nominations) + 1))
        chosen = union[top_indices(fusion[union], items.to_numpy()[union], 300)]
        for rank, item in enumerate(chosen):
            rows.append(
                (
                    int(uid),
                    int(items[item]),
                    *(float(v[item]) for v in original),
                    float(fusion[item]),
                    rank + 1,
                )
            )
    pairs = (
        pd.DataFrame(
            rows,
            columns=[
                "user_id",
                "item_id",
                "repeat_score",
                "pop_score",
                "knn_score",
                "artist_score",
                "retrieval_rrf",
                "retrieval_rank",
            ],
        )
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )
    row_users, row_items = users.get_indexer(pairs.user_id), items.get_indexer(pairs.item_id)
    frame, manual = manual_music_features(pairs, history, artists, cutoff)
    for name, matrix in graph_scores.items():
        frame[name] = _sparse_pairs(matrix, row_users, row_items)
    for (name, seed), encoder in encoders.items():
        for half in HALF_LIVES:
            frame[f"{name}_seed{seed}_h{half}"] = encoder.pair_scores(
                vectors[name, seed, half], row_users, pairs.item_id.to_numpy()
            )
    if targets:
        truth = truth[truth.user_id.isin(users)]
        frame = frame.merge(
            truth.assign(label=np.int8(1)),
            on=["user_id", "item_id"],
            how="left",
            validate="one_to_one",
        )
        frame["label"] = frame.label.fillna(0).astype(np.int8)
        truth.to_parquet(cache / f"day{day}_truth.parquet", index=False)
    else:
        frame["label"] = np.int8(0)
    frame["query_id"] = f"day{day}:" + frame.user_id.astype(str)
    if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Non-finite point-in-time features")
    frame.to_parquet(path, index=False)
    write_json(cache / "groups.json", {"manual": manual})
    write_json(
        cache / f"day{day}_audit.json",
        {
            "day": day,
            "history_cutoff_exclusive": cutoff,
            "label_cutoff": day * DAY,
            "history_max_timestamp": int(history.timestamp.max()),
            "history_rows": len(history),
            "catalog": len(items),
            "queries": len(queries),
            "pairs": len(frame),
            "chronological_edges": len(chronological),
            "targets_constructed": targets,
            "neural_retrieval_seed": 17,
        },
    )
    write_json(cache / f"day{day}_encoders.json", logs)
    del encoders, retrieval_nn, graph_scores, collaborative, binary, history
    gc.collect()
    return frame


def with_encoder_seed(frame, seed):
    result = frame.copy()
    for name in ("global", "conditional"):
        for half in HALF_LIVES:
            result[f"{name}_h{half}"] = result[f"{name}_seed{seed}_h{half}"]
    return result
