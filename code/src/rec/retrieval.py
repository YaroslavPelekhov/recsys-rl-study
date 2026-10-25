from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from rec.protocol import Config


def tokens(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(sorted(set(part.strip() for part in str(value).split(",") if part.strip())))


def token_matrix(items: pd.DataFrame, column: str) -> sparse.csr_matrix:
    lists = [tokens(v) for v in items[column]]
    vocabulary = {token: i for i, token in enumerate(sorted({t for row in lists for t in row}))}
    rows, cols = [], []
    for i, row in enumerate(lists):
        for token in row:
            rows.append(i)
            cols.append(vocabulary[token])
    return sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(len(items), len(vocabulary))
    )


def top_indices(scores: np.ndarray, ids: np.ndarray, count: int) -> np.ndarray:
    """Stable item-id tie break; argpartition alone is not deterministic at a tied boundary."""
    return np.lexsort((ids, -scores))[:count]


def prune_rows(matrix: sparse.csr_matrix, count: int) -> sparse.csr_matrix:
    rows, cols, values = [], [], []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        indices, data = matrix.indices[start:end], matrix.data[start:end]
        chosen = top_indices(data, indices, count)
        rows.extend([row] * len(chosen))
        cols.extend(indices[chosen])
        values.extend(data[chosen])
    return sparse.csr_matrix((np.asarray(values, np.float32), (rows, cols)), shape=matrix.shape)


@dataclass
class Retrieval:
    candidates: pd.DataFrame
    catalog: np.ndarray
    eligible_users: np.ndarray
    diagnostics: dict


def retrieve(
    history: pd.DataFrame,
    items: pd.DataFrame,
    cutoff: pd.Timestamp,
    config: Config,
    query_users: np.ndarray | None = None,
    extra_items: Mapping[int, np.ndarray] | None = None,
) -> Retrieval:
    """Fit retrieval exclusively on historical records. This function never receives labels."""
    if history.empty or not (history.timestamp < cutoff).all():
        raise ValueError("Retrieval requires a nonempty, strictly historical table")
    user_ids = np.sort(history.user_id.unique())
    item_ids = np.sort(history.item_id.unique())
    user_index = pd.Index(user_ids)
    item_index = pd.Index(item_ids)
    h = history.copy()
    h["u"] = user_index.get_indexer(h.user_id)
    h["i"] = item_index.get_indexer(h.item_id)
    h["age"] = (cutoff - h.timestamp).dt.total_seconds() / 86400
    shape = (len(user_ids), len(item_ids))
    binary = sparse.csr_matrix((np.ones(len(h), np.float32), (h.u, h.i)), shape=shape)
    binary.data[:] = 1
    recent = (
        h[h.age <= 90]
        .sort_values(["timestamp", "item_id"], kind="stable")
        .groupby("user_id", sort=False)
        .tail(config.max_history)
    )
    x = sparse.csr_matrix((np.ones(len(recent), np.float32), (recent.u, recent.i)), shape=shape)
    x.data[:] = 1
    counts = np.asarray(x.sum(axis=0)).ravel()
    gram = (x.T @ x).tocoo()
    # Cosine similarity multiplied by co-occurrence shrinkage c/(c+10).
    values = gram.data / np.sqrt(np.maximum(counts[gram.row] * counts[gram.col], 1))
    values *= gram.data / (gram.data + 10)
    similarity = sparse.csr_matrix(
        (values, (gram.row, gram.col)), shape=(len(item_ids), len(item_ids))
    )
    similarity.setdiag(0)
    similarity.eliminate_zeros()
    similarity = prune_rows(similarity, config.neighbors)
    weighted = sparse.csr_matrix(
        (np.exp2(-recent.age.to_numpy() / 30).astype(np.float32), (recent.u, recent.i)), shape=shape
    )
    collaborative = (weighted @ similarity).tocsr()
    metadata = items.reindex(item_ids)
    genres = token_matrix(metadata, "genres")
    profiles = (weighted @ genres).tocsr()
    profile_norm = np.sqrt(np.asarray(profiles.multiply(profiles).sum(axis=1)).ravel())
    item_norm = np.sqrt(np.asarray(genres.multiply(genres).sum(axis=1)).ravel())
    profiles = sparse.diags(1 / np.maximum(profile_norm, 1e-8)) @ profiles
    normalized_genres = sparse.diags(1 / np.maximum(item_norm, 1e-8)) @ genres
    content = (profiles @ normalized_genres.T).tocsr()
    pop = np.bincount(h.i, weights=np.exp2(-h.age / 7), minlength=len(item_ids))
    content_pop = 0.2 + np.log1p(pop) / max(float(np.log1p(pop).max()), 1)
    rows = []
    requested = None if query_users is None else set(query_users)
    for u, uid in enumerate(user_ids):
        if requested is not None and uid not in requested:
            continue
        seen = binary.indices[binary.indptr[u] : binary.indptr[u + 1]]
        collab = collaborative.getrow(u).toarray().ravel()
        affinity = content.getrow(u).toarray().ravel()
        source_scores = [pop.copy(), collab.copy(), affinity * content_pop]
        source_lists = []
        for scores in source_scores:
            scores[seen] = -np.inf
            chosen = top_indices(scores, item_ids, config.retrieval_per_source)
            source_lists.append(chosen[np.isfinite(scores[chosen]) & (scores[chosen] > 0)])
        union = np.unique(np.concatenate(source_lists))
        fusion = np.zeros(len(item_ids), np.float32)
        masks = np.zeros((3, len(item_ids)), np.int8)
        for source, indices in enumerate(source_lists):
            fusion[indices] += 1 / (20 + np.arange(1, len(indices) + 1))
            masks[source, indices] = 1
        chosen = union[top_indices(fusion[union], item_ids[union], config.candidates)]
        if extra_items is not None and uid in extra_items:
            extra = item_index.get_indexer(extra_items[uid])
            if (extra < 0).any():
                raise ValueError("Extra retrieval source nominated a future/unknown item")
            extra = np.setdiff1d(extra, seen)
            chosen = np.unique(np.concatenate([chosen, extra]))
            chosen = chosen[top_indices(fusion[chosen], item_ids[chosen], len(chosen))]
        for rank, i in enumerate(chosen):
            rows.append(
                (
                    int(uid),
                    int(item_ids[i]),
                    float(pop[i]),
                    float(collab[i]),
                    float(affinity[i]),
                    float(fusion[i]),
                    rank + 1,
                    int(masks[0, i]),
                    int(masks[1, i]),
                    int(masks[2, i]),
                )
            )
    columns = [
        "user_id",
        "item_id",
        "retrieval_pop",
        "retrieval_cf",
        "retrieval_content",
        "retrieval_rrf",
        "retrieval_rank",
        "source_pop",
        "source_cf",
        "source_content",
    ]
    candidates = (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )
    return Retrieval(
        candidates,
        item_ids,
        user_ids,
        {
            "catalog_items": len(item_ids),
            "eligible_users": len(user_ids),
            "candidate_rows": len(candidates),
            "graph_nonzeros": similarity.nnz,
            "max_history_timestamp": str(history.timestamp.max()),
        },
    )
