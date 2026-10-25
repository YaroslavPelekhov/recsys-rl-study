"""Chronological pair graphs and learned transition density-ratio representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

DAY = 86400
HALF_LIVES = (1, 24, 168)


def transition_pairs(history: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    if history.empty or not (history.timestamp < cutoff).all():
        raise ValueError("Transition fitting requires strictly historical events")
    h = history[(history.timestamp >= cutoff - 30 * DAY) & (history.played_ratio_pct >= 50)]
    h = h.sort_values(["user_id", "timestamp", "item_id"], kind="stable")
    previous = h.shift(1)
    delta = h.timestamp.to_numpy(dtype=np.int64) - previous.timestamp.fillna(-DAY).to_numpy(
        dtype=np.int64
    )
    valid = (
        (h.user_id.to_numpy() == previous.user_id.fillna(-1).to_numpy())
        & (delta > 0)
        & (delta <= 1800)
    )
    return pd.DataFrame(
        {
            "source": previous.loc[valid, "item_id"].to_numpy(dtype=np.int64),
            "target": h.loc[valid, "item_id"].to_numpy(dtype=np.int64),
            "user_id": h.loc[valid, "user_id"].to_numpy(dtype=np.int64),
            "day": h.loc[valid, "timestamp"].to_numpy(dtype=np.int64) // DAY,
        }
    )


def permuted_targets(pairs: pd.DataFrame, seed: int, conditional: bool) -> np.ndarray:
    targets = pairs.target.to_numpy().copy()
    rng = np.random.default_rng(seed)
    if conditional:
        for indices in pairs.groupby(["user_id", "day"], sort=True).indices.values():
            targets[indices] = rng.permutation(targets[indices])
    else:
        targets = rng.permutation(targets)
    return targets


def noise_targets(pairs: pd.DataFrame, seed: int, conditional: bool | str) -> np.ndarray:
    if conditional == "mixture":
        local = permuted_targets(pairs, seed, True)
        global_noise = permuted_targets(pairs, seed, False)
        choose_global = np.random.default_rng(seed + 937).random(len(pairs)) < 0.5
        return np.where(choose_global, global_noise, local)
    if not isinstance(conditional, bool):
        raise ValueError("Unknown noise construction")
    return permuted_targets(pairs, seed, conditional)


def prune_signed(matrix: sparse.csr_matrix, per_sign: int = 64) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True)
    matrix.eliminate_zeros()
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        values = matrix.data[start:end]
        keep = []
        for sign in (1, -1):
            candidates = np.flatnonzero(values * sign > 0)
            if len(candidates) > per_sign:
                order = np.lexsort(
                    (matrix.indices[start:end][candidates], -sign * values[candidates])
                )
                candidates = candidates[order[:per_sign]]
            keep.extend(candidates)
        mask = np.ones(len(values), dtype=bool)
        mask[keep] = False
        values[mask] = 0
    matrix.eliminate_zeros()
    return matrix


def graph_operators(pairs: pd.DataFrame, items: pd.Index) -> dict[str, sparse.csr_matrix]:
    sources = items.get_indexer(pairs.source)
    targets = items.get_indexer(pairs.target)
    if (sources < 0).any() or (targets < 0).any():
        raise ValueError("Transition item missing from historical catalog")
    shape = len(items), len(items)
    counts = sparse.csr_matrix((np.ones(len(pairs), np.float32), (sources, targets)), shape=shape)
    null = sparse.csr_matrix(shape, dtype=np.float32)
    for seed in (19, 23):
        randomized = items.get_indexer(permuted_targets(pairs, seed, True))
        null += sparse.csr_matrix(
            (np.full(len(pairs), 0.5, np.float32), (sources, randomized)), shape=shape
        )
    raw = counts.copy()
    raw.data /= raw.data + 5
    numerator = (counts - null).tocsr()
    denominator = (counts + null).tocsr()
    # Add a constant only on existing support, retaining sparsity.
    inverse = denominator.copy()
    inverse.data = 1 / (inverse.data + 5)
    signed = numerator.multiply(inverse).tocsr()
    return {"raw": prune_signed(raw), "signed": prune_signed(signed)}


def historical_profiles(
    history: pd.DataFrame, users: pd.Index, items: pd.Index, cutoff: int
) -> dict[int, sparse.csr_matrix]:
    h = (
        history[history.played_ratio_pct >= 50]
        .sort_values(["user_id", "timestamp", "item_id"], kind="stable")
        .groupby("user_id", sort=False)
        .tail(100)
    )
    rows, cols = users.get_indexer(h.user_id), items.get_indexer(h.item_id)
    age_hours = (cutoff - h.timestamp.to_numpy()) / 3600
    result = {}
    for half_life in HALF_LIVES:
        values = np.exp2(-age_hours / half_life).astype(np.float32)
        matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(users), len(items)))
        normalizer = np.asarray(matrix.sum(axis=1)).ravel()
        matrix = (sparse.diags(1 / np.maximum(normalizer, 1e-12)) @ matrix).tocsr()
        result[half_life] = matrix
    return result


@dataclass
class TransitionEmbedding:
    items: pd.Index
    source: np.ndarray
    target: np.ndarray
    losses: list[dict]
    metadata: dict

    @classmethod
    def fit(
        cls,
        pairs: pd.DataFrame,
        conditional: bool,
        seed: int,
        epochs: int = 15,
        dimensions: int = 64,
        batch_size: int = 8192,
    ):
        import torch
        from torch import nn
        from torch.nn import functional as functional

        if pairs.empty:
            raise ValueError("No historical chronological transitions")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)
        torch.set_num_threads(12)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        items = pd.Index(np.unique(np.r_[pairs.source, pairs.target]))
        source_indices = torch.as_tensor(
            items.get_indexer(pairs.source), dtype=torch.long, device=device
        )
        target_indices = torch.as_tensor(
            items.get_indexer(pairs.target), dtype=torch.long, device=device
        )
        sources = nn.Embedding(len(items), dimensions, sparse=True, device=device)
        targets = nn.Embedding(len(items), dimensions, sparse=True, device=device)
        nn.init.normal_(sources.weight, std=0.1)
        nn.init.normal_(targets.weight, std=0.1)
        optimizer = torch.optim.SparseAdam([sources.weight, targets.weight], lr=0.01)
        losses = []
        for epoch in range(epochs):
            noise = noise_targets(pairs, seed * 10000 + epoch, conditional)
            negative = torch.as_tensor(items.get_indexer(noise), dtype=torch.long, device=device)
            # Separate RNG stream ensures identical positive mini-batches for both null models.
            order = np.random.default_rng(seed * 100000 + epoch).permutation(len(pairs))
            total = 0.0
            for start in range(0, len(order), batch_size):
                index = torch.as_tensor(
                    order[start : start + batch_size], dtype=torch.long, device=device
                )
                u = sources(source_indices[index])
                p, n = targets(target_indices[index]), targets(negative[index])
                positive_logits = (u * p).sum(dim=1) / np.sqrt(dimensions)
                negative_logits = (u * n).sum(dim=1) / np.sqrt(dimensions)
                loss = 0.5 * (
                    functional.softplus(-positive_logits).mean()
                    + functional.softplus(negative_logits).mean()
                )
                loss = loss + 1e-4 * (u.square().mean() + p.square().mean() + n.square().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(index)
            losses.append({"epoch": epoch + 1, "loss": total / len(pairs)})
        metadata = {
            "device": device,
            "torch": torch.__version__,
            "conditional": conditional,
            "seed": seed,
            "epochs": epochs,
            "dimensions": dimensions,
            "pairs": len(pairs),
            "items": len(items),
            "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2
            if device == "cuda"
            else 0,
        }
        result = cls(
            items,
            sources.weight.detach().cpu().numpy().copy(),
            targets.weight.detach().cpu().numpy().copy(),
            losses,
            metadata,
        )
        del optimizer, sources, targets, source_indices, target_indices
        if device == "cuda":
            torch.cuda.empty_cache()
        return result

    def save(self, path):
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, items=self.items.to_numpy(), source=self.source, target=self.target
            )
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(pd.Index(data["items"]), data["source"], data["target"], [], {})

    def user_vectors(self, profile: sparse.csr_matrix, catalog: pd.Index) -> np.ndarray:
        locations = catalog.get_indexer(self.items)
        if (locations < 0).any():
            raise ValueError("Encoder item missing from historical catalog")
        return np.asarray(profile[:, locations] @ self.source, dtype=np.float32)

    def retrieval_scores(self, vectors: np.ndarray) -> np.ndarray:
        with threadpool_limits(12, user_api="blas"):
            return (vectors @ self.target.T) / np.sqrt(self.source.shape[1])

    def pair_scores(
        self, vectors: np.ndarray, row_users: np.ndarray, item_ids: np.ndarray
    ) -> np.ndarray:
        indices = self.items.get_indexer(item_ids)
        result = np.zeros(len(indices), np.float32)
        for start in range(0, len(indices), 50000):
            stop = min(start + 50000, len(indices))
            valid = indices[start:stop] >= 0
            selected = np.flatnonzero(valid) + start
            result[selected] = np.sum(
                vectors[row_users[selected]] * self.target[indices[selected]], axis=1
            ) / np.sqrt(self.source.shape[1])
        return result
