"""Historical closed-form recommenders and time-conditioned user representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg, sparse
from threadpoolctl import threadpool_limits


@dataclass
class HistoricalMatrix:
    users: pd.Index
    items: pd.Index
    binary: sparse.csr_matrix
    age: sparse.csr_matrix
    cutoff: pd.Timestamp

    @classmethod
    def build(cls, history: pd.DataFrame, cutoff: pd.Timestamp) -> HistoricalMatrix:
        if history.empty or not (history.timestamp < cutoff).all():
            raise ValueError("Expected strictly historical interactions")
        users = pd.Index(np.sort(history.user_id.unique()))
        items = pd.Index(np.sort(history.item_id.unique()))
        if history.duplicated(["user_id", "item_id"]).any():
            raise ValueError("This implementation expects unique last-watch user-item records")
        rows, columns = users.get_indexer(history.user_id), items.get_indexer(history.item_id)
        shape = len(users), len(items)
        binary = sparse.csr_matrix((np.ones(len(history)), (rows, columns)), shape=shape)
        age_values = (cutoff - history.timestamp).dt.total_seconds().to_numpy() / 86400
        age = sparse.csr_matrix((age_values, (rows, columns)), shape=shape)
        return cls(users, items, binary, age, cutoff)

    def profile(self, half_life: float | None = None) -> sparse.csr_matrix:
        if half_life is None:
            return self.binary
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        result = self.age.copy()
        result.data = np.exp2(-result.data / half_life)
        return result

    def ease(self, regularization: float, threads: int = 8) -> np.ndarray:
        """Steck (2019): B_ij = -P_ij/P_jj, diag(B)=0; P=(X'X+lambda I)^-1."""
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        with threadpool_limits(limits=threads, user_api="blas"):
            gram = (self.binary.T @ self.binary).toarray()
            gram.flat[:: len(gram) + 1] += regularization
            inverse = linalg.cho_solve(
                linalg.cho_factor(gram, lower=True, check_finite=False),
                np.eye(len(gram)),
                check_finite=False,
            )
        coefficients = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(coefficients, 0)
        return coefficients.astype(np.float32)

    def rp3beta(self, beta: float = 0.6, neighbors: int = 200) -> sparse.csr_matrix:
        """Degree-normalized user-item-user-item random walk, with popular-item penalty."""
        from rec.retrieval import prune_rows

        user_degree = np.asarray(self.binary.sum(axis=1)).ravel()
        item_degree = np.asarray(self.binary.sum(axis=0)).ravel()
        transition = (
            sparse.diags(1 / np.maximum(item_degree, 1))
            @ self.binary.T
            @ sparse.diags(1 / np.maximum(user_degree, 1))
            @ self.binary
        )
        transition = (transition @ sparse.diags(np.maximum(item_degree, 1) ** -beta)).tocsr()
        transition.setdiag(0)
        transition.eliminate_zeros()
        return prune_rows(transition, neighbors)

    def pair_scores(
        self,
        pairs: pd.DataFrame,
        coefficients: np.ndarray | sparse.csr_matrix,
        half_life: float | None = None,
    ) -> np.ndarray:
        """Score a fixed candidate table in user batches, with no target columns accessed."""
        row_users = self.users.get_indexer(pairs.user_id)
        row_items = self.items.get_indexer(pairs.item_id)
        if (row_users < 0).any() or (row_items < 0).any():
            raise ValueError("Unknown user or item in historical candidate table")
        profile = self.profile(half_life)
        result = np.empty(len(pairs), dtype=np.float32)
        unique_users = np.unique(row_users)
        for start in range(0, len(unique_users), 256):
            batch = unique_users[start : start + 256]
            scores = profile[batch] @ coefficients
            if sparse.issparse(scores):
                scores = scores.toarray()
            lookup = {u: i for i, u in enumerate(batch)}
            mask = np.isin(row_users, batch)
            result[mask] = scores[[lookup[u] for u in row_users[mask]], row_items[mask]]
        return result

    def recommend(
        self,
        coefficients: np.ndarray,
        query_users: np.ndarray,
        half_life: float | None = None,
        count: int = 120,
    ) -> dict[int, np.ndarray]:
        from rec.retrieval import top_indices

        indices = self.users.get_indexer(query_users)
        if (indices < 0).any():
            raise ValueError("Unknown query user")
        profile = self.profile(half_life)
        result = {}
        for start in range(0, len(indices), 128):
            batch = indices[start : start + 128]
            values = profile[batch] @ coefficients
            for row, u in enumerate(batch):
                seen = self.binary.indices[self.binary.indptr[u] : self.binary.indptr[u + 1]]
                values[row, seen] = -np.inf
                chosen = top_indices(values[row], self.items.to_numpy(), count)
                chosen = chosen[np.isfinite(values[row, chosen])]
                result[int(self.users[u])] = self.items.to_numpy()[chosen]
        return result
