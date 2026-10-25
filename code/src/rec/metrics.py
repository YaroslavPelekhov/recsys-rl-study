from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def user_metrics(
    pairs: pd.DataFrame, scores: np.ndarray, truth: pd.DataFrame, users: np.ndarray, k: int = 10
) -> pd.DataFrame:
    if len(scores) != len(pairs) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and aligned to candidates")
    if pairs.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Duplicate candidates")
    relevant = truth.groupby("user_id").item_id.agg(set).to_dict()
    ranked = (
        pairs[["user_id", "item_id"]]
        .assign(score=scores)
        .sort_values(["user_id", "score", "item_id"], ascending=[True, False, True], kind="stable")
    )
    lists = ranked.groupby("user_id", sort=False).item_id.agg(list).to_dict()
    discounts = 1 / np.log2(np.arange(2, k + 2))
    rows = []
    for uid in users:
        actual = relevant.get(uid, set())
        if not actual:
            continue  # Macro ranking is conditional on at least one relevant future target.
        candidates = lists.get(uid, [])
        top = candidates[:k]
        hits = np.asarray([item in actual for item in top], dtype=float)
        found = len(actual.intersection(candidates))
        ideal = float(discounts[: min(k, len(actual))].sum())
        dcg = float(hits @ discounts[: len(hits)])
        precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        conditional_ideal = float(discounts[: min(k, found)].sum())
        rows.append(
            {
                "user_id": int(uid),
                "n_relevant": len(actual),
                "n_candidates": len(candidates),
                "candidate_hits": found,
                "ndcg": dcg / ideal,
                "recall": float(hits.sum() / len(actual)),
                "map": float((precision * hits).sum() / min(k, len(actual))),
                "hit_rate": float(hits.sum() > 0),
                "candidate_recall": found / len(actual),
                "oracle_ndcg": float(discounts[: min(k, found)].sum() / ideal),
                "conditional_ndcg": dcg / conditional_ideal if conditional_ideal else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize(
    per_user: pd.DataFrame,
    pairs: pd.DataFrame | None = None,
    scores: np.ndarray | None = None,
    k: int = 10,
) -> dict:
    if per_user.empty:
        raise ValueError("No warm users with future positives in this fold")
    columns = [
        "ndcg",
        "recall",
        "map",
        "hit_rate",
        "candidate_recall",
        "oracle_ndcg",
        "conditional_ndcg",
    ]
    result = {
        f"{name}@{k}" if name in ("ndcg", "recall", "map", "hit_rate") else name: float(
            per_user[name].mean()
        )
        for name in columns
    }
    result["users"] = len(per_user)
    if pairs is not None and scores is not None and pairs.label.nunique() == 2:
        result["candidate_auc"] = float(roc_auc_score(pairs.label, scores))
    return result


def paired_bootstrap(
    first: pd.DataFrame, second: pd.DataFrame, seed: int, samples: int = 3000, metric: str = "ndcg"
) -> dict:
    """Cluster at the user level; callers average model seeds before resampling users."""
    joined = first[["user_id", metric]].merge(
        second[["user_id", metric]], on="user_id", suffixes=("_a", "_b"), validate="one_to_one"
    )
    if len(joined) != len(first) or len(joined) != len(second) or joined.empty:
        raise ValueError("Paired comparison requires identical, nonempty user populations")
    differences = (joined[f"{metric}_b"] - joined[f"{metric}_a"]).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for i in range(samples):
        means[i] = rng.choice(differences, len(differences), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "metric": metric,
        "difference_b_minus_a": float(differences.mean()),
        "ci95": [float(low), float(high)],
        "users": len(differences),
        "resamples": samples,
        "seed": seed,
    }
