"""Candidate-conditioned pooling of transition density-ratio experts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from rec.transition import TransitionEmbedding, historical_profiles, transition_pairs

VARIANTS = ("G0", "G1", "A0", "A1")
LOG_FLOOR = float(np.log(1e-12))


def pool_experts(logits, weights, log_background):
    """Four factorial scores on a common, strictly positive target support."""
    logits = np.asarray(logits, np.float64)
    weights = np.asarray(weights, np.float64)
    background = np.asarray(log_background, np.float64)
    if logits.shape != (len(weights), len(background)) or not len(weights):
        raise ValueError("Invalid expert matrix")
    if not np.isfinite(logits).all() or not np.isfinite(background).all():
        raise ValueError("Expert logits and background must be finite")
    if (weights <= 0).any() or not np.isfinite(weights).all():
        raise ValueError("Expert weights must be positive")
    weights = weights / weights.sum()
    result = {}
    for restored in (0, 1):
        values = logits + restored * background[None, :]
        log_probabilities = values - logsumexp(values, axis=1, keepdims=True)
        result[f"G{restored}"] = np.maximum(weights @ log_probabilities, LOG_FLOOR)
        result[f"A{restored}"] = np.maximum(
            logsumexp(log_probabilities + np.log(weights)[:, None], axis=0), LOG_FLOOR
        )
    return result


def reconstructed_scores(candidates, history, encoder: TransitionEmbedding, cutoff):
    """No labels or future catalog; q and role masks use the encoder's past pairs."""
    if not (history.timestamp < cutoff).all():
        raise ValueError("Reconstruction requires strictly historical events")
    pairs = transition_pairs(history, cutoff)
    users = pd.Index(np.sort(history.user_id.unique()))
    catalog = pd.Index(np.sort(history.item_id.unique()))
    profile = historical_profiles(history, users, catalog, cutoff)[24]
    locations = catalog.get_indexer(encoder.items)
    if (locations < 0).any():
        raise ValueError("Encoder vocabulary outside historical catalog")
    original_vectors = encoder.user_vectors(profile, catalog)
    result = candidates[["user_id", "item_id"]].copy().reset_index(drop=True)
    result["legacy"] = encoder.pair_scores(
        original_vectors, users.get_indexer(result.user_id), result.item_id.to_numpy()
    )
    for name in VARIANTS:
        result[name] = LOG_FLOOR
    counts = len(encoder.items)
    source_rows = encoder.items.get_indexer(pairs.source)
    target_rows = encoder.items.get_indexer(pairs.target)
    if (source_rows < 0).any() or (target_rows < 0).any():
        raise ValueError("Past pair missing from encoder vocabulary")
    source_count = np.bincount(source_rows, minlength=counts)
    target_count = np.bincount(target_rows, minlength=counts)
    compact_profile = profile[:, locations].tocsr()
    audit = {
        "queries": 0,
        "empty_source_queries": 0,
        "empty_target_queries": 0,
        "unsupported_candidate_pairs": 0,
        "candidate_pairs": len(result),
    }
    with threadpool_limits(1, user_api="blas"):
        for uid, indices in result.groupby("user_id", sort=False).indices.items():
            audit["queries"] += 1
            row = compact_profile.getrow(users.get_loc(uid))
            keep = (source_count[row.indices] > 0) & (row.data > 0)
            sources, weights = row.indices[keep], row.data[keep]
            targets = encoder.items.get_indexer(result.item_id.to_numpy()[indices])
            valid = targets >= 0
            valid[valid] &= target_count[targets[valid]] > 0
            audit["unsupported_candidate_pairs"] += int((~valid).sum())
            if not valid.any():
                audit["empty_target_queries"] += 1
                result.loc[indices, list(VARIANTS)] = 0.0
                continue
            selected = indices[valid]
            targets = targets[valid]
            log_q = np.log(target_count[targets].astype(np.float64))
            if not len(sources):
                audit["empty_source_queries"] += 1
                fallback = np.maximum(log_q - logsumexp(log_q), LOG_FLOOR)
                for name in VARIANTS:
                    result.loc[selected, name] = fallback
                continue
            logits = (encoder.source[sources] @ encoder.target[targets].T).astype(
                np.float64
            ) / np.sqrt(encoder.source.shape[1])
            pooled = pool_experts(logits, weights, log_q)
            for name in VARIANTS:
                result.loc[selected, name] = pooled[name]
    if not np.isfinite(result[list(VARIANTS) + ["legacy"]].to_numpy()).all():
        raise ValueError("Non-finite reconstructed score")
    audit["history_cutoff_exclusive"] = cutoff
    audit["history_max_timestamp"] = int(history.timestamp.max())
    audit["targets_used"] = False
    return result, audit
