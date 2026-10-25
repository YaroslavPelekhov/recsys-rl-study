from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rec.task2_data import (
    DAY,
    SPECIAL,
    encode,
    histories,
    make_queries,
    return_labels,
    training_arrays,
    vocabulary,
)
from rec.task2_metrics import paired_interval, ranking_rows


def events():
    return pd.DataFrame(
        {
            "uid": [7] * 5,
            "timestamp": [1, 1, 100, 4000, DAY + 1],
            "item_id": [10, 20, 10, 30, 999],
            "is_organic": [0] * 5,
            "played_ratio_pct": [100, 0, 80, 50, 100],
            "track_length_seconds": [100, 200, 100, 300, 1],
        }
    )


def test_catalog_metadata_future_invariance():
    frame = events()
    before = vocabulary(frame, limit=2, cutoff=200)
    frame.loc[frame.timestamp >= 200, ["item_id", "track_length_seconds", "played_ratio_pct"]] = [
        10,
        99999,
        0,
    ]
    after = vocabulary(frame, limit=2, cutoff=200)
    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(encode([10, 20, 999], before[0]), [3, 4, 1])


def test_equal_timestamp_excluded_from_supervised_order_and_query_history():
    frame = events()
    h = histories(frame, np.array([10, 20, 30]))
    arrays = training_arrays(h)
    assert arrays["y"][0, 0] == -100  # no 10 -> 20 within tied time
    assert arrays["y"][0, 1] == 3  # both tied events can precede timestamp100
    meta, contexts, labels = make_queries(h, [7], [0], True)
    assert meta.timestamp.tolist() == [100, 4000]
    assert contexts[0, -2:].tolist() == [3, 4]
    assert labels.token.tolist() == [3, 5]
    changed = frame.copy()
    changed.loc[changed.timestamp >= 100, "item_id"] = 888
    _, mutated, _ = make_queries(histories(changed, np.array([10, 20, 30])), [7], [0], False)
    np.testing.assert_array_equal(contexts[0], mutated[0])


def test_oov_kept_as_miss_and_single_target_metrics():
    values = ranking_rows([[3, 4], [4, 5], [5, 4]], [1, 5, 5])
    np.testing.assert_allclose(values.ndcg, [0, 1 / np.log2(3), 1])
    assert values.recall.mean() == 2 / 3
    assert values["map"].tolist() == [0, 0.5, 1]
    result = paired_interval([0, 1, 1], [0, 0, 0], [7, 7, 8])
    assert result["users"] == 2
    assert result["delta"] == 0.75


def test_return24_has_no_same_session_or_after_horizon_shortcut():
    times = np.array([0, 100, 3600, DAY, 2 * DAY])
    np.testing.assert_array_equal(return_labels(times), [1, 1, 1, 0, 0])


def test_causal_gpt_and_policy_gradient_direction():
    torch = pytest.importorskip("torch")
    from rec.task2_models import ItemGPT2, reinforce_loss, reward_value

    torch.manual_seed(1)
    model = ItemGPT2(11, width=16, heads=2, layers=1, dropout=0).eval()
    left = torch.tensor([[3, 4, 5, 6]])
    right = torch.tensor([[3, 4, 9, 10]])
    torch.testing.assert_close(model(left)[:, :2], model(right)[:, :2])
    padded = torch.tensor([[0, 0, 3, 4]])
    torch.testing.assert_close(model(padded, last_only=True), model(left[:, :2], last_only=True))
    logits = torch.zeros(2, requires_grad=True)
    logps = logits.log_softmax(0).reshape(2, 1)
    loss = reinforce_loss(logps, torch.tensor([[1.0], [0.0]]))
    loss.backward()
    assert logits.grad[0] < 0 < logits.grad[1]
    assert (model(left, last_only=True).argmax(-1) >= SPECIAL).all()
    predicted = torch.tensor([[[0.9, 0.2]], [[0.8, 0.3]], [[0.7, 0.2]]])
    actual = predicted.clone()
    actual[..., 0] -= 0.3
    base = reward_value("ensemble", predicted, actual, torch.tensor([100.0]))
    anchored = reward_value("anchored", predicted, actual, torch.tensor([100.0]))
    torch.testing.assert_close(base - anchored, torch.tensor([0.3]))
