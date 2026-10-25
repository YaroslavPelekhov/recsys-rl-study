"""Safety properties of the new support and historical-outcome features."""

import numpy as np

from rec.task2b import item_indices, record, reward_features


def small_history():
    return dict(
        t=np.array([10, 20, 20, 40]),
        item=np.array([100, 200, 300, 400]),
        ratio=np.array([0.2, 0.4, 0.6, 0.9], np.float32),
        organic=np.array([0, 1, 0, 0]),
    )


def test_query_excludes_target_and_all_tied_events():
    history = small_history()
    cat = np.array([100, 200, 300, 400])
    first = record(history, 1, cat, cat[:2])
    history["ratio"][1:] = 0
    history["item"][1:] = 999
    second = record(history, 1, cat, cat[:2])
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert first[0][-1] == 0 and (first[0][:-1] == -1).all()


def test_catalog_membership_has_no_searchsorted_alias():
    np.testing.assert_array_equal(
        item_indices(np.array([5, 10, 11, 30, 100]), np.array([10, 30])), [-1, 0, -1, 1, -1]
    )


def test_reward_features_do_not_read_target_or_current_outcome():
    ids, outcome, state = record(small_history(), 3, np.array([100, 200, 300]), np.array([100]))
    data = dict(
        history=ids[None],
        outcome=outcome[None],
        state=state[None],
        target=np.array([3]),
        ratio=np.array([0.9]),
    )
    cat = dict(
        items=np.array([100, 200, 300]),
        count200=np.array([2, 1, 3]),
        ratio200=np.array([0.3, 0.5, 0.8]),
        lengths=np.array([60, 120, 180]),
        global_ratio=0.5,
    )
    first = reward_features(data, np.array([[0, 1, -1]]), cat)
    data["target"][:] = -999
    data["ratio"][:] = 0
    np.testing.assert_array_equal(first, reward_features(data, np.array([[0, 1, -1]]), cat))
    assert np.isfinite(first).all()
    assert first.shape == (3, 20)
