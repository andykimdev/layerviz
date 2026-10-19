"""
Test the position budget that caps each layer and the whole bundle.

Purpose:

- Check water filling of the total budget and the per layer cap.
- Check the 1 position floor and warning when the total budget is too small.
- Check that each layer cap reserves the layer file overhead it is given.
- Check that layers on the same grid keep the same subsampled positions.
"""

import warnings

import numpy as np
import pytest

from layerviz.budget import (
    BYTES_PER_POSITION,
    COMPARE_BYTES_PER_POSITION,
    grid_key,
    plan_keep,
    positions_in,
    subsample,
)
from layerviz.shapes.base import LayerData


def make_layer(n, grid=(4, 4, 4), kind="voxel", labels=None):
    """
    Build a layer whose position row i holds the value i, so kept rows are traceable.

    Args:
        n: Number of positions.
        grid: Grid shape stored on the layer.
        kind: Shape kind stored on the layer.
        labels: Optional per position labels.

    Returns:
        A LayerData with n positions and 2 feature channels.
    """
    rows = np.arange(n, dtype=np.float32)
    positions = np.stack([rows, rows, rows], axis=1)
    features = np.stack([rows, -rows], axis=1)
    return LayerData(kind, positions, features, grid, (1.0, 1.0, 1.0), labels)


def test_bytes_per_position_counts_xyz_rgb_and_scores():
    assert BYTES_PER_POSITION == 79


def test_compare_bytes_per_position_adds_three_bytes_of_joint_rgb():
    assert COMPARE_BYTES_PER_POSITION == 82


def test_positions_in_16_mb_at_compare_bytes_per_position():
    assert positions_in(16, COMPARE_BYTES_PER_POSITION) == 204600


def test_plan_keep_compare_bytes_per_position_lowers_the_layer_cap():
    keep = plan_keep({"huge": 10**6, "tiny": 10}, 16, 1024, COMPARE_BYTES_PER_POSITION)
    assert keep == {"huge": 204600, "tiny": 10}


def test_plan_keep_tiny_total_warning_counts_compare_bytes_per_position():
    # 164 bytes hold 2 positions of 82 bytes, but 5 layers need 5 positions (410 bytes).
    counts = {name: 100 for name in "abcde"}
    match = r"total_mb=0\.000156403 MB fits 2 positions.*5 positions \(0\.000391006 MB\)"
    with pytest.warns(UserWarning, match=match):
        keep = plan_keep(counts, 16, 164 / 2**20, COMPARE_BYTES_PER_POSITION)
    assert keep == {name: 1 for name in "abcde"}


def test_plan_keep_small_layer_frees_budget_for_large_layers():
    # 1 MB holds 1048576 // 79 = 13273 positions. The even share is 4424, the small
    # layer uses 1000, and the two large layers split the remaining 12273.
    keep = plan_keep({"small": 1000, "big1": 50000, "big2": 50000}, layer_mb=16, total_mb=1)
    assert keep == {"small": 1000, "big1": 6136, "big2": 6136}


def test_plan_keep_mixed_set_stays_under_both_caps():
    counts = {"a": 100, "b": 50000, "c": 50000, "d": 20000}
    keep = plan_keep(counts, layer_mb=1, total_mb=2)
    layer_cap = 2**20 // BYTES_PER_POSITION
    total_cap = 2 * 2**20 // BYTES_PER_POSITION
    # 2 MB holds 26546 positions. After a keeps 100, three layers split 26446.
    assert keep == {"a": 100, "b": 8815, "c": 8815, "d": 8815}
    assert all(keep[name] <= min(layer_cap, counts[name]) for name in counts)
    assert sum(keep.values()) <= total_cap


def test_plan_keep_layer_cap_binds_when_total_is_large():
    keep = plan_keep({"huge": 100000, "tiny": 10}, layer_mb=1, total_mb=1024)
    assert keep == {"huge": 13273, "tiny": 10}


def test_plan_keep_layer_cap_reserves_each_layer_overhead():
    # 1 MB less 790 bytes of overhead holds 13263 positions of 79 bytes.
    keep = plan_keep({"huge": 100000, "bare": 100000}, 1, 1024, overhead={"huge": 790})
    assert keep == {"huge": 13263, "bare": 13273}


def test_plan_keep_overhead_over_the_layer_cap_still_keeps_one_position():
    keep = plan_keep({"a": 100, "empty": 0}, 100 / 2**20, 1024, overhead={"a": 500})
    assert keep == {"a": 1, "empty": 0}


def test_plan_keep_tiny_total_keeps_one_position_per_layer_and_warns():
    # 158 bytes hold 2 positions of 79 bytes, but 5 layers need 5 positions (395 bytes).
    counts = {"a": 100, "b": 100, "c": 100, "d": 100, "e": 100, "empty": 0}
    match = r"total_mb=0\.000150681 MB fits 2 positions.*5 positions \(0\.000376701 MB\)"
    with pytest.warns(UserWarning, match=match):
        keep = plan_keep(counts, layer_mb=16, total_mb=158 / 2**20)
    assert keep == {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "empty": 0}


def test_plan_keep_budget_with_room_for_every_layer_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        keep = plan_keep({"small": 1000, "big1": 50000, "big2": 50000}, 16, 1)
    assert keep == {"small": 1000, "big1": 6136, "big2": 6136}


def test_plan_keep_empty_counts_returns_empty():
    assert plan_keep({}) == {}


def test_plan_keep_is_reproducible():
    counts = {"a": 7, "b": 123456, "c": 99999}
    assert plan_keep(counts, 2, 3) == plan_keep(dict(counts), 2, 3)


def test_grid_key_holds_kind_grid_and_count():
    assert grid_key(make_layer(10, grid=(2, 5))) == ("voxel", (2, 5), 10)


def test_subsample_same_grid_keeps_same_indices():
    first, idx_a = subsample(make_layer(500), 40)
    second, idx_b = subsample(make_layer(500), 40)
    np.testing.assert_array_equal(idx_a, idx_b)
    np.testing.assert_array_equal(first.positions, second.positions)


def test_subsample_returns_sorted_unique_rows_of_the_layer():
    layer = make_layer(500, labels=[str(i) for i in range(500)])
    kept, idx = subsample(layer, 40)
    assert idx.shape == (40,)
    assert np.all(np.diff(idx) > 0)
    np.testing.assert_array_equal(kept.positions, layer.positions[idx])
    np.testing.assert_array_equal(kept.features, layer.features[idx])
    assert kept.labels == [str(i) for i in idx]
    assert (kept.kind, kept.grid, kept.scale) == (layer.kind, layer.grid, layer.scale)


def test_subsample_numpy_int_grid_keeps_same_rows_as_python_int_grid():
    numpy_grid = tuple(np.array([4, 4, 4], dtype=np.int64))
    _, idx_numpy = subsample(make_layer(500, grid=numpy_grid), 40)
    _, idx_python = subsample(make_layer(500, grid=(4, 4, 4)), 40)
    np.testing.assert_array_equal(idx_numpy, idx_python)


def test_grid_key_keeps_missing_grid_as_none():
    assert grid_key(make_layer(3, grid=None)) == ("voxel", None, 3)


def test_subsample_different_grid_gives_different_indices():
    _, idx_a = subsample(make_layer(500, grid=(4, 4, 4)), 40)
    _, idx_b = subsample(make_layer(500, grid=(8, 8, 8)), 40)
    assert not np.array_equal(idx_a, idx_b)


def test_subsample_different_seed_gives_different_indices():
    _, idx_a = subsample(make_layer(500), 40, seed=0)
    _, idx_b = subsample(make_layer(500), 40, seed=1)
    assert not np.array_equal(idx_a, idx_b)


@pytest.mark.parametrize("keep", [12, 50])
def test_subsample_keep_at_least_n_returns_all_rows_in_order(keep):
    layer = make_layer(12)
    kept, idx = subsample(layer, keep)
    np.testing.assert_array_equal(idx, np.arange(12))
    np.testing.assert_array_equal(kept.positions, layer.positions)
    np.testing.assert_array_equal(kept.features, layer.features)


def test_subsample_keeps_labels_that_are_not_per_position():
    layer = make_layer(100, labels=["class0", "class1"])
    kept, _ = subsample(layer, 10)
    assert kept.labels == ["class0", "class1"]
