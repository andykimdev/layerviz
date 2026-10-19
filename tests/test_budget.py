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

from layerviz import budget, framing, reduce
from layerviz.budget import (
    BYTES_PER_POSITION,
    COMPARE_BYTES_PER_POSITION,
    grid_key,
    plan_keep,
    position_bytes,
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


@pytest.mark.parametrize(
    ("dim", "joint", "expected"),
    [(3, False, 79), (3, True, 82), (2, False, 75), (2, True, 78), (1, False, 71)],
)
def test_position_bytes_counts_coordinates_rgb_scores_and_joint_rgb(dim, joint, expected):
    assert position_bytes(dim, joint) == expected


def test_score_count_is_one_object_across_reduce_budget_and_framing():
    assert budget.SCORE_COUNT is reduce.SCORE_COUNT
    assert framing.SCORE_COUNT is reduce.SCORE_COUNT


@pytest.mark.parametrize(
    ("sizes", "total_bytes"),
    [(79, 3 * 79), ({"a": 75, "b": 79, "c": 100, "empty": 120}, 75 + 79 + 100)],
)
def test_plan_keep_total_of_one_position_per_layer_keeps_one_each_without_warning(
    sizes, total_bytes
):
    counts = {"a": 100, "b": 100, "c": 100, "empty": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        keep = plan_keep(counts, 16, total_bytes / 2**20, sizes)
    assert keep == {"a": 1, "b": 1, "c": 1, "empty": 0}


@pytest.mark.parametrize(
    ("sizes", "total_bytes"),
    [(79, 2 * 79), ({"a": 75, "b": 79, "c": 100, "empty": 120}, 75 + 79 + 100 - 1)],
)
def test_plan_keep_total_one_position_short_keeps_one_each_and_warns(sizes, total_bytes):
    counts = {"a": 100, "b": 100, "c": 100, "empty": 0}
    with pytest.warns(UserWarning, match=r"fewer than the 3 positions"):
        keep = plan_keep(counts, 16, total_bytes / 2**20, sizes)
    assert keep == {"a": 1, "b": 1, "c": 1, "empty": 0}


def test_plan_keep_size_mapping_gives_the_larger_layer_fewer_positions():
    # 1 MB less the first position of each layer leaves 1048422 bytes, 524211 per layer.
    # That is 6989 more positions of 75 bytes and 6635 more of 79 bytes.
    sizes = {"flat": 75, "deep": 79}
    keep = plan_keep({"flat": 10**6, "deep": 10**6}, 16, 1, sizes)
    assert keep == {"flat": 6990, "deep": 6636}
    assert sum(keep[name] * sizes[name] for name in keep) <= 2**20


def test_plan_keep_size_mapping_without_a_non_empty_layer_raises_key_error_naming_it():
    match = r"bytes_per_position mapping has no size for layer 'missing'"
    with pytest.raises(KeyError, match=match):
        plan_keep({"present": 10, "missing": 10}, 16, 1, {"present": 79})


def test_plan_keep_size_mapping_without_an_empty_layer_plans_it_at_zero():
    keep = plan_keep({"present": 10, "empty": 0}, 16, 1, {"present": 79})
    assert keep == {"present": 10, "empty": 0}


def test_positions_in_16_mb_at_compare_bytes_per_position():
    assert positions_in(16, COMPARE_BYTES_PER_POSITION) == 204600


def test_positions_in_size_mapping_counts_positions_of_the_largest_size():
    assert positions_in(1, {"flat": 75, "deep": 79}) == 13273


def test_positions_in_empty_size_mapping_raises_value_error():
    with pytest.raises(ValueError, match="empty mapping"):
        positions_in(1, {})


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


def test_plan_keep_empty_counts_returns_empty():
    assert plan_keep({}) == {}


def test_plan_keep_layer_and_total_caps_give_the_written_out_plan():
    # 2 MB caps a layer at 26546 positions and 3 MB holds 39819. After a keeps 7, b and c
    # split the remaining 39812.
    keep = plan_keep({"a": 7, "b": 123456, "c": 99999}, 2, 3)
    assert keep == {"a": 7, "b": 19906, "c": 19906}


def test_grid_key_holds_kind_grid_and_count():
    assert grid_key(make_layer(10, grid=(2, 5))) == ("voxel", (2, 5), 10)


def test_subsample_same_grid_keeps_the_written_out_rows():
    # Rows drawn by a NumPy generator seeded from sha256 of (0, ('voxel', (4, 4, 4), 20)).
    first, idx_a = subsample(make_layer(20), 5)
    second, idx_b = subsample(make_layer(20, labels=["x"]), 5)
    np.testing.assert_array_equal(idx_a, [9, 10, 12, 13, 14])
    np.testing.assert_array_equal(idx_b, [9, 10, 12, 13, 14])
    np.testing.assert_array_equal(first.positions[:, 0], [9, 10, 12, 13, 14])
    np.testing.assert_array_equal(second.positions[:, 0], [9, 10, 12, 13, 14])


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
