"""
Test the alignment of two models' components, the joint PCA colors, and linear CKA.

Purpose:

- Check that matched components make pane B colors equal pane A colors.
- Check linear CKA against a dense Gram matrix reference and the joint fit colors.
"""

import numpy as np
import pytest

from layerviz.align import (
    aligned_rgb,
    aligned_scores,
    joint_rgb,
    linear_cka,
    match_components,
)
from layerviz.reduce import PCAFit, Reduced, reduce_layer, rgb_from_scores


def skewed_scores(n_rows: int, k: int, seed: int) -> np.ndarray:
    """Build float16 score columns with distinct skewed distributions."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_rows, k))
    return (base + 0.4 * base**2).astype(np.float16)


def reduced_from_scores(scores: np.ndarray) -> Reduced:
    """Build a Reduced whose RGB comes from its stored float16 scores."""
    k = scores.shape[1]
    fit = PCAFit(
        components=np.eye(k, dtype=np.float32),
        explained=np.full(k, 1.0 / k, dtype=np.float32),
        mean=np.zeros(k, dtype=np.float32),
        std=np.ones(k, dtype=np.float32),
    )
    return Reduced(
        rgb=rgb_from_scores(scores.astype(np.float32)),
        scores=scores,
        explained=fit.explained,
        retained=1.0,
        fit=fit,
    )


def permuted_and_flipped(scores: np.ndarray, order: list[int], signs: list[int]) -> np.ndarray:
    """Reorder score columns by order and multiply each by its sign."""
    return scores[:, order] * np.asarray(signs, dtype=scores.dtype)


def skewed_features(n_rows: int, n_channels: int, seed: int) -> np.ndarray:
    """Build float32 features with decreasing channel scales and skew."""
    rng = np.random.default_rng(seed)
    scales = np.linspace(3.0, 0.5, n_channels)
    x = rng.standard_normal((n_rows, n_channels)) * scales
    return (x + 0.3 * x**2).astype(np.float32)


def dense_cka(x_a: np.ndarray, x_b: np.ndarray) -> float:
    """Compute linear CKA from centered N by N Gram matrices as an independent reference."""
    n = len(x_a)
    h = np.eye(n) - np.ones((n, n)) / n
    k = h @ (x_a @ x_a.T) @ h
    m = h @ (x_b @ x_b.T) @ h
    return float(np.sum(k * m) / (np.linalg.norm(k) * np.linalg.norm(m)))


ORDER = [4, 2, 0, 1, 3]
SIGNS = [1, -1, -1, 1, -1]


def test_match_components_recovers_order_and_signs_of_b():
    scores_a = skewed_scores(4096, 5, seed=1)
    scores_b = permuted_and_flipped(scores_a, ORDER, SIGNS)

    perm, signs = match_components(scores_a, scores_b)

    expected_perm = np.argsort(ORDER)
    np.testing.assert_array_equal(perm, expected_perm)
    np.testing.assert_array_equal(signs, np.asarray(SIGNS)[expected_perm])


def test_aligned_scores_reorders_and_flips_columns():
    scores_b = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    out = aligned_scores(scores_b, np.array([2, 0, 1]), np.array([1, -1, 1]))

    np.testing.assert_array_equal(out, [[3.0, -1.0, 2.0], [6.0, -4.0, 5.0]])


def test_match_components_appends_extra_b_columns_unmatched():
    scores_a = skewed_scores(2048, 4, seed=2)
    scores_b = np.concatenate([-scores_a[:, [1, 0]], scores_a[:, [3, 2]]], axis=1)

    perm, signs = match_components(scores_a[:, :2], scores_b)

    np.testing.assert_array_equal(perm, [1, 0, 2, 3])
    np.testing.assert_array_equal(signs, [-1, -1, 1, 1])


def test_match_components_matches_only_leading_a_columns_when_b_is_smaller():
    scores_a = skewed_scores(2048, 4, seed=3)
    scores_b = scores_a[:, [1, 0]]

    perm, signs = match_components(scores_a, scores_b)

    np.testing.assert_array_equal(perm, [1, 0])
    np.testing.assert_array_equal(signs, [1, 1])


def test_match_components_rejects_different_row_counts():
    with pytest.raises(ValueError, match="rows"):
        match_components(np.zeros((10, 3)), np.zeros((12, 3)))


def test_match_components_recovers_reversed_order_and_signs_of_32_components():
    scores_a = skewed_scores(1024, 32, seed=4)
    reversed_order = list(range(31, -1, -1))
    alternating = [1, -1] * 16
    scores_b = permuted_and_flipped(scores_a, reversed_order, alternating)

    perm, signs = match_components(scores_a, scores_b)

    np.testing.assert_array_equal(perm, reversed_order)
    np.testing.assert_array_equal(signs, [-1, 1] * 16)


def test_aligned_rgb_equals_a_rgb_when_b_scores_are_permuted_and_flipped():
    scores_a = skewed_scores(4096, 5, seed=5)
    reduced_a = reduced_from_scores(scores_a)
    reduced_b = reduced_from_scores(permuted_and_flipped(scores_a, ORDER, SIGNS))

    rgb = aligned_rgb(reduced_a, reduced_b)

    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb, reduced_a.rgb)


def test_aligned_rgb_equals_a_rgb_when_b_features_are_permuted_and_flipped():
    features = skewed_features(4096, 16, seed=6)
    rng = np.random.default_rng(7)
    channel_order = rng.permutation(16)
    channel_signs = rng.choice([-1.0, 1.0], 16).astype(np.float32)
    reduced_a = reduce_layer(features)
    reduced_b = reduce_layer(features[:, channel_order] * channel_signs)

    rgb = aligned_rgb(reduced_a, reduced_b)

    np.testing.assert_array_equal(rgb, reduced_a.rgb)


def test_joint_rgb_gives_identical_colors_for_identical_inputs():
    z = skewed_features(1024, 8, seed=8)

    rgb_a, rgb_b = joint_rgb(z, z)

    assert rgb_a.shape == (1024, 3) and rgb_a.dtype == np.uint8
    assert len(np.unique(rgb_a[:, 0])) > 100
    np.testing.assert_array_equal(rgb_a, rgb_b)


def test_joint_rgb_splits_colors_of_one_fit_on_the_stacked_rows():
    # Each row loads on at most one channel, so the stacked fit's components are the
    # channel axes and each color channel is 255 exactly where its channel is nonzero.
    z_a = np.array([[4, 0, 0], [4, 0, 0], [0, 2, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float32)
    z_b = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.float32)

    rgb_a, rgb_b = joint_rgb(z_a, z_b)

    np.testing.assert_array_equal(
        rgb_a, [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 255, 0], [0, 0, 255]]
    )
    np.testing.assert_array_equal(rgb_b, [[0, 0, 255], [0, 0, 0], [0, 0, 0]])


def test_linear_cka_of_features_with_themselves_is_one():
    x = skewed_features(500, 12, seed=11)

    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_linear_cka_of_features_with_themselves_is_one_after_a_large_offset(dtype):
    x = (np.random.default_rng(16).standard_normal((500, 12)) + 1e6).astype(dtype)

    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-6)


def test_linear_cka_of_independent_gaussians_is_small():
    rng = np.random.default_rng(12)
    x_a = rng.standard_normal((2000, 8))
    x_b = rng.standard_normal((2000, 8))

    assert linear_cka(x_a, x_b) < 0.05


def test_linear_cka_matches_dense_gram_reference():
    rng = np.random.default_rng(13)
    x_a = rng.standard_normal((300, 10)) + 2.0
    x_b = np.tanh(x_a @ rng.standard_normal((10, 6))) + 0.5 * rng.standard_normal((300, 6))

    assert linear_cka(x_a, x_b) == pytest.approx(dense_cka(x_a, x_b), abs=1e-4)


NOISE = np.random.default_rng(14).standard_normal((50, 3))
FLAT_SIDES = {
    "constant": np.ones((50, 3)),
    "float64 noise around 1": 1.0 + 1e-9 * NOISE,
    "float32 rounding noise around 1000": (1000.0 + 3e-5 * NOISE).astype(np.float32),
}


@pytest.mark.parametrize("flat", FLAT_SIDES.values(), ids=FLAT_SIDES.keys())
@pytest.mark.parametrize("flat_side", ["a", "b"])
def test_linear_cka_is_zero_when_one_side_has_no_signal(flat, flat_side):
    pair = (flat, NOISE) if flat_side == "a" else (NOISE, flat)

    assert linear_cka(*pair) == 0.0


def test_linear_cka_rejects_different_row_counts():
    with pytest.raises(ValueError, match="rows"):
        linear_cka(np.zeros((10, 3)), np.zeros((12, 3)))


def test_match_components_rejects_empty_scores():
    with pytest.raises(ValueError, match="at least one row"):
        match_components(np.zeros((0, 3)), np.zeros((0, 3)))


def test_joint_rgb_rejects_different_channel_counts():
    with pytest.raises(ValueError, match="channels"):
        joint_rgb(np.zeros((10, 4)), np.zeros((10, 5)))
