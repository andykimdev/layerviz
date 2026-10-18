"""
Test the PCA reduction of one layer's features into RGB, scores, and explained variance.

Purpose:

- Check determinism, explained variance, the k rule, and degenerate inputs (A10 to A13).
- Check channel quantization under a byte cap.
"""

import numpy as np
import pytest

from layerviz.reduce import (
    bad_values,
    channel_bytes,
    fit_pca,
    project,
    reduce_layer,
    rgb_from_scores,
    standardize,
)


def make_features(n, c, seed=0):
    """
    Build skewed, correlated float32 features so PCA signs are well defined.

    Args:
        n: Number of positions.
        c: Number of channels.
        seed: Seed for the random generator.

    Returns:
        float32 array of shape [n, c].
    """
    rng = np.random.default_rng(seed)
    latent = rng.exponential(size=(n, c)) * np.linspace(3.0, 0.5, c)
    mixing = rng.standard_normal((c, c))
    return (latent @ mixing + rng.standard_normal(c)).astype(np.float32)


def reference_explained(features):
    """
    Compute the explained variance ratio with a direct float64 SVD.

    Args:
        features: Array of shape [N, C].

    Returns:
        Tuple of the explained ratios [min(C, N - 1)] and the scores U * s.
    """
    x = features.astype(np.float64)
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    k_full = min(x.shape[1], x.shape[0] - 1)
    return (s**2 / np.sum(s**2))[:k_full], (u * s)[:, :k_full]


def skewness(values):
    """
    Return the sample skewness of a 1D array, or 0 when it has no variance.

    Args:
        values: 1D array.

    Returns:
        Skewness as a float.
    """
    centered = values - values.mean()
    m2 = np.mean(centered**2)
    return 0.0 if m2 == 0 else float(np.mean(centered**3) / m2**1.5)


def test_reduce_layer_gives_identical_rgb_on_repeat_calls():
    features = make_features(200, 16)

    first = reduce_layer(features)
    second = reduce_layer(features.copy())

    assert first.rgb.dtype == np.uint8
    assert first.rgb.shape == (200, 3)
    assert first.rgb.tobytes() == second.rgb.tobytes()


def test_reduce_layer_rgb_ignores_channel_sign_flips():
    features = make_features(200, 16)
    flips = np.where(np.random.default_rng(1).random(16) < 0.5, -1.0, 1.0)

    original = reduce_layer(features)
    flipped = reduce_layer((features * flips).astype(np.float32))

    assert np.array_equal(original.rgb, flipped.rgb)


def test_reduce_layer_scores_have_non_negative_skewness():
    reduced = reduce_layer(make_features(300, 8))

    skews = [skewness(reduced.scores[:, i].astype(np.float64)) for i in range(8)]

    assert min(skews) >= 0.0


def test_reduce_layer_explained_matches_reference_svd():
    features = make_features(100, 12)
    expected, _ = reference_explained(features)

    reduced = reduce_layer(features)

    assert reduced.explained.dtype == np.float32
    np.testing.assert_allclose(reduced.explained, expected, atol=1e-4)


def test_reduce_layer_retained_is_explained_sum_over_stored_k():
    features = make_features(100, 64)
    expected, _ = reference_explained(features)

    reduced = reduce_layer(features)

    assert reduced.scores.shape[1] == 32
    assert reduced.retained == pytest.approx(float(np.sum(expected[:32])), abs=1e-4)
    assert reduced.retained < 1.0 - 1e-3


def test_reduce_layer_scores_match_reference_up_to_sign():
    features = make_features(100, 12)
    _, expected = reference_explained(features)

    reduced = reduce_layer(features)

    assert reduced.scores.dtype == np.float16
    np.testing.assert_allclose(
        np.abs(reduced.scores.astype(np.float64)), np.abs(expected), rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize(
    ("n", "c", "k"), [(100, 64, 32), (100, 8, 8), (5, 64, 4), (1, 8, 0)]
)
def test_reduce_layer_stores_k_of_min_32_c_and_n_minus_1(n, c, k):
    reduced = reduce_layer(make_features(n, c))

    assert reduced.scores.shape == (n, k)
    assert reduced.explained.shape == (min(c, n - 1),)
    assert reduced.rgb.shape == (n, 3)


def test_reduce_layer_zeroes_rgb_columns_without_a_component():
    reduced = reduce_layer(make_features(50, 2))

    assert np.array_equal(reduced.rgb[:, 2], np.zeros(50, dtype=np.uint8))
    assert reduced.rgb[:, :2].max(axis=0).tolist() == [255, 255]
    assert reduced.rgb[:, :2].min(axis=0).tolist() == [0, 0]


def test_reduce_layer_handles_zero_variance_channel():
    features = make_features(60, 6)
    features[:, 2] = 4.0

    reduced = reduce_layer(features)

    assert np.all(np.isfinite(reduced.scores))
    assert np.all(np.isfinite(reduced.explained))
    assert np.isfinite(reduced.retained)
    assert reduced.explained[-1] == pytest.approx(0.0, abs=1e-6)
    assert float(np.sum(reduced.explained)) == pytest.approx(1.0, abs=1e-5)


def test_reduce_layer_handles_constant_features():
    features = np.full((40, 5), 7.0, dtype=np.float32)

    reduced = reduce_layer(features)

    assert np.array_equal(reduced.scores, np.zeros((40, 5), dtype=np.float16))
    assert np.array_equal(reduced.explained, np.zeros(5, dtype=np.float32))
    assert reduced.retained == 0.0
    assert np.array_equal(reduced.rgb, np.zeros((40, 3), dtype=np.uint8))


def test_reduce_layer_handles_one_position():
    reduced = reduce_layer(make_features(1, 8))

    assert reduced.scores.shape == (1, 0)
    assert reduced.explained.shape == (0,)
    assert reduced.retained == 0.0
    assert np.array_equal(reduced.rgb, np.zeros((1, 3), dtype=np.uint8))


def test_reduce_layer_rejects_nan_input():
    features = make_features(20, 4)
    features[3, 1] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        reduce_layer(features)


def test_reduce_layer_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2D"):
        reduce_layer(np.zeros((4, 3, 2), dtype=np.float32))


def test_bad_values_names_nan():
    features = make_features(20, 4)
    features[0, 0] = np.nan
    features[5, 2] = np.nan

    assert bad_values(features) == "2 NaN values"


def test_bad_values_names_inf():
    features = make_features(20, 4)
    features[1, 3] = -np.inf

    assert bad_values(features) == "1 inf values"


def test_bad_values_names_nan_and_inf_together():
    features = make_features(20, 4)
    features[0, 0] = np.nan
    features[1, 1] = np.inf

    assert bad_values(features) == "1 NaN values, 1 inf values"


def test_bad_values_returns_none_for_finite_input():
    assert bad_values(make_features(20, 4)) is None


def test_standardize_gives_unit_std_and_zero_for_constant_channel():
    features = make_features(50, 4)
    features[:, 1] = -2.0

    z, mean, std = standardize(features)

    assert z.dtype == np.float32
    np.testing.assert_allclose(mean, features.astype(np.float64).mean(axis=0), rtol=1e-5)
    assert std[1] == 0.0
    assert np.array_equal(z[:, 1], np.zeros(50, dtype=np.float32))
    np.testing.assert_allclose(z[:, [0, 2, 3]].std(axis=0), 1.0, atol=1e-5)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-5)


def test_fit_pca_without_subsampling_ignores_max_rows():
    z, _, _ = standardize(make_features(80, 6))

    exact = fit_pca(z, max_rows=80)
    loose = fit_pca(z, max_rows=8000)

    assert np.array_equal(exact.components, loose.components)
    assert np.array_equal(exact.explained, loose.explained)


def test_fit_pca_subsamples_to_max_rows_deterministically():
    z, _, _ = standardize(make_features(200, 64))

    first = fit_pca(z, max_rows=30, seed=3)
    second = fit_pca(z, max_rows=30, seed=3)

    assert first.components.shape == (29, 64)
    assert np.array_equal(first.components, second.components)


def test_fit_pca_breaks_skewness_tie_with_positive_largest_loading():
    # Symmetric scores have zero skewness, so the loading rule decides the sign.
    z = np.array([[-1.0, 0.5], [1.0, -0.5], [-1.0, 0.5], [1.0, -0.5]], dtype=np.float32)

    fit = fit_pca(z)

    largest = fit.components[0, np.argmax(np.abs(fit.components[0]))]
    assert largest > 0


def test_project_returns_float32_scores_for_first_k_components():
    z, _, _ = standardize(make_features(50, 6))
    fit = fit_pca(z)

    scores = project(z, fit, 3)

    assert scores.dtype == np.float32
    np.testing.assert_allclose(scores, z @ fit.components[:3].T, rtol=1e-5, atol=1e-5)


def test_rgb_from_scores_clips_to_percentiles_and_scales():
    column = np.arange(101, dtype=np.float32)
    scores = np.stack([column, -column, np.zeros(101, dtype=np.float32)], axis=1)

    rgb = rgb_from_scores(scores)

    assert rgb[0].tolist() == [0, 255, 0]
    assert rgb[1].tolist() == [0, 255, 0]
    assert rgb[50].tolist() == [128, 128, 0]
    assert rgb[100].tolist() == [255, 0, 0]


def test_channel_bytes_keeps_all_channels_under_the_cap():
    features = make_features(50, 6)

    data, kept = channel_bytes(features, cap_bytes=50 * 6)

    assert data.dtype == np.uint8
    assert data.shape == (50, 6)
    assert kept.tolist() == [0, 1, 2, 3, 4, 5]
    assert data.min(axis=0).tolist() == [0] * 6
    assert data.max(axis=0).tolist() == [255] * 6


def test_channel_bytes_keeps_top_variance_channels_over_the_cap():
    rng = np.random.default_rng(0)
    scales = np.array([1.0, 9.0, 2.0, 7.0, 0.5], dtype=np.float32)
    features = (rng.standard_normal((40, 5)) * scales).astype(np.float32)

    data, kept = channel_bytes(features, cap_bytes=40 * 2 + 39)

    assert kept.tolist() == [1, 3]
    assert data.shape == (40, 2)
    expected, _ = channel_bytes(features[:, [1, 3]], cap_bytes=10**6)
    assert np.array_equal(data, expected)


def test_channel_bytes_zeroes_constant_channel():
    features = make_features(30, 3)
    features[:, 0] = 1.5

    data, _ = channel_bytes(features, cap_bytes=10**6)

    assert np.array_equal(data[:, 0], np.zeros(30, dtype=np.uint8))


def test_reduce_layer_zeroes_rgb_columns_1_and_2_for_one_channel():
    reduced = reduce_layer(make_features(50, 1))

    assert reduced.scores.shape == (50, 1)
    assert np.array_equal(reduced.rgb[:, 1:], np.zeros((50, 2), dtype=np.uint8))
    assert reduced.rgb[:, 0].max() == 255
    assert reduced.rgb[:, 0].min() == 0


def test_channel_bytes_rejects_non_2d_input():
    with pytest.raises(ValueError, match=r"\(4, 3, 2\)"):
        channel_bytes(np.zeros((4, 3, 2), dtype=np.float32), cap_bytes=100)
