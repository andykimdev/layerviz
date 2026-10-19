"""
Test how reduced layers are stored as quantized channel bytes and float16 scores.

Purpose:

- Check channel quantization under a byte cap and the cap checks.
- Check that stored float16 scores stay finite for extreme input.
"""

import numpy as np
import pytest

from layerviz.reduce import channel_bytes, reduce_layer, to_float16


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
    low, high = np.percentile(features[:, [1, 3]].astype(np.float64), [1, 99], axis=0)
    unit = np.clip((features[:, [1, 3]] - low) / (high - low), 0.0, 1.0)
    assert np.array_equal(data, np.rint(unit * 255).astype(np.uint8))


def test_channel_bytes_rejects_a_negative_cap():
    with pytest.raises(ValueError, match="cap_bytes must be at least 0, got -1"):
        channel_bytes(make_features(10, 3), cap_bytes=-1)


def test_channel_bytes_keeps_no_channel_under_a_zero_cap():
    data, kept = channel_bytes(make_features(10, 3), cap_bytes=0)

    assert data.dtype == np.uint8
    assert data.shape == (10, 0)
    assert kept.tolist() == []


def test_channel_bytes_zeroes_constant_channel():
    features = make_features(30, 3)
    features[:, 0] = 1.5

    data, _ = channel_bytes(features, cap_bytes=10**6)

    assert np.array_equal(data[:, 0], np.zeros(30, dtype=np.uint8))


def test_channel_bytes_rejects_non_2d_input():
    with pytest.raises(ValueError, match=r"\(4, 3, 2\)"):
        channel_bytes(np.zeros((4, 3, 2), dtype=np.float32), cap_bytes=100)


def test_to_float16_clips_scores_to_the_float16_range():
    scores = np.array([[1e6, -1e6, 1.5]], dtype=np.float32)

    stored = to_float16(scores)

    assert stored.dtype == np.float16
    assert stored.tolist() == [[65504.0, -65504.0, 1.5]]


def test_reduce_layer_stores_finite_scores_for_extreme_float32_input():
    top = np.finfo(np.float32).max
    features = make_features(60, 3)
    features[:, 0] = np.where(np.arange(60) % 2 == 0, top, -top)
    features[:, 1] = -top
    features[0, 1] = top

    reduced = reduce_layer(features)

    assert reduced.scores.dtype == np.float16
    assert reduced.scores.shape == (60, 3)
    assert np.isfinite(reduced.scores).all()
