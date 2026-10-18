"""
Align two models' PCA components so their colors compare, and score their similarity.

Purpose:

- Match model B's score columns to model A's by Hungarian assignment on absolute correlation.
- Color pane B with the aligned components or with one joint PCA fit on both models.
- Compute linear CKA between two layers' features.

Special Notes:

- Pure NumPy and SciPy with arrays in and arrays out. No torch and no file IO.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from layerviz.reduce import Reduced, fit_pca, project, rgb_from_scores

VARIANCE_TOL = 1e-12


def _unit_columns(scores: np.ndarray) -> np.ndarray:
    """Center each column and scale it to unit norm, leaving constant columns at zero."""
    x = np.asarray(scores, dtype=np.float64)
    x = x - x.mean(axis=0)
    norms = np.linalg.norm(x, axis=0)
    return x / np.where(norms > 0, norms, 1.0)


def _correlation(scores_a: np.ndarray, scores_b: np.ndarray) -> np.ndarray:
    """Return the Pearson correlation between every column of scores_a and of scores_b."""
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores need the same rows to match, got {len(scores_a)} and {len(scores_b)}"
        )
    if len(scores_a) == 0:
        raise ValueError("scores need at least one row to match, got 0")
    return _unit_columns(scores_a).T @ _unit_columns(scores_b)


def match_components(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Match model B's score columns to model A's by largest total absolute correlation.

    Args:
        scores_a: Model A scores of shape [N, ka].
        scores_b: Model B scores of shape [N, kb] over the same N rows.

    Returns:
        perm and signs, both of length kb. perm[j] is the column of B matched to column j of
        A, and signs[j] is +1 or -1 so the matched pair correlates positively.

    Raises:
        ValueError: When the two score arrays have different row counts or no rows.

    Notes:
        Only the first k = min(ka, kb) columns of each are matched. B columns from k on are
        appended in order with sign +1.
    """
    k = min(scores_a.shape[1], scores_b.shape[1])
    corr = _correlation(scores_a[:, :k], scores_b[:, :k])
    _, matched = linear_sum_assignment(-np.abs(corr))
    perm = np.concatenate([matched, np.arange(k, scores_b.shape[1])]).astype(np.int64)
    signs = np.ones(len(perm), dtype=np.int64)
    signs[:k] = np.where(corr[np.arange(k), matched] < 0, -1, 1)
    return perm, signs


def aligned_scores(scores_b: np.ndarray, perm: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """
    Reorder and sign flip model B's score columns to line up with model A's.

    Args:
        scores_b: Model B scores of shape [N, kb].
        perm: Column of B for each output column, from match_components.
        signs: +1 or -1 for each output column, from match_components.

    Returns:
        Scores of shape [N, kb] with the dtype of scores_b.
    """
    return scores_b[:, perm] * np.asarray(signs, dtype=scores_b.dtype)


def _signed_bytes(channel: np.ndarray, sign: int) -> np.ndarray:
    """Return a color channel unchanged for sign +1 and inverted on 0..255 for sign -1."""
    return channel if sign > 0 else 255 - channel


def aligned_rgb(reduced_a: Reduced, reduced_b: Reduced) -> np.ndarray:
    """
    Color pane B with its components matched to pane A's first three.

    Args:
        reduced_a: Reduced layer of model A.
        reduced_b: Reduced layer of model B over the same rows.

    Returns:
        uint8 array of shape [N, 3].

    Notes:
        A channel matched to one of B's first three components reuses B's own RGB bytes,
        inverted for sign -1. Stored scores are float16, so requantizing them would shift
        about 1 percent of bytes away from the colors reduce_layer gave.
    """
    perm, signs = match_components(reduced_a.scores, reduced_b.scores)
    scores = aligned_scores(reduced_b.scores.astype(np.float32), perm, signs)
    rgb = rgb_from_scores(scores)
    for j in range(min(3, len(perm))):
        if perm[j] < 3:
            rgb[:, j] = _signed_bytes(reduced_b.rgb[:, perm[j]], signs[j])
    return rgb


def joint_rgb(z_a: np.ndarray, z_b: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Color both models from one PCA fit on their stacked standardized features.

    Args:
        z_a: Model A standardized features of shape [Na, C].
        z_b: Model B standardized features of shape [Nb, C].
        seed: Seed for row subsampling in the PCA fit.

    Returns:
        rgb_a of shape [Na, 3] and rgb_b of shape [Nb, 3], both uint8 on one color scale.

    Raises:
        ValueError: When z_a and z_b have different channel counts.
    """
    if z_a.shape[1] != z_b.shape[1]:
        raise ValueError(
            f"joint fit needs the same channels, got {z_a.shape[1]} and {z_b.shape[1]}"
        )
    stacked = np.concatenate([z_a, z_b])
    rgb = rgb_from_scores(project(stacked, fit_pca(stacked, seed=seed), 3))
    return rgb[: len(z_a)], rgb[len(z_a) :]


def _has_variance(gram_norm: float, raw: np.ndarray) -> bool:
    """Tell whether a centered Gram norm stands above float noise relative to the raw features."""
    return bool(gram_norm > VARIANCE_TOL * np.sum(raw**2))


def linear_cka(z_a: np.ndarray, z_b: np.ndarray) -> float:
    """
    Compute linear centered kernel alignment between two feature sets over the same rows.

    Args:
        z_a: Features of shape [N, Ca].
        z_b: Features of shape [N, Cb].

    Returns:
        Similarity in 0..1, 1 for features equal up to rotation and scale, and 0 when either
        side has no variance beyond VARIANCE_TOL of its raw squared norm.

    Raises:
        ValueError: When the two feature sets have different row counts.
    """
    if len(z_a) != len(z_b):
        raise ValueError(f"CKA needs the same rows, got {len(z_a)} and {len(z_b)}")
    raw_a = np.asarray(z_a, dtype=np.float64)
    raw_b = np.asarray(z_b, dtype=np.float64)
    x_a = raw_a - raw_a.mean(axis=0)
    x_b = raw_b - raw_b.mean(axis=0)
    norm_a = np.linalg.norm(x_a.T @ x_a)
    norm_b = np.linalg.norm(x_b.T @ x_b)
    if not (_has_variance(norm_a, raw_a) and _has_variance(norm_b, raw_b)):
        return 0.0
    cross = np.linalg.norm(x_a.T @ x_b) ** 2
    return float(np.clip(cross / (norm_a * norm_b), 0.0, 1.0))
