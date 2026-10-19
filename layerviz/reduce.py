"""
Reduce one layer's [N, C] features to deterministic PCA RGB, scores, and explained variance.

Purpose:

- Fit PCA on standardized features with a sign rule so repeat captures give identical RGB.
- Quantize raw channels to bytes under a size cap and name NaN or inf input.

Special Notes:

- Pure NumPy with arrays in and arrays out. No torch and no file IO.
- Degenerate input (zero variance channels, constant features, one position) gives no NaN.
"""

from dataclasses import dataclass, replace

import numpy as np

SKEW_TIE = 1e-6
STD_FLOOR = 1e-5
SCORE_COUNT = 32


@dataclass(frozen=True)
class PCAFit:
    """
    Hold the PCA components, explained variance ratios, and standardization of one layer.

    """

    components: np.ndarray
    explained: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class Reduced:
    """
    Hold the RGB, stored scores, explained variance, and fit for one reduced layer.

    """

    rgb: np.ndarray
    scores: np.ndarray
    explained: np.ndarray
    retained: float
    fit: PCAFit


def standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Center each channel and divide it by its standard deviation.

    Args:
        features: Array of shape [N, C].

    Returns:
        Tuple of z [N, C], mean [C], and std [C], all float32.

    Notes:
        A channel whose std is at most STD_FLOOR times the larger of its absolute mean
        and the largest channel std becomes all zeros. The floor covers two cases.

        - Tiny noise around 0, which unit scaling would blow up to a full color share.
        - float32 rounding noise around a large mean, such as 1000.
    """
    x = np.asarray(features, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    varies = std > STD_FLOOR * np.maximum(np.abs(mean), np.max(std, initial=0.0))
    safe = np.where(varies, std, 1.0)
    z = np.where(varies, (x - mean) / safe, 0.0)
    return z.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _fit_rows(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    """
    Choose at most max_rows row indices in ascending order, all rows when n_rows fits.

    Args:
        n_rows: Number of available rows.
        max_rows: Largest number of rows to fit on.
        seed: Seed for np.random.default_rng.

    Returns:
        int array of row indices.
    """
    if n_rows <= max_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False))


def _skewness(scores: np.ndarray) -> np.ndarray:
    """
    Compute the skewness of each score column, 0 for a column with no variance.

    Args:
        scores: float64 array of shape [N, k].

    Returns:
        float64 array of shape [k].
    """
    centered = scores - scores.mean(axis=0)
    m2 = np.mean(centered**2, axis=0)
    m3 = np.mean(centered**3, axis=0)
    safe = np.where(m2 > 0, m2, 1.0)
    return np.where(m2 > 0, m3 / safe**1.5, 0.0)


def _orient(components: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """
    Flip each component so its score skewness is non negative.

    Args:
        components: float64 array of shape [k, C].
        rows: float64 fitted rows of shape [N, C].

    Returns:
        float64 components with the sign rule applied.

    Notes:
        On a tie, when the absolute skewness is under SKEW_TIE, the largest absolute
        loading is made positive.
    """
    if components.shape[0] == 0:
        return components
    skew = _skewness(rows @ components.T)
    largest = components[np.arange(len(components)), np.argmax(np.abs(components), axis=1)]
    tie_sign = np.where(largest < 0, -1.0, 1.0)
    signs = np.where(np.abs(skew) < SKEW_TIE, tie_sign, np.sign(skew))
    return components * signs[:, None]


def fit_pca(z: np.ndarray, max_rows: int = 50000, seed: int = 0) -> PCAFit:
    """
    Fit PCA on standardized features by SVD of at most max_rows rows.

    Args:
        z: Standardized float32 features of shape [N, C].
        max_rows: Largest number of rows passed to the SVD.
        seed: Seed for choosing rows when N exceeds max_rows.

    Returns:
        PCAFit with k_full = min(C, rows fitted - 1) components and their variance ratios.

    Notes:
        The fit describes its own input, so mean is zeros and std is ones. reduce_layer
        replaces them with the raw feature statistics.
    """
    rows = np.asarray(z, dtype=np.float64)[_fit_rows(len(z), max_rows, seed)]
    k_full = max(min(rows.shape[1], len(rows) - 1), 0)
    _, singular, vt = np.linalg.svd(rows, full_matrices=False)
    variance = singular**2
    total = variance.sum()
    ratios = variance / total if total > 0 else np.zeros_like(variance)
    components = _orient(vt[:k_full], rows)
    n_channels = rows.shape[1]
    return PCAFit(
        components=components.astype(np.float32),
        explained=ratios[:k_full].astype(np.float32),
        mean=np.zeros(n_channels, dtype=np.float32),
        std=np.ones(n_channels, dtype=np.float32),
    )


def project(z: np.ndarray, fit: PCAFit, k: int) -> np.ndarray:
    """
    Project standardized features onto the first k components.

    Args:
        z: Standardized features of shape [N, C].
        fit: PCAFit with at least k components.
        k: Number of components to keep.

    Returns:
        float32 scores of shape [N, k].
    """
    return (np.asarray(z, dtype=np.float32) @ fit.components[:k].T).astype(np.float32)


def _quantize_columns(values: np.ndarray) -> np.ndarray:
    """
    Scale each column from its 1st to 99th percentile onto 0..255.

    Args:
        values: Array of shape [N, M].

    Returns:
        uint8 array of shape [N, M]. A column whose percentiles coincide is all zeros.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return np.zeros(x.shape, dtype=np.uint8)
    low, high = np.percentile(x, [1, 99], axis=0)
    span = high - low
    safe = np.where(span > 0, span, 1.0)
    unit = np.where(span > 0, np.clip((x - low) / safe, 0.0, 1.0), 0.0)
    return np.rint(unit * 255).astype(np.uint8)


def rgb_from_scores(scores: np.ndarray) -> np.ndarray:
    """
    Map the first three score columns to RGB bytes.

    Args:
        scores: Array of shape [N, k].

    Returns:
        uint8 array of shape [N, 3]. Channels without a component (k < 3) are zero.
    """
    rgb = np.zeros((scores.shape[0], 3), dtype=np.uint8)
    used = min(scores.shape[1], 3)
    rgb[:, :used] = _quantize_columns(scores[:, :used])
    return rgb


def bad_values(features: np.ndarray) -> str | None:
    """
    Name the NaN and inf values in features so the caller can skip the layer.

    Args:
        features: Array of any shape.

    Returns:
        A description such as 2 NaN values, 1 inf values, or None when all are finite.
    """
    x = np.asarray(features)
    counts = [(int(np.isnan(x).sum()), "NaN"), (int(np.isinf(x).sum()), "inf")]
    parts = [f"{count} {name} values" for count, name in counts if count]
    return ", ".join(parts) if parts else None


def _as_2d(features: np.ndarray) -> np.ndarray:
    """
    Turn raw features into a float32 array of shape [N, C].

    Args:
        features: Array-like features.

    Returns:
        float32 array of shape [N, C].

    Raises:
        ValueError: When features are not 2D, naming the shape.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D [N, C], got shape {x.shape}")
    return x


def _parse_features(features: np.ndarray) -> np.ndarray:
    """
    Turn raw features into a finite float32 array of shape [N, C].

    Args:
        features: Array-like features.

    Returns:
        float32 array of shape [N, C].

    Raises:
        ValueError: When features are not 2D or hold NaN or inf.
    """
    x = _as_2d(features)
    problem = bad_values(x)
    if problem:
        raise ValueError(f"features of shape {x.shape} contain {problem}")
    return x


def to_float16(scores: np.ndarray) -> np.ndarray:
    """
    Clip scores to the float16 range and cast them, so no stored score is inf.

    Args:
        scores: Array of any shape.

    Returns:
        float16 array of the same shape, with values beyond 65504 set to 65504.
    """
    top = float(np.finfo(np.float16).max)
    return np.clip(np.asarray(scores), -top, top).astype(np.float16)


def reduce_layer(features: np.ndarray, k: int = SCORE_COUNT, seed: int = 0) -> Reduced:
    """
    Reduce one layer's features to RGB, float16 scores, and explained variance.

    Args:
        features: Array of shape [N, C].
        k: Largest number of components to store.
        seed: Seed for row subsampling in the PCA fit.

    Returns:
        Reduced with min(k, C, N - 1) stored components, and [N, 0] scores when N is 1.

    Raises:
        ValueError: When features are not 2D or hold NaN or inf.
    """
    x = _parse_features(features)
    z, mean, std = standardize(x)
    fit = replace(fit_pca(z, seed=seed), mean=mean, std=std)
    stored = min(k, fit.components.shape[0])
    scores = project(z, fit, stored)
    return Reduced(
        rgb=rgb_from_scores(scores),
        scores=to_float16(scores),
        explained=fit.explained,
        retained=float(np.sum(fit.explained[:stored], dtype=np.float64)),
        fit=fit,
    )


def channel_bytes(features: np.ndarray, cap_bytes: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize each channel to bytes, keeping the highest variance channels under a cap.

    Args:
        features: Array of shape [N, C].
        cap_bytes: Largest number of bytes, N times the kept channel count, to return.

    Returns:
        Tuple of uint8 data [N, C_kept] and the kept channel indices in ascending order.

    Raises:
        ValueError: When features are not 2D or cap_bytes is negative.

    Notes:
        Each channel is scaled from its 1st to 99th percentile onto 0..255.
    """
    if cap_bytes < 0:
        raise ValueError(f"cap_bytes must be at least 0, got {cap_bytes}")
    x = _as_2d(features)
    n_rows, n_channels = x.shape
    if n_rows * n_channels <= cap_bytes:
        kept = np.arange(n_channels)
    else:
        keep = cap_bytes // max(n_rows, 1)
        order = np.argsort(-x.var(axis=0), kind="stable")
        kept = np.sort(order[:keep])
    return _quantize_columns(x[:, kept]), kept
