"""
Decide how many positions each layer keeps under the per layer and total size caps.

Purpose:

- Split the total budget across layers by water filling, capped per layer.
- Share the bytes the layer files leave under the total across the channel files.
- Subsample a layer so layers on the same grid keep the same positions.

Special Notes:

- Pure NumPy with no file IO. The seed comes from a stable digest, not Python hash(),
  so indices are the same across processes.
"""

import hashlib
import warnings
from dataclasses import replace
from typing import Optional

import numpy as np

from layerviz.shapes.base import LayerData

# Each kept position stores 3 float32 coordinates (12 bytes), 3 uint8 rgb values
# (3 bytes), and 32 float16 scores (64 bytes).
BYTES_PER_POSITION = 3 * 4 + 3 * 1 + 32 * 2
# A comparison pane also stores 3 uint8 joint rgb values per position.
COMPARE_BYTES_PER_POSITION = BYTES_PER_POSITION + 3

MB = 2**20


def positions_in(mb: float, bytes_per_position: int = BYTES_PER_POSITION) -> int:
    """
    Convert a size in megabytes to the number of positions that fit in it.

    Args:
        mb: Size in megabytes of 2**20 bytes.
        bytes_per_position: Bytes one kept position costs, larger in compare mode.

    Returns:
        The whole number of positions of bytes_per_position bytes that fit.
    """
    return int(mb * MB) // bytes_per_position


def water_fill(demand: dict[str, int], budget: int) -> dict[str, int]:
    """
    Share a budget so layers under their even share keep all they ask for.

    Args:
        demand: Positions each layer asks for, already capped per layer.
        budget: Total positions to share.

    Returns:
        Positions per layer. Layers under the final share keep their demand and the
        rest get the share.
    """
    keep: dict[str, int] = {}
    open_names = sorted(demand)
    while open_names:
        share = budget // len(open_names)
        satisfied = [name for name in open_names if demand[name] <= share]
        if not satisfied:
            return keep | {name: share for name in open_names}
        for name in satisfied:
            keep[name] = demand[name]
            budget -= demand[name]
        open_names = [name for name in open_names if name not in keep]
    return keep


def layer_cap(layer_mb: float, overhead: int, bytes_per_position: int) -> int:
    """
    Return the positions one layer file holds within layer_mb after its overhead.

    Args:
        layer_mb: Largest size of one layer file in megabytes.
        overhead: Bytes the file adds beyond its positions, such as its header.
        bytes_per_position: Bytes one kept position costs.

    Returns:
        The whole number of positions that fit, at least 1 so no layer is emptied.
    """
    return max(1, (int(layer_mb * MB) - overhead) // bytes_per_position)


def plan_keep(
    counts: dict[str, int],
    layer_mb: float = 16,
    total_mb: float = 1024,
    bytes_per_position: int = BYTES_PER_POSITION,
    overhead: Optional[dict[str, int]] = None,
) -> dict[str, int]:
    """
    Plan how many positions each layer keeps under the per layer and total caps.

    Args:
        counts: Number of positions in each layer, by layer name.
        layer_mb: Largest size of one layer in megabytes.
        total_mb: Largest size of all layers together in megabytes.
        bytes_per_position: Bytes one kept position costs, COMPARE_BYTES_PER_POSITION
            in compare mode.
        overhead: Bytes each named layer's file adds beyond its positions, reserved from
            its layer cap. A missing name reserves nothing.

    Returns:
        Positions kept per layer, each at most the layer cap and the layer's count, and at
        least 1 for a non empty layer. Empty counts give an empty dict.

    Notes:
        When total_mb fits fewer positions than there are non empty layers, each non
        empty layer keeps 1 position, so the total can exceed total_mb, and a
        UserWarning names total_mb and the minimum needed in positions and MB.
    """
    if not counts:
        return {}
    reserve = overhead or {}
    demand = {
        name: min(count, layer_cap(layer_mb, reserve.get(name, 0), bytes_per_position))
        for name, count in counts.items()
    }
    budget = positions_in(total_mb, bytes_per_position)
    needed = sum(1 for want in demand.values() if want > 0)
    if budget < needed:
        warn_tiny_budget(total_mb, budget, needed, bytes_per_position)
        keep = {name: min(want, 1) for name, want in demand.items()}
    else:
        keep = water_fill(demand, budget)
    return {name: keep[name] for name in counts}


def plan_channel_caps(
    need: dict[str, int], layer_mb: float, total_mb: float, used: int
) -> dict[str, int]:
    """
    Share the bytes left under total_mb across the channel files, each capped at layer_mb.

    Args:
        need: Bytes each layer's full channel file takes, header included, by layer name.
        layer_mb: Largest size of one file in megabytes.
        total_mb: Largest size of all layer and channel files together in megabytes.
        used: Bytes the layer files already take.

    Returns:
        Largest channel file size per layer in bytes. A file under its even share of what is
        left keeps all it needs, and the rest share the remainder. Zero when the layer
        files leave nothing.
    """
    cap = int(layer_mb * MB)
    demand = {name: min(bytes_needed, cap) for name, bytes_needed in need.items()}
    spare = max(0, int(total_mb * MB) - used)
    caps = water_fill(demand, spare)
    return {name: caps[name] for name in need}


def warn_tiny_budget(
    total_mb: float, budget: int, needed: int, bytes_per_position: int = BYTES_PER_POSITION
) -> None:
    """
    Warn that the total budget is below 1 position per non empty layer.

    Args:
        total_mb: Requested total size in megabytes.
        budget: Positions that fit in total_mb.
        needed: Non empty layers, so the minimum positions needed.
        bytes_per_position: Bytes one kept position costs.
    """
    needed_mb = needed * bytes_per_position / MB
    warnings.warn(
        f"total_mb={total_mb:g} MB fits {budget} positions, fewer than the {needed} "
        f"positions ({needed_mb:g} MB) needed for 1 position per layer. Keeping 1 "
        "position per layer.",
        stacklevel=3,
    )


def grid_key(layer: LayerData) -> tuple:
    """
    Build the key that layers on an identical grid share.

    Args:
        layer: The layer to key.

    Returns:
        The tuple (kind, grid, N) where N is the number of positions. The grid is
        converted to Python ints, so NumPy and Python grids give the same key.
    """
    grid = None if layer.grid is None else tuple(int(x) for x in layer.grid)
    return (layer.kind, grid, int(layer.positions.shape[0]))


def stable_seed(seed: int, key: tuple) -> int:
    """
    Derive a random seed from a base seed and a grid key that is stable across runs.

    Args:
        seed: Base seed.
        key: Grid key from grid_key.

    Returns:
        A 64 bit integer seed.
    """
    digest = hashlib.sha256(repr((seed, key)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def subsample(layer: LayerData, keep: int, seed: int = 0) -> tuple[LayerData, np.ndarray]:
    """
    Keep a random subset of a layer's positions, the same subset for the same grid.

    Args:
        layer: The layer to subsample.
        keep: Number of positions to keep.
        seed: Base seed mixed with the grid key.

    Returns:
        The subsampled layer and the sorted kept row indices. When keep is at least N
        the layer comes back unchanged with indices 0 to N - 1.

    Notes:
        Labels are subset only when there is one label per position.
    """
    n = int(layer.positions.shape[0])
    if keep >= n:
        return layer, np.arange(n)
    rng = np.random.default_rng(stable_seed(seed, grid_key(layer)))
    idx = np.sort(rng.choice(n, size=keep, replace=False))
    labels = layer.labels
    if labels is not None and len(labels) == n:
        labels = [labels[i] for i in idx]
    kept = replace(
        layer, positions=layer.positions[idx], features=layer.features[idx], labels=labels
    )
    return kept, idx
