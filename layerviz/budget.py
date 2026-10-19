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
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from layerviz.reduce import SCORE_COUNT
from layerviz.shapes.base import LayerData

# One int prices every layer alike. A mapping prices each layer by name.
PositionBytes = int | Mapping[str, int]

MB = 2**20


def position_bytes(dim: int, joint: bool = False) -> int:
    """
    Count the bytes one kept position costs in a layer file.

    Args:
        dim: Coordinates per position, each a 4 byte float32.
        joint: Whether the position also stores 3 uint8 joint rgb values, as in compare mode.

    Returns:
        The bytes of the coordinates, 3 uint8 rgb values, SCORE_COUNT float16 scores, and
        the joint rgb when joint is true.
    """
    return dim * 4 + 3 + SCORE_COUNT * 2 + (3 if joint else 0)


BYTES_PER_POSITION = position_bytes(3)
COMPARE_BYTES_PER_POSITION = position_bytes(3, joint=True)


def size_of(bytes_per_position: PositionBytes, name: str) -> int:
    """
    Look up the bytes one kept position of the named layer costs.

    Args:
        bytes_per_position: One size for every layer, or a size per layer name.
        name: Layer name to price.

    Returns:
        The int itself, or the mapping's value for name.

    Raises:
        KeyError: When the mapping has no size for name.
    """
    if isinstance(bytes_per_position, int):
        return bytes_per_position
    if name not in bytes_per_position:
        raise KeyError(f"bytes_per_position mapping has no size for layer {name!r}")
    return bytes_per_position[name]


def positions_in(mb: float, bytes_per_position: PositionBytes = BYTES_PER_POSITION) -> int:
    """
    Convert a size in megabytes to the number of positions that fit in it.

    Args:
        mb: Size in megabytes of 2**20 bytes.
        bytes_per_position: Bytes one kept position costs, larger in compare mode. A
            mapping of layer name to size counts positions of its largest size.

    Returns:
        The whole number of positions of bytes_per_position bytes that fit.

    Raises:
        ValueError: When bytes_per_position is an empty mapping.
    """
    if isinstance(bytes_per_position, int):
        return int(mb * MB) // bytes_per_position
    if not bytes_per_position:
        raise ValueError("bytes_per_position is an empty mapping, so it prices no layer")
    return int(mb * MB) // max(bytes_per_position.values())


def water_fill(demand: dict[str, int], budget: int) -> dict[str, int]:
    """
    Share a budget so layers under their even share keep all they ask for.

    Args:
        demand: Amount each layer asks for, in any unit such as positions or bytes,
            already capped per layer.
        budget: Total amount to share, in the same unit as demand.

    Returns:
        Amount per layer in that unit. Layers under the final share keep their demand
        and the rest get the share.
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
    bytes_per_position: PositionBytes = BYTES_PER_POSITION,
    overhead: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Plan how many positions each layer keeps under the per layer and total caps.

    Args:
        counts: Number of positions in each layer, by layer name.
        layer_mb: Largest size of one layer in megabytes.
        total_mb: Largest size of all layers together in megabytes.
        bytes_per_position: Bytes one kept position costs, COMPARE_BYTES_PER_POSITION
            in compare mode. A mapping gives each layer name its own size.
        overhead: Bytes each named layer's file adds beyond its positions, reserved from
            its layer cap. A missing name reserves nothing.

    Returns:
        Positions kept per layer, each at most the layer cap and the layer's count, and at
        least 1 for a non empty layer. Empty counts give an empty dict.

    Raises:
        KeyError: When a bytes_per_position mapping has no size for a non empty layer.

    Notes:
        Each non empty layer first keeps 1 position, then the bytes left under total_mb
        are water filled, so a layer with costlier positions keeps fewer. When total_mb
        is below 1 position per non empty layer, the total can exceed total_mb, and a
        UserWarning names total_mb and the minimum needed in positions and MB.
    """
    # Empty layers keep nothing and cost nothing, so only non empty layers are priced.
    filled = [name for name, count in counts.items() if count > 0]
    size = {name: size_of(bytes_per_position, name) for name in filled}
    reserve = overhead or {}
    demand = {
        name: min(counts[name], layer_cap(layer_mb, reserve.get(name, 0), size[name]))
        for name in filled
    }
    spare = int(total_mb * MB) - sum(size[name] for name in filled)
    if spare < 0:
        warn_tiny_budget(total_mb, filled, bytes_per_position)
    wants = {name: (demand[name] - 1) * size[name] for name in filled}
    extra = water_fill(wants, max(0, spare))
    keep = {name: 0 for name in counts}
    return keep | {name: 1 + extra[name] // size[name] for name in filled}


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
    total_mb: float,
    filled: Sequence[str],
    bytes_per_position: PositionBytes = BYTES_PER_POSITION,
) -> None:
    """
    Warn that the total budget is below 1 position per non empty layer.

    Args:
        total_mb: Requested total size in megabytes.
        filled: Names of the non empty layers, so the minimum positions needed.
        bytes_per_position: Bytes one kept position costs, one int or a size per layer.

    Notes:
        With sizes per layer, the positions that fit are counted at the largest size.
    """
    sizes = {name: size_of(bytes_per_position, name) for name in filled}
    budget = positions_in(total_mb, sizes)
    needed = len(filled)
    needed_mb = sum(sizes.values()) / MB
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
