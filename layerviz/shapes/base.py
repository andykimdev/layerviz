"""
Define the layer record, the shape plugin protocol, and the plugin registry.

Purpose:

- Describe one captured layer as a LayerData record in input coordinates.
- Keep the ordered plugin registry and pick the plugin for a module output.
- Map the cells of a downsampled grid to input coordinates.
- Copy tensors to float32 NumPy arrays for every plugin.

Special Notes:

- Imports numpy and torch only, never a plugin module, so plugins can import this file.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

_KIND_BY_RANK = {4: "image2d", 5: "volume3d", 3: "tokens", 2: "vector"}


@dataclass
class LayerData:
    """
    Hold one layer as positions in input coordinates and per position features.

    """

    kind: str
    positions: np.ndarray
    features: np.ndarray
    grid: tuple | None
    scale: tuple | None
    labels: list | None


@dataclass
class ShapeContext:
    """
    Describe the model input that a plugin needs to place a layer.

    """

    sample_index: int
    input_grid: tuple | None
    input_positions: np.ndarray | None
    input_rows: int | None


class ShapePlugin(Protocol):
    """
    Recognize one kind of module output and turn it into LayerData.

    """

    name: str

    def accepts(self, output: Any, ctx: ShapeContext) -> bool:
        """
        Return whether this plugin handles the module output.

        Args:
            output: The module output.
            ctx: The shape context of the current sample.

        Returns:
            True when this plugin can turn the output into LayerData.
        """
        ...

    def tensors(self, output: Any) -> list:
        """
        Return the tensors inside the module output.

        Args:
            output: The module output.

        Returns:
            The tensors the output holds, in a list.
        """
        ...

    def extract(self, output: Any, ctx: ShapeContext) -> LayerData:
        """
        Return the module output as a LayerData record.

        Args:
            output: The module output.
            ctx: The shape context of the current sample.

        Returns:
            The layer in input coordinates.
        """
        ...


# Public so tests can swap in an empty list with monkeypatch.
REGISTRY: list[ShapePlugin] = []


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Return a float32 NumPy copy of a tensor, detached and on the CPU.

    Args:
        tensor: Any torch tensor, on any device and of any float dtype.

    Returns:
        A float32 NumPy array with the same shape and values that shares no memory
        with the input.

    Notes:
        The copy guards against later in-place edits of the module output, such as
        an inplace ReLU that runs after the capture hook fires.
    """
    return tensor.detach().to(device="cpu", dtype=torch.float32, copy=True).numpy()


def grid_positions(grid: tuple, input_grid: tuple) -> np.ndarray:
    """
    Return the input coordinates of every cell of a grid, in row major order.

    Args:
        grid: Spatial shape of the layer, one size per axis.
        input_grid: Spatial shape of the model input, with the same number of axes.

    Returns:
        Float32 array of shape [prod(grid), len(grid)]. With stride s equal to
        input_grid / grid on an axis, index i maps to i * s + (s - 1) / 2.

    Raises:
        ValueError: When grid and input_grid have different numbers of axes.
    """
    if len(grid) != len(input_grid):
        raise ValueError(
            f"grid has {len(grid)} axes, input_grid has {len(input_grid)}, "
            f"so their ranks differ: grid {grid}, input_grid {input_grid}"
        )
    stride = np.asarray(input_grid, dtype=np.float64) / np.asarray(grid, dtype=np.float64)
    indices = np.indices(grid).reshape(len(grid), -1).T
    return (indices * stride + (stride - 1) / 2).astype(np.float32)


def _index_of(name: str) -> int:
    """
    Return the registry position of the plugin with the given name.

    Raises:
        KeyError: When no registered plugin has that name.
    """
    for index, plugin in enumerate(REGISTRY):
        if plugin.name == name:
            return index
    raise KeyError(f"no shape plugin named {name!r}")


def _require_new_name(plugin: ShapePlugin) -> None:
    """
    Check that no registered plugin already uses this plugin's name.

    Raises:
        ValueError: When a plugin with the same name is already registered.
    """
    if any(p.name == plugin.name for p in REGISTRY):
        raise ValueError(f"a shape plugin named {plugin.name!r} is already registered")


def register_shape(plugin: ShapePlugin, before: str | None = None) -> None:
    """
    Add a plugin to the registry, at the front or before a named plugin.

    Args:
        plugin: The plugin to register.
        before: Name of a registered plugin to insert ahead of, or None for the front.

    Raises:
        ValueError: When a plugin with the same name is already registered.
        KeyError: When before names no registered plugin.
    """
    _require_new_name(plugin)
    position = 0 if before is None else _index_of(before)
    REGISTRY.insert(position, plugin)


def append_shape(plugin: ShapePlugin) -> None:
    """
    Add a plugin to the end of the registry, behind every plugin already registered.

    Args:
        plugin: The plugin to register.

    Raises:
        ValueError: When a plugin with the same name is already registered.

    Notes:
        Built-in defaults use this, so a plugin a user registers earlier keeps priority.
    """
    _require_new_name(plugin)
    REGISTRY.append(plugin)


def plugins() -> list[ShapePlugin]:
    """
    Return a copy of the registered plugins in lookup order.

    Returns:
        A new list, so changing it leaves the registry alone.
    """
    return list(REGISTRY)


def plugin_for(output: Any, ctx: ShapeContext, override: str | None = None) -> ShapePlugin | None:
    """
    Return the plugin that should handle a module output.

    Args:
        output: The module output.
        ctx: The shape context of the current sample.
        override: Name of a plugin to use regardless of acceptance, or None.

    Returns:
        The plugin named override, else the first plugin that accepts, else None.

    Raises:
        KeyError: When override names no registered plugin.
    """
    if override is not None:
        return REGISTRY[_index_of(override)]
    return next((p for p in REGISTRY if p.accepts(output, ctx)), None)


def default_kind_by_rank(tensor: torch.Tensor) -> str:
    """
    Return the layer kind a dense tensor of this rank usually holds.

    Args:
        tensor: A batched tensor of rank 2 to 5.

    Returns:
        image2d for 4D, volume3d for 5D, tokens for 3D, and vector for 2D.

    Raises:
        ValueError: When the tensor has any other rank.
    """
    rank = tensor.dim()
    if rank not in _KIND_BY_RANK:
        raise ValueError(f"no default layer kind for a tensor of rank {rank}")
    return _KIND_BY_RANK[rank]
