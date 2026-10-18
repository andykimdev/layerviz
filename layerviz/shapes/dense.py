"""
Turn plain image2d, volume3d, tokens, and vector tensors into LayerData records.

Purpose:

- Provide one shape plugin per dense tensor rank, placing cells in input coordinates.
- Register the four plugins as the defaults by rank when this module is imported.

Special Notes:

- Each plugin accepts a torch.Tensor of its own rank only, so lookup order among them
  does not matter.
"""

from typing import Any, Optional

import numpy as np
import torch

from layerviz.shapes import base


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Return a tensor as a float32 NumPy array on the CPU."""
    return tensor.detach().cpu().to(torch.float32).numpy()


def _is_rank(output: Any, rank: int) -> bool:
    """Return whether the output is a torch.Tensor with exactly this many dimensions."""
    return isinstance(output, torch.Tensor) and output.dim() == rank


def _index_positions(count: int) -> np.ndarray:
    """Return the indices 0 to count - 1 as a float32 array of shape [count, 1]."""
    return np.arange(count, dtype=np.float32)[:, None]


def _grid_layer(kind: str, sample: torch.Tensor, input_grid: Optional[tuple]) -> base.LayerData:
    """
    Return one sample of a channels first grid as cells in input coordinates.

    Args:
        kind: Layer kind to record.
        sample: Tensor of shape [C, *grid] for one sample.
        input_grid: Spatial shape of the model input, or None to use the layer grid.

    Returns:
        LayerData with features [prod(grid), C] in row major order and scale equal to
        the stride per axis.
    """
    grid = tuple(sample.shape[1:])
    target = grid if input_grid is None else tuple(input_grid[-len(grid) :])
    features = _to_numpy(sample.reshape(sample.shape[0], -1).T)
    scale = tuple(float(t) / float(g) for t, g in zip(target, grid))
    return base.LayerData(
        kind=kind,
        positions=base.grid_positions(grid, target),
        features=features,
        grid=grid,
        scale=scale,
        labels=None,
    )


class _GridPlugin:
    """
    Handle a batched channels first grid tensor of one rank.

    """

    name = ""
    rank = 0

    def accepts(self, output: Any, ctx: base.ShapeContext) -> bool:
        """Return whether the output is a tensor of this plugin's rank."""
        return _is_rank(output, self.rank)

    def tensors(self, output: Any) -> list:
        """Return the output tensor in a list."""
        return [output]

    def extract(self, output: Any, ctx: base.ShapeContext) -> base.LayerData:
        """Return sample ctx.sample_index of the output as grid cells in input coordinates."""
        return _grid_layer(self.name, output[ctx.sample_index], ctx.input_grid)


class Image2dPlugin(_GridPlugin):
    """
    Handle a [B, C, H, W] tensor as an image grid.

    """

    name = "image2d"
    rank = 4


class Volume3dPlugin(_GridPlugin):
    """
    Handle a [B, C, D, H, W] tensor as a volume grid.

    """

    name = "volume3d"
    rank = 5


class TokensPlugin:
    """
    Handle a [B, N, C] tensor as a token sequence placed by token index.

    """

    name = "tokens"

    def accepts(self, output: Any, ctx: base.ShapeContext) -> bool:
        """Return whether the output is a 3D tensor."""
        return _is_rank(output, 3)

    def tensors(self, output: Any) -> list:
        """Return the output tensor in a list."""
        return [output]

    def extract(self, output: Any, ctx: base.ShapeContext) -> base.LayerData:
        """Return sample ctx.sample_index as [N, C] features at token positions [N, 1]."""
        sample = output[ctx.sample_index]
        return base.LayerData(
            kind=self.name,
            positions=_index_positions(sample.shape[0]),
            features=_to_numpy(sample),
            grid=None,
            scale=None,
            labels=None,
        )


class VectorPlugin:
    """
    Handle a [B, C] tensor as one point per batch sample.

    """

    name = "vector"

    def accepts(self, output: Any, ctx: base.ShapeContext) -> bool:
        """Return whether the output is a 2D tensor."""
        return _is_rank(output, 2)

    def tensors(self, output: Any) -> list:
        """Return the output tensor in a list."""
        return [output]

    def extract(self, output: Any, ctx: base.ShapeContext) -> base.LayerData:
        """Return the whole batch as [B, C] features at batch positions [B, 1]."""
        # One sample is a single point, so the batch is kept to give a cloud to compare.
        return base.LayerData(
            kind=self.name,
            positions=_index_positions(output.shape[0]),
            features=_to_numpy(output),
            grid=None,
            scale=None,
            labels=None,
        )


def register_defaults() -> None:
    """
    Register the four dense plugins in rank order, skipping any already registered.

    Notes:
        Plugins go to the front of the registry, in the order image2d, volume3d,
        tokens, vector. Calling this again changes nothing.
    """
    defaults = [Image2dPlugin(), Volume3dPlugin(), TokensPlugin(), VectorPlugin()]
    registered = {p.name for p in base.plugins()}
    # Inserting in reverse at the front leaves the defaults in rank order.
    for plugin in reversed(defaults):
        if plugin.name not in registered:
            base.register_shape(plugin)


register_defaults()
