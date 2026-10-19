"""
Place sparse voxel and point cloud layers in input coordinates.

Purpose:

- Provide the sparse3d plugin for any object shaped like a spconv SparseConvTensor.
- Provide the points plugin for [B, N, C] tensors aligned with the input points.
- Let a plain [N, C] tensor inherit the positions of a sparse or points ancestor.

Special Notes:

- Duck typed on features, indices, spatial_shape, and batch_size, so spconv is never imported.
- Imports dense first, so points always lands ahead of the tokens plugin.
"""

from typing import Any

import numpy as np
import torch

from layerviz.shapes import base
# Imported for its side effect, so tokens is registered before points anchors on it.
from layerviz.shapes import dense  # noqa: F401
from layerviz.shapes.base import LayerData, ShapeContext, to_numpy

_SPARSE_FIELDS = ("features", "indices", "spatial_shape", "batch_size")


def _stride(grid: tuple, input_grid: tuple | None) -> np.ndarray:
    """
    Return the input cells per layer cell on each axis.

    Args:
        grid: Spatial shape of the layer.
        input_grid: Spatial shape of the model input, or None for stride 1.

    Returns:
        Float64 array with one stride per axis.

    Raises:
        ValueError: When grid and input_grid have different numbers of axes.
    """
    if input_grid is None:
        return np.ones(len(grid))
    if len(grid) != len(input_grid):
        raise ValueError(f"grid {grid} and input_grid {input_grid} differ in rank")
    return np.asarray(input_grid, dtype=np.float64) / np.asarray(grid, dtype=np.float64)


class Sparse3dPlugin:
    """
    Turn a sparse voxel tensor into voxel centers in input coordinates.

    """

    name = "sparse3d"

    def accepts(self, output: Any, ctx: ShapeContext) -> bool:
        """
        Return whether the output has the four sparse tensor attributes.

        Args:
            output: Module output of any type.
            ctx: Context of the layer, unused because the attributes decide alone.

        Returns:
            True when output has features, indices, spatial_shape, and batch_size.
        """
        return all(hasattr(output, field) for field in _SPARSE_FIELDS)

    def tensors(self, output: Any) -> list:
        """
        Return the voxel feature tensor.

        Args:
            output: Sparse tensor this plugin accepted.

        Returns:
            A one item list holding the features tensor [N, C].
        """
        return [output.features]

    def extract(self, output: Any, ctx: ShapeContext) -> LayerData:
        """
        Return the voxels of one sample as positions and features.

        Args:
            output: Sparse tensor with indices [N, 4] in columns (batch, z, y, x).
            ctx: Context giving the sample index and the input voxel grid.

        Returns:
            LayerData of kind sparse3d with positions [M, 3] in (z, y, x) input
            coordinates, grid equal to spatial_shape, and scale equal to the stride.
        """
        grid = tuple(int(size) for size in output.spatial_shape)
        stride = _stride(grid, ctx.input_grid)
        rows = output.indices[:, 0] == ctx.sample_index
        voxels = output.indices[rows, 1:].cpu().numpy()
        positions = (voxels * stride + (stride - 1) / 2).astype(np.float32)
        features = to_numpy(output.features[rows])
        scale = tuple(float(s) for s in stride)
        return LayerData("sparse3d", positions, features, grid, scale, None)


class PointsPlugin:
    """
    Place per point features of a [B, N, C] tensor on the input point coordinates.

    """

    name = "points"

    def accepts(self, output: Any, ctx: ShapeContext) -> bool:
        """
        Return whether the output is [B, N, C] with N equal to the input point count.

        Args:
            output: Module output of any type.
            ctx: Context giving the input positions [N, D], or None without points.

        Returns:
            True when output is a 3D tensor whose second axis matches the point count.
        """
        return (
            isinstance(output, torch.Tensor)
            and output.dim() == 3
            and ctx.input_positions is not None
            and output.shape[1] == len(ctx.input_positions)
        )

    def tensors(self, output: Any) -> list:
        """
        Return the output tensor itself.

        Args:
            output: Tensor [B, N, C] this plugin accepted.

        Returns:
            A one item list holding output.
        """
        return [output]

    def extract(self, output: Any, ctx: ShapeContext) -> LayerData:
        """
        Return the points of one sample at the input coordinates.

        Args:
            output: Float tensor [B, N, C].
            ctx: Context giving the sample index and the input positions [N, D].

        Returns:
            LayerData of kind points with the input positions and features [N, C].
        """
        positions = np.asarray(ctx.input_positions, dtype=np.float32)
        features = to_numpy(output[ctx.sample_index])
        return LayerData("points", positions, features, None, None, None)


def inherit_positions(features_2d: Any, ancestor: LayerData) -> LayerData | None:
    """
    Give a plain [N, C] tensor the layout of an ancestor with N rows.

    Args:
        features_2d: Module output, used only when it is a 2D tensor.
        ancestor: LayerData of the sparse or points layer it descends from.

    Returns:
        LayerData with the ancestor's kind, positions, grid, and scale and the new
        features, or None when the output is not 2D or its row count differs.
    """
    if not isinstance(features_2d, torch.Tensor) or features_2d.dim() != 2:
        return None
    if features_2d.shape[0] != len(ancestor.positions):
        return None
    features = to_numpy(features_2d)
    return LayerData(
        ancestor.kind, ancestor.positions, features, ancestor.grid, ancestor.scale, None
    )


def _registered(name: str) -> bool:
    """Return whether a plugin with the given name is in the registry."""
    return any(plugin.name == name for plugin in base.REGISTRY)


def register_defaults() -> None:
    """
    Register points directly before tokens and sparse3d at the end, once each.

    Notes:
        Points goes to the end when tokens is missing, as in a registry that a test
        has replaced. Nothing goes to the front, so a plugin a user registered
        earlier keeps priority.
    """
    if not _registered(PointsPlugin.name):
        if _registered("tokens"):
            base.register_shape(PointsPlugin(), before="tokens")
        else:
            base.append_shape(PointsPlugin())
    if not _registered(Sparse3dPlugin.name):
        base.append_shape(Sparse3dPlugin())


register_defaults()
