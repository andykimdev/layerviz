"""
Test layer extraction: rank mismatch skips, sample index checks, and the share chain walk.

Purpose:

- Check that a layer with more spatial axes than the input is skipped with a reason.
- Check that a sample index outside the batch fails before any layer is extracted.
- Check that a plain [N, C] layer inherits positions through a container that shares.

Special Notes:

- Runs without spconv, using FakeSparseTensor and small modules defined here.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from layerviz.pipeline import CaptureOptions, run_pipeline
from tests.models import (
    FakeSparseTensor,
    PointsNet,
    SparseDownsample,
    SparseLinear,
    make_voxel_indices,
)


class Lift(nn.Module):
    """
    Add a depth axis of size 1, turning a [B, C, H, W] grid into [B, C, 1, H, W].

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input with a new axis at position 2."""
        return x.unsqueeze(2)


class SharingWrapper(nn.Module):
    """
    Return its child's sparse output unchanged, so its trace node shares the child's.

    """

    def __init__(self) -> None:
        """Create the downsampling child."""
        super().__init__()
        self.inner = SparseDownsample()

    def forward(self, sparse: FakeSparseTensor) -> FakeSparseTensor:
        """Return the child's output."""
        return self.inner(sparse)


class WrappedHead(nn.Module):
    """
    Feed the features of a sharing container's output to a plain Linear layer.

    """

    def __init__(self) -> None:
        """Create the container and the Linear head."""
        super().__init__()
        self.wrap = SharingWrapper()
        self.head = nn.Linear(4, 3)

    def forward(self, sparse: FakeSparseTensor) -> torch.Tensor:
        """Return the head output [N, 3] over the downsampled voxels."""
        return self.head(self.wrap(sparse).features)


def two_sample_sparse() -> FakeSparseTensor:
    """Return a sparse batch with 30 voxels in sample 0 and 20 voxels in sample 1."""
    torch.manual_seed(3)
    second = make_voxel_indices(20, 8)
    second[:, 0] = 1
    indices = torch.cat([make_voxel_indices(30, 8), second])
    return FakeSparseTensor(torch.randn(50, 4), indices, [8, 8, 8], 2)


def dense_pair() -> tuple[nn.Module, Any]:
    """Return a 1x1 convolution and a dense batch of 2 images."""
    torch.manual_seed(3)
    return nn.Conv2d(3, 4, 1).eval(), torch.randn(2, 3, 6, 6)


def sparse_pair() -> tuple[nn.Module, Any]:
    """Return a SparseLinear model and a sparse batch of 2 samples."""
    torch.manual_seed(3)
    return SparseLinear(4, 8).eval(), two_sample_sparse()


def points_pair() -> tuple[nn.Module, Any]:
    """Return PointsNet and a (coords, feats) pair of 2 samples."""
    torch.manual_seed(3)
    return PointsNet().eval(), (torch.randn(2, 20, 3), torch.randn(2, 20, 4))


def layers_by_name(result) -> dict:
    """Map each captured layer name to its CapturedLayer."""
    return {layer.node.name: layer for layer in result.layers}


def test_layer_with_more_axes_than_input_is_skipped_with_both_ranks():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Conv2d(3, 4, 1), Lift()).eval()

    result = run_pipeline(model, torch.randn(1, 3, 6, 6), CaptureOptions())

    assert list(layers_by_name(result)) == ["0"]
    assert list(result.skipped) == ["1"]
    assert "grid has 3 axes, input_grid has 2" in result.skipped["1"]


@pytest.mark.parametrize("build", [dense_pair, sparse_pair, points_pair])
@pytest.mark.parametrize("index", [2, -1])
def test_sample_index_outside_batch_raises_naming_index_and_size(
    build: Callable[[], tuple[nn.Module, Any]], index: int
):
    model, batch = build()

    with pytest.raises(ValueError, match=rf"sample_index {index} .*batch of 2 samples"):
        run_pipeline(model, batch, CaptureOptions(sample_index=index))


def test_linear_after_sharing_container_inherits_child_voxel_positions():
    torch.manual_seed(3)
    batch = FakeSparseTensor(torch.randn(30, 4), make_voxel_indices(30, 8), [8, 8, 8], 1)

    layers = layers_by_name(run_pipeline(WrappedHead().eval(), batch, CaptureOptions()))

    child, head = layers["wrap.inner"].data, layers["head"].data
    assert (child.kind, head.kind) == ("sparse3d", "sparse3d")
    assert len(child.positions) < len(batch.features)
    np.testing.assert_array_equal(head.positions, child.positions)
