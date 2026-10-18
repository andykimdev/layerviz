"""
Test the dense image2d, volume3d, tokens, and vector shape plugins and their registration.

Purpose:

- Check that each plugin turns a tiny model's last layer output into positions and [N, C].
- Check that downsampled grids land on input coordinates and defaults register once.
"""

import numpy as np
import pytest
import torch

from layerviz.shapes import base, dense
from tests.models import ALL_SHAPES

DEFAULT_NAMES = ["image2d", "volume3d", "tokens", "vector"]


def run_model(shape: str) -> tuple:
    """Return the batch and the last layer output of the named tiny model."""
    model, batch = ALL_SHAPES[shape]()
    with torch.no_grad():
        return batch, model(batch)


def context(input_grid, sample_index: int = 0) -> base.ShapeContext:
    """Return a shape context with the given input grid and sample."""
    return base.ShapeContext(
        sample_index=sample_index, input_grid=input_grid, input_positions=None, input_rows=None
    )


def expected_grid(grid: tuple, stride: float) -> np.ndarray:
    """Return the expected input coordinates of a grid, written out by axis."""
    axes = np.meshgrid(*[np.arange(n) for n in grid], indexing="ij")
    return np.stack([a.ravel() * stride + (stride - 1) / 2 for a in axes], axis=1)


def test_image2d_layer_maps_to_input_coordinates() -> None:
    batch, output = run_model("image2d")
    ctx = context(tuple(batch.shape[2:]))
    plugin = base.plugin_for(output, ctx)
    layer = plugin.extract(output, ctx)
    assert plugin.name == "image2d"
    assert layer.kind == "image2d"
    assert layer.grid == (8, 8)
    assert layer.scale == (2.0, 2.0)
    assert layer.labels is None
    assert layer.positions.shape == (64, 2)
    assert layer.positions.dtype == np.float32
    np.testing.assert_array_equal(layer.positions, expected_grid((8, 8), 2.0))
    assert layer.positions[9].tolist() == [2.5, 2.5]
    want = output[0].permute(1, 2, 0).reshape(64, 6).numpy()
    assert layer.features.shape == (64, 6)
    assert layer.features.dtype == np.float32
    np.testing.assert_array_equal(layer.features, want)


def test_volume3d_layer_maps_to_input_coordinates() -> None:
    batch, output = run_model("volume3d")
    ctx = context(tuple(batch.shape[2:]))
    plugin = base.plugin_for(output, ctx)
    layer = plugin.extract(output, ctx)
    assert plugin.name == "volume3d"
    assert layer.kind == "volume3d"
    assert layer.grid == (4, 4, 4)
    assert layer.scale == (2.0, 2.0, 2.0)
    assert layer.labels is None
    assert layer.positions.shape == (64, 3)
    np.testing.assert_array_equal(layer.positions, expected_grid((4, 4, 4), 2.0))
    assert layer.positions[21].tolist() == [2.5, 2.5, 2.5]
    want = output[0].permute(1, 2, 3, 0).reshape(64, 6).numpy()
    assert layer.features.shape == (64, 6)
    np.testing.assert_array_equal(layer.features, want)


def test_tokens_layer_uses_token_index_as_position() -> None:
    _, output = run_model("tokens")
    ctx = context(None)
    plugin = base.plugin_for(output, ctx)
    layer = plugin.extract(output, ctx)
    assert plugin.name == "tokens"
    assert layer.kind == "tokens"
    assert layer.grid is None
    assert layer.scale is None
    assert layer.labels is None
    assert layer.positions.dtype == np.float32
    np.testing.assert_array_equal(layer.positions, np.arange(12, dtype=np.float32)[:, None])
    assert layer.features.shape == (12, 16)
    np.testing.assert_array_equal(layer.features, output[0].numpy())


def test_tokens_layer_reads_the_requested_sample() -> None:
    output = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    layer = dense.TokensPlugin().extract(output, context(None, sample_index=1))
    np.testing.assert_array_equal(layer.features, output[1].numpy())


def test_vector_layer_keeps_the_whole_batch() -> None:
    _, output = run_model("vector")
    ctx = context(None, sample_index=2)
    plugin = base.plugin_for(output, ctx)
    layer = plugin.extract(output, ctx)
    assert plugin.name == "vector"
    assert layer.kind == "vector"
    assert layer.grid is None
    assert layer.scale is None
    assert layer.labels is None
    assert layer.positions.dtype == np.float32
    np.testing.assert_array_equal(layer.positions, np.arange(4, dtype=np.float32)[:, None])
    assert layer.features.shape == (4, 5)
    np.testing.assert_array_equal(layer.features, output.numpy())


def test_image2d_reads_the_requested_sample() -> None:
    output = torch.randn(2, 3, 4, 4)
    layer = dense.Image2dPlugin().extract(output, context((8, 8), sample_index=1))
    want = output[1].permute(1, 2, 0).reshape(16, 3).numpy()
    np.testing.assert_array_equal(layer.features, want)


def test_missing_input_grid_gives_stride_one() -> None:
    output = torch.randn(1, 3, 2, 3)
    layer = dense.Image2dPlugin().extract(output, context(None))
    assert layer.grid == (2, 3)
    assert layer.scale == (1.0, 1.0)
    np.testing.assert_array_equal(layer.positions, expected_grid((2, 3), 1.0))


def test_image2d_uses_the_last_two_input_axes() -> None:
    output = torch.randn(1, 3, 2, 2)
    layer = dense.Image2dPlugin().extract(output, context((3, 8, 4)))
    assert layer.scale == (4.0, 2.0)
    assert layer.positions[:, 0].tolist() == [1.5, 1.5, 5.5, 5.5]
    assert layer.positions[:, 1].tolist() == [0.5, 2.5, 0.5, 2.5]


@pytest.mark.parametrize(
    "plugin, rank",
    [
        (dense.Image2dPlugin(), 4),
        (dense.Volume3dPlugin(), 5),
        (dense.TokensPlugin(), 3),
        (dense.VectorPlugin(), 2),
    ],
)
def test_each_plugin_accepts_only_its_rank(plugin, rank) -> None:
    ctx = context(None)
    for other in range(1, 7):
        tensor = torch.zeros((1,) * other)
        assert plugin.accepts(tensor, ctx) is (other == rank)
    assert plugin.accepts(np.zeros((1,) * rank), ctx) is False
    tensor = torch.zeros((1,) * rank)
    assert plugin.tensors(tensor) == [tensor]


def test_import_registers_defaults_once_in_rank_order() -> None:
    names = [p.name for p in base.plugins()]
    assert [n for n in names if n in DEFAULT_NAMES] == DEFAULT_NAMES
    dense.register_defaults()
    assert [p.name for p in base.plugins()] == names


def test_register_defaults_fills_an_empty_registry(monkeypatch) -> None:
    monkeypatch.setattr(base, "REGISTRY", [])
    dense.register_defaults()
    dense.register_defaults()
    assert [p.name for p in base.plugins()] == DEFAULT_NAMES
