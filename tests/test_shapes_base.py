"""
Test the shape plugin base: grid to input coordinates and the plugin registry.

Purpose:

- Check that downsampled grid cells map to input coordinates.
- Check registration order, acceptance lookup, and name overrides.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from layerviz.shapes import base


@dataclass
class FakePlugin:
    """
    Accept outputs according to a fixed answer and return a fixed kind.

    """

    name: str
    answer: bool = True

    def accepts(self, output, ctx):
        """Return the fixed acceptance answer."""
        return self.answer

    def tensors(self, output):
        """Return the output as the only tensor."""
        return [output]

    def extract(self, output, ctx):
        """Return an empty vector layer."""
        return base.LayerData(
            kind="vector",
            positions=np.zeros((0, 2), np.float32),
            features=np.zeros((0, 1), np.float32),
            grid=None,
            scale=None,
            labels=None,
        )


@pytest.fixture(autouse=True)
def empty_registry(monkeypatch):
    """Give each test its own empty registry."""
    monkeypatch.setattr(base, "REGISTRY", [])


def make_ctx():
    """Build a context for a 2D input of shape 4 by 4."""
    return base.ShapeContext(
        sample_index=0, input_grid=(4, 4), input_positions=None, input_rows=None
    )


def test_grid_positions_full_resolution_2d_returns_cell_indices():
    result = base.grid_positions((2, 3), (2, 3))
    expected = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=np.float32
    )
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


def test_grid_positions_half_resolution_2d_returns_cell_centers():
    result = base.grid_positions((2, 2), (4, 4))
    expected = np.array(
        [[0.5, 0.5], [0.5, 2.5], [2.5, 0.5], [2.5, 2.5]], dtype=np.float32
    )
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


def test_grid_positions_half_resolution_3d_returns_2i_plus_half():
    result = base.grid_positions((2, 1, 2), (4, 2, 4))
    expected = np.array(
        [[0.5, 0.5, 0.5], [0.5, 0.5, 2.5], [2.5, 0.5, 0.5], [2.5, 0.5, 2.5]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result, expected)


def test_grid_positions_full_resolution_3d_returns_cell_indices():
    result = base.grid_positions((1, 2, 2), (1, 2, 2))
    expected = np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=np.float32
    )
    np.testing.assert_array_equal(result, expected)


def test_grid_positions_rank_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="rank"):
        base.grid_positions((2, 2), (4, 4, 4))


def test_register_shape_puts_plugin_first_by_default():
    first = FakePlugin("first")
    second = FakePlugin("second")
    base.register_shape(first)
    base.register_shape(second)
    assert [p.name for p in base.plugins()] == ["second", "first"]


def test_register_shape_before_named_plugin_inserts_ahead_of_it():
    base.register_shape(FakePlugin("dense"))
    base.register_shape(FakePlugin("sparse"))
    base.register_shape(FakePlugin("custom"), before="dense")
    assert [p.name for p in base.plugins()] == ["sparse", "custom", "dense"]


def test_register_shape_before_unknown_name_raises_key_error():
    with pytest.raises(KeyError, match="missing"):
        base.register_shape(FakePlugin("custom"), before="missing")


def test_register_shape_duplicate_name_raises_value_error_and_keeps_registry():
    original = FakePlugin("dense")
    base.register_shape(original)
    with pytest.raises(ValueError, match="dense"):
        base.register_shape(FakePlugin("dense"))
    assert base.plugins() == [original]


def test_plugins_returns_copy_that_does_not_change_registry():
    base.register_shape(FakePlugin("dense"))
    base.plugins().clear()
    assert [p.name for p in base.plugins()] == ["dense"]


def test_plugin_for_returns_newly_registered_fake_plugin_first():
    base.register_shape(FakePlugin("dense"))
    fake = FakePlugin("fake")
    base.register_shape(fake)
    assert base.plugin_for(torch.zeros(1, 2), make_ctx()) is fake


def test_plugin_for_skips_plugins_that_do_not_accept():
    dense = FakePlugin("dense")
    base.register_shape(dense)
    base.register_shape(FakePlugin("picky", answer=False))
    assert base.plugin_for(torch.zeros(1, 2), make_ctx()) is dense


def test_plugin_for_returns_none_when_no_plugin_accepts():
    base.register_shape(FakePlugin("picky", answer=False))
    assert base.plugin_for(torch.zeros(1, 2), make_ctx()) is None


def test_plugin_for_override_wins_over_acceptance_order():
    dense = FakePlugin("dense", answer=False)
    base.register_shape(dense)
    base.register_shape(FakePlugin("fake"))
    assert base.plugin_for(torch.zeros(1, 2), make_ctx(), override="dense") is dense


def test_plugin_for_unknown_override_raises_key_error_with_name():
    base.register_shape(FakePlugin("dense"))
    with pytest.raises(KeyError, match="nosuch"):
        base.plugin_for(torch.zeros(1, 2), make_ctx(), override="nosuch")


@pytest.mark.parametrize(
    ("shape", "kind"),
    [
        ((1, 3, 4, 4), "image2d"),
        ((1, 3, 2, 4, 4), "volume3d"),
        ((1, 5, 8), "tokens"),
        ((1, 8), "vector"),
    ],
)
def test_default_kind_by_rank_maps_rank_to_kind(shape, kind):
    assert base.default_kind_by_rank(torch.zeros(shape)) == kind


def test_default_kind_by_rank_unsupported_rank_raises_value_error():
    with pytest.raises(ValueError, match="rank 1"):
        base.default_kind_by_rank(torch.zeros(4))
