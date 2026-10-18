"""
Test the sparse3d and points plugins and the inherited positions of plain [N, C] tensors.

Purpose:

- Check sparse voxel positions in input coordinates at full and half resolution.
- Check point features sit on the input coordinates.
- Check a plain [N, C] tensor inherits an ancestor's positions only when N matches.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from layerviz.shapes import base, sparse
from tests.models import FakeSparseTensor, points_model, sparse_model

SPARSE_GRID = (8, 8, 8)
REPO_ROOT = Path(__file__).resolve().parents[1]
DENSE_NAMES = ("image2d", "volume3d", "tokens", "vector")


def fresh_import_names(code):
    """
    Import only layerviz.shapes.sparse in a fresh interpreter and evaluate an expression.

    Args:
        code: Python expression evaluated after the import, printed with repr.

    Returns:
        The evaluated value, read back with eval.
    """
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import layerviz.shapes.sparse; "
        "from layerviz.shapes import base; "
        f"print(repr({code}))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(REPO_ROOT)],
        capture_output=True,
        text=True,
        check=True,
    )
    return eval(result.stdout)


def sparse_outputs():
    """Return the batch and the first two layer outputs of the sparse test model."""
    model, batch = sparse_model()
    with torch.no_grad():
        full = model[0](batch)
        half = model[1](full)
    return batch, full, half


def sparse_ctx(sample_index=0):
    """Return a context for the 8 by 8 by 8 sparse input."""
    return base.ShapeContext(sample_index, SPARSE_GRID, None, None)


def points_ctx(coords, sample_index=0):
    """Return a context whose input positions are the given point coordinates."""
    positions = coords[sample_index].numpy()
    return base.ShapeContext(sample_index, None, positions, len(positions))


def test_importing_sparse_does_not_import_spconv():
    loaded = fresh_import_names("sorted(m for m in sys.modules if m.startswith('spconv'))")
    assert loaded == []


def test_importing_sparse_alone_orders_registry_for_sparse_and_points():
    names = fresh_import_names("[p.name for p in base.plugins()]")
    assert sorted(names) == sorted(DENSE_NAMES + ("sparse3d", "points"))
    assert names.index("points") < names.index("tokens")
    assert names.index("sparse3d") < min(names.index(name) for name in DENSE_NAMES)


def test_sparse3d_accepts_duck_typed_sparse_object_only():
    _, full, _ = sparse_outputs()
    plugin = sparse.Sparse3dPlugin()
    assert plugin.accepts(full, sparse_ctx())
    assert not plugin.accepts(full.features, sparse_ctx())


def test_sparse3d_tensors_returns_features():
    _, full, _ = sparse_outputs()
    (tensor,) = sparse.Sparse3dPlugin().tensors(full)
    assert tensor is full.features


def test_sparse3d_full_resolution_positions_equal_voxel_indices():
    _, full, _ = sparse_outputs()
    data = sparse.Sparse3dPlugin().extract(full, sparse_ctx())
    assert data.kind == "sparse3d"
    np.testing.assert_array_equal(data.positions, full.indices[:, 1:].numpy())
    assert data.positions.dtype == np.float32
    np.testing.assert_array_equal(data.features, full.features.numpy())
    assert data.grid == (8, 8, 8)
    assert data.scale == (1.0, 1.0, 1.0)
    assert data.labels is None


def test_sparse3d_half_resolution_positions_are_stride_two_centers():
    _, _, half = sparse_outputs()
    data = sparse.Sparse3dPlugin().extract(half, sparse_ctx())
    expected = half.indices[:, 1:].numpy() * 2 + 0.5
    np.testing.assert_array_equal(data.positions, expected)
    np.testing.assert_array_equal(data.features, half.features.numpy())
    assert data.grid == (4, 4, 4)
    assert data.scale == (2.0, 2.0, 2.0)


def test_sparse3d_keeps_only_rows_of_the_sample():
    indices = torch.tensor([[0, 1, 2, 3], [1, 4, 5, 6], [1, 7, 0, 1]], dtype=torch.int32)
    features = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    output = FakeSparseTensor(features, indices, [8, 8, 8], 2)
    data = sparse.Sparse3dPlugin().extract(output, sparse_ctx(sample_index=1))
    np.testing.assert_array_equal(data.positions, [[4, 5, 6], [7, 0, 1]])
    np.testing.assert_array_equal(data.features, [[2, 3], [4, 5]])


def test_sparse3d_without_input_grid_uses_stride_one():
    _, _, half = sparse_outputs()
    ctx = base.ShapeContext(0, None, None, None)
    data = sparse.Sparse3dPlugin().extract(half, ctx)
    np.testing.assert_array_equal(data.positions, half.indices[:, 1:].numpy())
    assert data.scale == (1.0, 1.0, 1.0)


def test_sparse3d_rank_mismatch_between_grid_and_input_grid_raises():
    _, full, _ = sparse_outputs()
    ctx = base.ShapeContext(0, (8, 8), None, None)
    with pytest.raises(ValueError, match="differ in rank"):
        sparse.Sparse3dPlugin().extract(full, ctx)


def test_points_accepts_3d_tensor_with_matching_rows_only():
    model, (coords, feats) = points_model()
    output = model((coords, feats))
    plugin = sparse.PointsPlugin()
    assert plugin.accepts(output, points_ctx(coords))
    assert not plugin.accepts(output, base.ShapeContext(0, None, None, None))
    assert not plugin.accepts(output[:, :10], points_ctx(coords))
    assert not plugin.accepts(output[0], points_ctx(coords))


def test_points_positions_equal_input_coordinates():
    model, (coords, feats) = points_model()
    with torch.no_grad():
        output = model((coords, feats))
    plugin = sparse.PointsPlugin()
    data = plugin.extract(output, points_ctx(coords))
    assert plugin.tensors(output) == [output]
    assert data.kind == "points"
    np.testing.assert_array_equal(data.positions, coords[0].numpy())
    assert data.positions.dtype == np.float32
    np.testing.assert_array_equal(data.features, output[0].numpy())
    assert data.grid is None
    assert data.scale is None
    assert data.labels is None


def test_inherit_positions_copies_ancestor_layout_for_matching_rows():
    _, full, _ = sparse_outputs()
    ancestor = sparse.Sparse3dPlugin().extract(full, sparse_ctx())
    features = torch.randn(len(ancestor.positions), 5)
    data = sparse.inherit_positions(features, ancestor)
    assert data.kind == "sparse3d"
    assert data.positions is ancestor.positions
    np.testing.assert_array_equal(data.features, features.numpy())
    assert data.grid == ancestor.grid
    assert data.scale == ancestor.scale


def test_inherit_positions_returns_none_for_other_row_count():
    _, full, _ = sparse_outputs()
    ancestor = sparse.Sparse3dPlugin().extract(full, sparse_ctx())
    assert sparse.inherit_positions(torch.randn(7, 5), ancestor) is None


def test_inherit_positions_returns_none_for_non_2d_tensor():
    _, full, _ = sparse_outputs()
    ancestor = sparse.Sparse3dPlugin().extract(full, sparse_ctx())
    rows = len(ancestor.positions)
    assert sparse.inherit_positions(torch.randn(1, rows, 5), ancestor) is None


class NamedPlugin:
    """
    Stand in for a registered dense plugin that only carries a name.

    """

    def __init__(self, name):
        """Store the plugin name."""
        self.name = name


def test_register_defaults_puts_sparse3d_first_and_points_before_tokens(monkeypatch):
    monkeypatch.setattr(base, "REGISTRY", [NamedPlugin("image2d"), NamedPlugin("tokens")])
    sparse.register_defaults()
    sparse.register_defaults()
    names = [plugin.name for plugin in base.REGISTRY]
    assert names == ["sparse3d", "image2d", "points", "tokens"]


def test_import_registers_both_plugins():
    names = [plugin.name for plugin in base.REGISTRY]
    assert "sparse3d" in names
    assert "points" in names


def test_plugin_for_picks_sparse3d_for_sparse_output():
    _, full, _ = sparse_outputs()
    assert base.plugin_for(full, sparse_ctx()).name == "sparse3d"

