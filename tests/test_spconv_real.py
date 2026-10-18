"""
Test that capture places the layers of a real spconv model, and skip without spconv.

Purpose:

- Check that every spconv layer is a sparse3d node with one feature column per channel.
- Check that BatchNorm1d and ReLU inherit the conv's positions and a stride 2 conv halves them.

Special Notes:

- Skipped when spconv.pytorch cannot be imported, so only a machine with spconv runs it.
- The import sits in a fixture, so pytest still collects the tests and exits 0 when skipping.
- Runs on the CPU when this spconv build allows it, otherwise on cuda, otherwise skips.
- Layers are checked by module name: 0 SubMConv3d, 1 BatchNorm1d, 2 ReLU, 3 SparseConv3d.
"""

import unittest

import numpy as np
import pytest
import torch
from torch import nn

from layerviz.api import capture
from tests.test_api import read_graph, read_positions
from tests.test_bundle import read_frame

VOXELS = 50
GRID = 16
CHANNELS_IN = 4
CHANNELS = {"0": 8, "1": 8, "2": 8, "3": 6}


@pytest.fixture(scope="module")
def spconv():
    """Import spconv.pytorch, or skip every test of this file when it is missing."""
    return pytest.importorskip("spconv.pytorch")


def build_spconv_model(spconv) -> nn.Module:
    """Build the four layer SparseSequential, with the last conv at stride 2."""
    torch.manual_seed(0)
    model = spconv.SparseSequential(
        spconv.SubMConv3d(CHANNELS_IN, 8, 3, padding=1, indice_key="subm"),
        nn.BatchNorm1d(8),
        nn.ReLU(),
        spconv.SparseConv3d(8, 6, 3, stride=2, padding=1),
    )
    return model.eval()


def build_spconv_batch(spconv):
    """Build a SparseConvTensor of VOXELS unique random voxels in a GRID cube, on the CPU."""
    flat = torch.randperm(GRID**3)[:VOXELS]
    cells = torch.stack([flat // GRID**2, (flat // GRID) % GRID, flat % GRID], dim=1)
    indices = torch.cat([torch.zeros(VOXELS, 1, dtype=torch.long), cells], dim=1)
    features = torch.randn(VOXELS, CHANNELS_IN)
    return spconv.SparseConvTensor(features, indices.int(), [GRID] * 3, 1)


def forward_on(spconv, device: str) -> tuple:
    """Run the model on a device and return (model, batch, output), raising if it cannot."""
    model, cpu_batch = build_spconv_model(spconv).to(device), build_spconv_batch(spconv)
    batch = spconv.SparseConvTensor(
        cpu_batch.features.to(device), cpu_batch.indices.to(device), cpu_batch.spatial_shape, 1
    )
    with torch.no_grad():
        return model, batch, model(batch)


def run_on_available_device(spconv) -> tuple:
    """Return (model, batch, output) from the CPU, else cuda, else skip the test."""
    try:
        return forward_on(spconv, "cpu")
    except Exception as cpu_error:  # spconv builds fail on the CPU in many different ways.
        if not torch.cuda.is_available():
            reason = f"spconv cannot run on the CPU and cuda is missing: {cpu_error}"
            raise unittest.SkipTest(reason)
        return forward_on(spconv, "cuda")


@pytest.fixture(scope="module")
def captured(spconv, tmp_path_factory) -> dict:
    """Capture the spconv model once and return its folder, input batch, and output."""
    model, batch, output = run_on_available_device(spconv)
    out = tmp_path_factory.mktemp("spconv") / "bundle"
    capture(model, batch, out, inline_mb=0)
    return {"out": out, "batch": batch, "output": output}


def leaf_nodes(captured: dict) -> dict:
    """Map the module name of each numbered layer to its graph node."""
    nodes = read_graph(captured["out"])["nodes"]
    return {node["name"]: node for node in nodes if node["name"] in CHANNELS}


def layer_positions(captured: dict, name: str) -> np.ndarray:
    """Decode the positions file of one named layer."""
    path = captured["out"] / leaf_nodes(captured)[name]["file"]
    return read_positions(path.read_bytes())


def sorted_rows(array: np.ndarray) -> np.ndarray:
    """Return the rows of an array in lexicographic order, since voxel order is not fixed."""
    return np.unique(np.asarray(array, dtype=np.float32), axis=0)


def voxel_cells(sparse) -> np.ndarray:
    """Return the (z, y, x) cells of a sparse tensor as a float array [N, 3]."""
    return sparse.indices[:, 1:].cpu().numpy().astype(np.float32)


def test_every_spconv_layer_is_a_sparse3d_node(captured):
    nodes = leaf_nodes(captured)

    assert sorted(nodes) == sorted(CHANNELS)
    assert {name: node["kind"] for name, node in nodes.items()} == dict.fromkeys(
        CHANNELS, "sparse3d"
    )


def test_every_spconv_layer_has_one_feature_channel_per_output_channel(captured):
    nodes = leaf_nodes(captured)

    for name, channels in CHANNELS.items():
        header, _, _ = read_frame((captured["out"] / nodes[name]["channels"]).read_bytes())
        assert header["c"] == channels, name


def test_submanifold_conv_keeps_the_input_voxels_at_their_cells(captured):
    positions = layer_positions(captured, "0")

    assert positions.shape == (VOXELS, 3)
    expected = voxel_cells(captured["batch"])
    np.testing.assert_array_equal(sorted_rows(positions), sorted_rows(expected))


@pytest.mark.parametrize("name", ["1", "2"])
def test_batchnorm_and_relu_inherit_the_positions_of_the_conv(captured, name):
    np.testing.assert_array_equal(
        layer_positions(captured, name), layer_positions(captured, "0")
    )


def test_stride_two_conv_places_voxels_at_twice_the_cell_plus_a_half(captured):
    expected = voxel_cells(captured["output"]) * 2 + 0.5

    positions = layer_positions(captured, "3")

    assert 0 < len(positions) <= VOXELS
    np.testing.assert_array_equal(sorted_rows(positions), sorted_rows(expected))
