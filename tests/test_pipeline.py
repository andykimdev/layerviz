"""
Test that run_pipeline turns one model and batch into captured, reduced layers.

Purpose:

- Check layer kinds per shape, inherited sparse positions, overrides, and skips.
- Check the budget, the channel cap, and that the model modes are restored.
"""

import numpy as np
import pytest
import torch
from torch import nn

from layerviz.budget import BYTES_PER_POSITION, COMPARE_BYTES_PER_POSITION, MB
from layerviz.bundle import LayerPayload, encode_channels, encode_layer, layer_overhead
from layerviz.pipeline import CaptureOptions, prepared, run_pipeline
from layerviz.shapes import base
from tests.models import (
    ALL_SHAPES,
    FakeSparseTensor,
    graph_model,
    image2d_model,
    make_voxel_indices,
    sparse_model,
    wide_volume3d_model,
)

TOKEN_LAYERS = ["0", "1.dropout1", "1.norm1", "1.linear1", "1.dropout", "1.linear2"]
EXPECTED_LAYERS = {
    "image2d": [("0", "image2d"), ("1", "image2d")],
    "volume3d": [("0", "volume3d"), ("1", "volume3d")],
    "tokens": [(name, "tokens") for name in TOKEN_LAYERS + ["1.dropout2", "1.norm2"]],
    "vector": [("0", "vector"), ("1", "vector")],
    "points": [("first", "points"), ("second", "points")],
    "sparse": [(name, "sparse3d") for name in ["0", "0.linear", "1", "2", "2.linear"]],
    "graph": [
        (name, "image2d")
        for name in ["stem", "res_conv", "down", "up", "head", "extra_negative"]
    ],
}
EXPECTED_SKIPS = {"tokens": {"1.self_attn": "no shape plugin accepts tuple"}}


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test its own copy of the plugin registry."""
    monkeypatch.setattr(base, "REGISTRY", base.plugins())


def layers_by_name(result) -> dict:
    """Map each captured layer name to its CapturedLayer."""
    return {layer.node.name: layer for layer in result.layers}


def mb_for(positions: int, overhead: int = 0) -> float:
    """Return the size in megabytes that holds exactly this many positions and overhead bytes."""
    return (positions * BYTES_PER_POSITION + overhead) / MB


def largest_overhead(model, batch) -> int:
    """Return the largest layer file overhead among the model's full, untrimmed layers."""
    layers = run_pipeline(model, batch, CaptureOptions()).layers
    return max(layer_overhead(layer.data) for layer in layers)


class Tagger(nn.Module):
    """
    Return a string label that the parent model ignores.

    """

    def forward(self, x: torch.Tensor) -> str:
        """Return a fixed label."""
        return "label"


class NanLayer(nn.Module):
    """
    Return the input times NaN.

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor of NaN values shaped like x."""
        return x * float("nan")


class Failing(nn.Module):
    """
    Raise inside the forward pass.

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raise a RuntimeError."""
        raise RuntimeError("boom")


class TaggedNet(nn.Module):
    """
    Run a Linear layer around a Tagger, a NanLayer, and a final Linear layer.

    """

    def __init__(self) -> None:
        """Create the layers."""
        super().__init__()
        self.first = nn.Linear(10, 3)
        self.tag = Tagger()
        self.bad = NanLayer()
        self.last = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return last(first(x)) after calling tag and bad on the side."""
        hidden = self.first(x)
        self.tag(hidden)
        self.bad(hidden)
        return self.last(hidden)


class CustomPlugin:
    """
    Accept every tensor and place its rows by index under the kind custom.

    """

    name = "custom"

    def accepts(self, output, ctx) -> bool:
        """Return whether the output is a tensor."""
        return isinstance(output, torch.Tensor)

    def tensors(self, output) -> list:
        """Return the output in a list."""
        return [output]

    def extract(self, output, ctx) -> base.LayerData:
        """Return the output rows at index positions."""
        features = output.detach().reshape(output.shape[0], -1).numpy()
        positions = np.arange(len(features), dtype=np.float32)[:, None]
        return base.LayerData("custom", positions, features, None, None, None)


@pytest.mark.parametrize("shape", sorted(ALL_SHAPES))
def test_every_shape_captures_one_layer_per_module_with_its_kind(shape: str) -> None:
    model, batch = ALL_SHAPES[shape]()
    result = run_pipeline(model, batch, CaptureOptions())
    assert [(layer.node.name, layer.data.kind) for layer in result.layers] == (
        EXPECTED_LAYERS[shape]
    )
    assert result.skipped == EXPECTED_SKIPS.get(shape, {})


def test_model_name_and_input_layer_describe_the_batch() -> None:
    model, batch = image2d_model()
    result = run_pipeline(model, batch, CaptureOptions())
    assert result.model_name == "Sequential"
    assert result.input.kind == "image2d"
    assert result.input.features.shape == (256, 3)
    assert [node.name for node in result.trace.nodes] == ["input", "", "0", "1"]


def test_points_input_layer_sits_on_the_batch_coordinates() -> None:
    model, batch = ALL_SHAPES["points"]()
    result = run_pipeline(model, batch, CaptureOptions())
    assert result.input.kind == "points"
    np.testing.assert_array_equal(result.input.positions, batch[0][0].numpy())
    np.testing.assert_array_equal(result.input.features, batch[1][0].numpy())


def test_budget_subsamples_every_layer_and_reduces_the_kept_rows() -> None:
    model, batch = image2d_model()
    layer_mb = mb_for(10, largest_overhead(model, batch))
    result = run_pipeline(model, batch, CaptureOptions(layer_mb=layer_mb))
    layers = layers_by_name(result)
    assert [(layer.total, layer.kept) for layer in result.layers] == [(256, 10), (64, 10)]
    first = layers["0"]
    assert first.data.positions.shape == (10, 2)
    assert first.data.features.shape == (10, 4)
    assert first.reduced.rgb.shape == (10, 3)
    assert first.channels[0].dtype == np.uint8
    assert first.channels[0].shape == (10, 4)
    np.testing.assert_array_equal(first.channels[1], np.arange(4))


def test_channel_file_with_its_header_fits_the_layer_cap() -> None:
    torch.manual_seed(0)
    model, batch = nn.Sequential(nn.Linear(10, 100)), torch.randn(64, 10)
    layer_mb = 4096 / MB
    layer = run_pipeline(model, batch, CaptureOptions(layer_mb=layer_mb)).layers[0]
    payload = LayerPayload(layer.data, layer.reduced, channels=layer.channels)
    size, kept = len(encode_channels(payload)), len(layer.channels[1])
    positions = (4096 - largest_overhead(model, batch)) // BYTES_PER_POSITION
    assert (layer.kept, size <= 4096, 0 < kept < 100) == (positions, True, True)


def test_layer_trimmed_to_the_cap_encodes_within_layer_mb() -> None:
    model, batch = wide_volume3d_model()
    layer = run_pipeline(model, batch, CaptureOptions(layer_mb=4096 / MB)).layers[-1]
    size = len(encode_layer(LayerPayload(layer.data, layer.reduced)))
    assert (layer.kept < layer.total, size <= 4096) == (True, True)


def test_channels_under_the_layer_cap_are_all_kept() -> None:
    torch.manual_seed(0)
    model, batch = nn.Sequential(nn.Linear(10, 6)), torch.randn(4, 10)
    data, kept = run_pipeline(model, batch, CaptureOptions(layer_mb=1)).layers[0].channels
    assert (data.shape, kept.tolist()) == ((4, 6), list(range(6)))


def test_bytes_per_position_option_sets_the_kept_positions() -> None:
    # 2360 bytes past the file overhead hold 29 positions of 79 bytes but only 28 of 82 bytes.
    model, batch = image2d_model()
    layer_mb = (2360 + largest_overhead(model, batch)) / MB
    opts = CaptureOptions(layer_mb=layer_mb, bytes_per_position=COMPARE_BYTES_PER_POSITION)
    assert [layer.kept for layer in run_pipeline(model, batch, opts).layers] == [28, 28]


def test_default_bytes_per_position_is_the_single_mode_size() -> None:
    assert CaptureOptions().bytes_per_position == BYTES_PER_POSITION


def test_layer_cap_below_one_compare_position_names_its_size() -> None:
    model, batch = image2d_model()
    opts = CaptureOptions(layer_mb=80 / MB, bytes_per_position=COMPARE_BYTES_PER_POSITION)
    with pytest.raises(ValueError, match=r"at least 7\.82e-05 MB"):
        run_pipeline(model, batch, opts)


def test_layer_cap_below_one_position_raises() -> None:
    model, batch = image2d_model()
    with pytest.raises(ValueError, match="layer_mb"):
        run_pipeline(model, batch, CaptureOptions(layer_mb=mb_for(0.5)))


def test_sparse_linear_layers_inherit_positions_of_their_sparse_ancestor() -> None:
    model, batch = sparse_model()
    layers = layers_by_name(run_pipeline(model, batch, CaptureOptions()))
    for child, ancestor, grid in [("0.linear", "0", (8, 8, 8)), ("2.linear", "1", (4, 4, 4))]:
        data = layers[child].data
        assert (data.kind, data.grid) == ("sparse3d", grid)
        np.testing.assert_array_equal(data.positions, layers[ancestor].data.positions)
        np.testing.assert_allclose(data.features, layers[child].node.output.numpy())


def two_sample_sparse_batch() -> FakeSparseTensor:
    """Return a sparse batch with 30 voxels in sample 0 and 20 voxels in sample 1."""
    torch.manual_seed(1)
    second = make_voxel_indices(20, 8)
    second[:, 0] = 1
    indices = torch.cat([make_voxel_indices(30, 8), second])
    return FakeSparseTensor(torch.randn(50, 4), indices, [8, 8, 8], 2)


@pytest.mark.parametrize("sample", [0, 1])
def test_linear_layers_of_a_two_sample_sparse_batch_inherit_the_sample_rows(
    sample: int,
) -> None:
    model, _ = sparse_model()
    result = run_pipeline(model, two_sample_sparse_batch(), CaptureOptions(sample_index=sample))
    layers = layers_by_name(result)
    assert [(layer.node.name, layer.data.kind) for layer in result.layers] == (
        EXPECTED_LAYERS["sparse"]
    )
    for child, ancestor in [("0.linear", "0"), ("2.linear", "1")]:
        rows = layers[ancestor].node.output.indices[:, 0] == sample
        data = layers[child].data
        assert data.grid == layers[ancestor].data.grid
        np.testing.assert_array_equal(data.positions, layers[ancestor].data.positions)
        np.testing.assert_allclose(data.features, layers[child].node.output[rows].numpy())
    assert len(layers["0.linear"].data.positions) == (30 if sample == 0 else 20)


def test_shapes_option_overrides_a_2d_output_to_tokens() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Sequential(nn.Linear(10, 16), nn.Linear(16, 5)))
    batch = torch.randn(4, 10)
    result = run_pipeline(model, batch, CaptureOptions(shapes={"*.1": "tokens"}))
    layers = layers_by_name(result)
    assert [(name, layer.data.kind) for name, layer in layers.items()] == [
        ("0.0", "vector"),
        ("0.1", "tokens"),
    ]
    expected = model(batch)[0].detach().numpy()[:, None]
    np.testing.assert_allclose(layers["0.1"].data.features, expected, rtol=1e-6)
    np.testing.assert_array_equal(
        layers["0.1"].data.positions, np.arange(5, dtype=np.float32)[:, None]
    )


def test_registered_plugin_handles_the_layers_it_accepts() -> None:
    base.register_shape(CustomPlugin())
    model, batch = ALL_SHAPES["vector"]()
    result = run_pipeline(model, batch, CaptureOptions())
    assert [(layer.node.name, layer.data.kind) for layer in result.layers] == [
        ("0", "custom"),
        ("1", "custom"),
    ]


def test_unaccepted_and_nan_outputs_are_skipped_and_the_run_finishes() -> None:
    torch.manual_seed(0)
    result = run_pipeline(TaggedNet(), torch.randn(4, 10), CaptureOptions())
    assert [layer.node.name for layer in result.layers] == ["first", "last"]
    assert result.skipped == {
        "tag": "no shape plugin accepts str",
        "bad": "features hold 12 NaN values",
    }


def test_sample_with_no_voxels_skips_its_sparse_layers() -> None:
    model, batch = sparse_model()
    result = run_pipeline(model, batch, CaptureOptions(sample_index=1))
    reason = "no positions in this sample"
    assert result.skipped == {
        name: reason for name in ["0", "0.linear", "1", "2", "2.linear"]
    }
    assert result.layers == []


def test_include_and_exclude_patterns_filter_layers() -> None:
    model, batch = graph_model()
    opts = CaptureOptions(include=["stem", "head*", "extra_*"], exclude=["extra_*"])
    result = run_pipeline(model, batch, opts)
    assert [layer.node.name for layer in result.layers] == ["stem", "head"]
    assert result.skipped == {}


def test_train_and_grad_modes_are_restored_after_capture() -> None:
    model, batch = image2d_model()
    model.train()
    model[1].eval()
    run_pipeline(model, batch, CaptureOptions())
    assert [m.training for m in model.modules()] == [True, True, False]
    assert torch.is_grad_enabled()


def test_train_and_grad_modes_are_restored_when_the_forward_raises() -> None:
    model = nn.Sequential(nn.Linear(10, 3), Failing()).train()
    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(model, torch.randn(4, 10), CaptureOptions())
    assert [m.training for m in model.modules()] == [True, True, True]
    assert torch.is_grad_enabled()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_prepared_unwraps_a_compiled_model_into_eval_and_no_grad() -> None:
    model, _ = image2d_model()
    model.train()
    with prepared(torch.compile(model)) as inner:
        assert inner is model
        assert not model.training
        assert not torch.is_grad_enabled()
    assert model.training
