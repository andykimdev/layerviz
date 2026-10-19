"""
Test that run_pipeline turns one model and batch into captured, reduced layers.

Purpose:

- Check layer kinds per shape, inherited sparse positions, overrides, and skips.
- Check that the model modes are restored and a raising forward leaves no trace behind.

Special Notes:

- The budget and size cap tests live in test_pipeline_budget.py.
"""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from layerviz.pipeline import CaptureOptions, prepared, run_pipeline
from layerviz.shapes import base
from layerviz.tags import tags_of
from tests.models import (
    ALL_SHAPES,
    FakeSparseTensor,
    graph_model,
    image2d_model,
    make_voxel_indices,
    sparse_model,
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


class FailingOnce(nn.Module):
    """
    Raise on the first forward while armed, then return the input unchanged.

    """

    def __init__(self, armed: bool = True) -> None:
        """Set whether the next forward raises."""
        super().__init__()
        self.armed = armed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raise a RuntimeError once when armed, else return x."""
        if self.armed:
            self.armed = False
            raise RuntimeError("boom")
        return x


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


@pytest.mark.parametrize("shape", sorted(ALL_SHAPES))
def test_every_shape_skips_only_the_outputs_no_plugin_accepts(shape: str) -> None:
    model, batch = ALL_SHAPES[shape]()
    result = run_pipeline(model, batch, CaptureOptions())
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


# Each plain [N, C] Linear child of a sparse model and the sparse layer it inherits from.
SPARSE_CHILDREN = [("0.linear", "0"), ("2.linear", "1")]


@pytest.mark.parametrize(("child", "grid"), [("0.linear", (8, 8, 8)), ("2.linear", (4, 4, 4))])
def test_sparse_linear_layers_take_the_kind_and_grid_of_their_ancestor(
    child: str, grid: tuple
) -> None:
    model, batch = sparse_model()
    data = layers_by_name(run_pipeline(model, batch, CaptureOptions()))[child].data
    assert (data.kind, data.grid) == ("sparse3d", grid)


@pytest.mark.parametrize(("child", "ancestor"), SPARSE_CHILDREN)
def test_sparse_linear_layers_inherit_positions_of_their_sparse_ancestor(
    child: str, ancestor: str
) -> None:
    model, batch = sparse_model()
    layers = layers_by_name(run_pipeline(model, batch, CaptureOptions()))
    np.testing.assert_array_equal(layers[child].data.positions, layers[ancestor].data.positions)


@pytest.mark.parametrize("child", ["0.linear", "2.linear"])
def test_sparse_linear_layers_keep_their_own_output_as_features(child: str) -> None:
    model, batch = sparse_model()
    layer = layers_by_name(run_pipeline(model, batch, CaptureOptions()))[child]
    np.testing.assert_allclose(layer.data.features, layer.node.output.numpy())


def two_sample_sparse_batch() -> FakeSparseTensor:
    """Return a sparse batch with 30 voxels in sample 0 and 20 voxels in sample 1."""
    torch.manual_seed(1)
    second = make_voxel_indices(20, 8)
    second[:, 0] = 1
    indices = torch.cat([make_voxel_indices(30, 8), second])
    return FakeSparseTensor(torch.randn(50, 4), indices, [8, 8, 8], 2)


def two_sample_layers(sample: int) -> dict:
    """Capture the sparse model on the two sample batch and map names to captured layers."""
    model, _ = sparse_model()
    opts = CaptureOptions(sample_index=sample)
    return layers_by_name(run_pipeline(model, two_sample_sparse_batch(), opts))


@pytest.mark.parametrize("sample", [0, 1])
def test_every_sample_of_a_two_sample_sparse_batch_captures_every_sparse_layer(
    sample: int,
) -> None:
    layers = two_sample_layers(sample)
    assert [(name, layer.data.kind) for name, layer in layers.items()] == (
        EXPECTED_LAYERS["sparse"]
    )


@pytest.mark.parametrize("sample", [0, 1])
@pytest.mark.parametrize(("child", "ancestor"), SPARSE_CHILDREN)
def test_linear_layers_of_a_two_sample_sparse_batch_inherit_the_sample_positions(
    sample: int, child: str, ancestor: str
) -> None:
    layers = two_sample_layers(sample)
    child_data, ancestor_data = layers[child].data, layers[ancestor].data
    assert child_data.grid == ancestor_data.grid
    np.testing.assert_array_equal(child_data.positions, ancestor_data.positions)


@pytest.mark.parametrize("sample", [0, 1])
@pytest.mark.parametrize(("child", "ancestor"), SPARSE_CHILDREN)
def test_linear_layers_of_a_two_sample_sparse_batch_keep_the_sample_rows(
    sample: int, child: str, ancestor: str
) -> None:
    layers = two_sample_layers(sample)
    rows = layers[ancestor].node.output.indices[:, 0] == sample
    expected = layers[child].node.output[rows].numpy()
    np.testing.assert_allclose(layers[child].data.features, expected)


@pytest.mark.parametrize(("sample", "voxels"), [(0, 30), (1, 20)])
def test_two_sample_sparse_batch_keeps_the_voxels_of_the_chosen_sample(
    sample: int, voxels: int
) -> None:
    assert len(two_sample_layers(sample)["0.linear"].data.positions) == voxels


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
    model, one = sparse_model()
    # Every voxel stays in sample 0, so sample 1 is in the batch but holds no voxels.
    batch = FakeSparseTensor(one.features, one.indices, one.spatial_shape, 2)
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


def trace_summary(result) -> tuple:
    """Return the node names, the provenance edges, and the captured layer names of a run."""
    nodes = [node.name for node in result.trace.nodes]
    return nodes, result.trace.edges, [layer.node.name for layer in result.layers]


def test_capture_after_a_raising_forward_traces_like_a_fresh_model() -> None:
    torch.manual_seed(0)
    model, batch = nn.Sequential(nn.Linear(10, 3), FailingOnce()), torch.randn(4, 10)
    fresh = copy.deepcopy(model)
    fresh[1].armed = False
    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(model, batch, CaptureOptions())
    again = trace_summary(run_pipeline(model, batch, CaptureOptions()))
    assert again[0] == ["input", "", "0", "1"]
    assert again == trace_summary(run_pipeline(fresh, batch, CaptureOptions()))



def test_plain_forward_after_a_raising_capture_returns_an_untagged_output() -> None:
    # A hook left behind would tag the output of every later forward with its module name.
    torch.manual_seed(0)
    model, batch = nn.Sequential(nn.Linear(10, 3), FailingOnce()), torch.randn(4, 10)
    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(model, batch, CaptureOptions())
    assert tags_of([model(batch)]) == frozenset()



def test_batch_after_a_raising_capture_carries_no_tag() -> None:
    torch.manual_seed(0)
    model, batch = nn.Sequential(nn.Linear(10, 3), FailingOnce()), torch.randn(4, 10)
    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(model, batch, CaptureOptions())
    assert tags_of([batch]) == frozenset()


@pytest.mark.slow
def test_prepared_unwraps_a_compiled_model_into_eval_and_no_grad() -> None:
    model, _ = image2d_model()
    model.train()
    with prepared(torch.compile(model)) as inner:
        assert inner is model
        assert not model.training
        assert not torch.is_grad_enabled()
    assert model.training
