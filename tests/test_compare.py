"""
Test the comparison bundle that pairs two models' layers by module name.

Purpose:

- Check that channel permuted, sign flipped copies color byte equal in both panes.
- Check the unpaired names, the CKA per node, the graph, and the options both models share.
- Check that compare restores the models and keeps every layer file within layer_mb.

Special Notes:

- Bundles are written as folders with a fake web folder, so tests read graph.json and the
  layer files directly.
- Joint colors, pane alignment, and the joint budget are tested in test_compare_joint.py.
"""

import copy

import numpy as np
import pytest
import torch
from torch import nn

import layerviz
from layerviz.budget import MB
from layerviz.comparison import (
    NO_SHARED, CompareOptions, compare, compare_graph, pair_payloads, shared_rows,
)
from layerviz.graph import EXPLAINED_COUNT
from layerviz.pipeline import CaptureOptions, run_pipeline
from tests.harness import (
    captured, expect_paired, other_model, pane_sections, run_compare, zscore,
)
from tests.models import graph_model, image2d_model, wide_volume3d_model

PERM0 = [2, 0, 3, 1]
SIGN0 = [1.0, -1.0, -1.0, 1.0]
PERM1 = [5, 3, 0, 1, 4, 2]
SIGN1 = [-1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
GRAPH_NAMES = ["input", "", "stem", "res_conv", "down", "up", "head", "extra_negative"]
GRAPH_EDGES = [
    ["L0000", "L0002"], ["L0002", "L0003"], ["L0003", "L0004"], ["L0002", "L0004"],
    ["L0004", "L0005"], ["L0005", "L0006"], ["L0002", "L0006"], ["L0006", "L0007"],
]


class TwoInputBlock(nn.Module):
    """Feed the first input to the left conv and the second input to the right conv."""

    def __init__(self) -> None:
        """Create the two 1x1 convs."""
        super().__init__()
        self.left, self.right = nn.Conv2d(4, 4, 1), nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the sum of the left conv of x and the right conv of y."""
        return self.left(x) + self.right(y)


class TwoInputNet(nn.Module):
    """Run two convs on the input and feed both outputs to one two input block."""

    def __init__(self) -> None:
        """Create the two producer convs and the block."""
        super().__init__()
        self.a, self.b, self.block = nn.Conv2d(3, 4, 1), nn.Conv2d(3, 4, 1), TwoInputBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the block output of the two producers."""
        return self.block(self.a(x), self.b(x))


class InputRecorder(nn.Module):
    """Keep a copy of every input it receives and pass it through a 1x1 conv."""

    def __init__(self) -> None:
        """Create the conv and the empty list of seen inputs."""
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Record a copy of x and return its conv."""
        self.seen.append(x.detach().clone())
        return self.conv(x)


def results(model_a, model_b, batch):
    """Run the pipeline with default options on two models and return both results."""
    return tuple(run_pipeline(model, batch, CaptureOptions()) for model in (model_a, model_b))


def permuted_copy(model):
    """Copy a two conv model with permuted, sign flipped channels and the same function up to them."""
    twin = copy.deepcopy(model)
    first, second = twin[0], twin[1]
    sign0, sign1 = torch.tensor(SIGN0), torch.tensor(SIGN1)
    with torch.no_grad():
        first.weight.copy_(first.weight[PERM0] * sign0[:, None, None, None])
        first.bias.copy_(first.bias[PERM0] * sign0)
        # The next conv undoes the first permutation and sign flip, so only its own remain.
        weight = second.weight[:, PERM0] * sign0[None, :, None, None]
        second.weight.copy_(weight[PERM1] * sign1[:, None, None, None])
        second.bias.copy_(second.bias[PERM1] * sign1)
    return twin


def reference_cka(x, y):
    """Compute linear CKA of standardized features with plain NumPy."""
    a, b = zscore(x), zscore(y)
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    return np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b))


def test_permuted_copy_colors_both_panes_byte_equal(tmp_path):
    model, batch = image2d_model()
    graph, out = run_compare(tmp_path, model, permuted_copy(model), batch)
    for node in expect_paired(graph, ["0", "1"]):
        pane_a, pane_b = pane_sections(out, node)
        np.testing.assert_array_equal(pane_a["positions"], pane_b["positions"])
        np.testing.assert_array_equal(pane_a["rgb"], pane_b["rgb"])


def test_identical_copy_has_equal_joint_rgb_and_cka_one(tmp_path):
    model, batch = image2d_model()
    graph, out = run_compare(tmp_path, model, copy.deepcopy(model), batch)
    for node in expect_paired(graph, ["0", "1"]):
        pane_a, pane_b = pane_sections(out, node)
        np.testing.assert_array_equal(pane_a["rgb_joint"], pane_b["rgb_joint"])
        assert node["cka"] == pytest.approx(1.0, abs=1e-6)


def test_cka_per_node_matches_numpy_reference(tmp_path):
    model, batch = image2d_model()
    other = other_model()
    graph, out = run_compare(tmp_path, model, other, batch)
    layers_a, layers_b = captured(model, batch), captured(other, batch)
    for node in expect_paired(graph, ["0", "1"]):
        name = node["name"]
        expected = reference_cka(layers_a[name].data.features, layers_b[name].data.features)
        assert node["cka"] == pytest.approx(expected, abs=1e-4)
        assert all("rgb_joint" in sections for sections in pane_sections(out, node))


def test_extra_layer_in_b_is_listed_unpaired(tmp_path):
    model, batch = image2d_model()
    longer = nn.Sequential(*copy.deepcopy(model), nn.Conv2d(6, 5, 1)).eval()
    graph, _ = run_compare(tmp_path, model, longer, batch, labels=("base", "long"))
    assert graph["meta"]["unpaired"] == {"a": [], "b": ["2"]}
    assert graph["meta"]["labels"] == ["base", "long"]
    assert graph["meta"]["mode"] == "compare"
    expect_paired(graph, ["0", "1"])


def test_extra_layer_in_a_stays_in_graph_without_panes(tmp_path):
    model, batch = image2d_model()
    longer = nn.Sequential(*copy.deepcopy(model), nn.Conv2d(6, 5, 1)).eval()
    graph, _ = run_compare(tmp_path, longer, model, batch)
    extra = next(node for node in graph["nodes"] if node["name"] == "2")
    assert graph["meta"]["unpaired"] == {"a": ["2"], "b": []}
    assert (extra["files"], extra["cka"], extra["skipped"]) == (None, None, "only in A")


def test_graph_nodes_follow_execution_ids_and_leaf_edges(tmp_path):
    model, batch = image2d_model()
    graph, _ = run_compare(tmp_path, model, copy.deepcopy(model), batch)
    ids = {node["name"]: node["id"] for node in graph["nodes"]}
    assert ids == {"input": "L0000", "": "L0001", "0": "L0002", "1": "L0003"}
    assert graph["input"] == "L0000"
    assert graph["edges"] == [["L0000", "L0002"], ["L0002", "L0003"]]
    root = next(node for node in graph["nodes"] if node["name"] == "")
    assert (root["leaf"], root["shares"]) == (False, "L0003")


def test_container_output_edge_leaves_from_its_last_leaf(tmp_path):
    model, batch = image2d_model()
    nested = nn.Sequential(nn.Sequential(model[0], model[1]), nn.Conv2d(6, 5, 1)).eval()
    graph, _ = run_compare(tmp_path, nested, copy.deepcopy(nested), batch)
    block = next(node for node in graph["nodes"] if node["name"] == "0")
    box = next(box for box in graph["boxes"] if box["name"] == "0")
    assert graph["edges"] == [["L0000", "L0003"], ["L0003", "L0004"], ["L0004", "L0005"]]
    assert (block["x"], block["y"]) == ((box["x0"] + box["x1"]) / 2, (box["y0"] + box["y1"]) / 2)


def test_compare_graph_of_a_model_with_itself_matches_the_written_out_graph():
    model, batch = graph_model()
    result_a, result_b = results(model, copy.deepcopy(model), batch)
    graph, _ = compare_graph(result_a, result_b, ("A", "B"), seed=0)
    nodes = graph["nodes"]
    assert [node["name"] for node in nodes] == GRAPH_NAMES
    assert [node["id"] for node in nodes] == [f"L{i:04d}" for i in range(8)]
    assert [node["kept"] for node in nodes] == [None, None, 64, 64, 16, 64, 64, 64]
    assert [node["y"] for node in nodes] == [0.0, 3.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert sorted(graph["edges"]) == sorted(GRAPH_EDGES)
    assert graph["input"] == "L0000"
    assert [(box["name"], box["y0"], box["y1"]) for box in graph["boxes"]] == [("", 0.7, 6.3)]
    assert {key: graph["meta"][key] for key in ("model", "labels", "mode", "unpaired")} == {
        "model": "GraphNet", "labels": ["A", "B"], "mode": "compare",
        "unpaired": {"a": [], "b": []},
    }
    pairs = [node for node in nodes if node["kept"]]
    assert [(node["cka"], node["joint"], node["total_b"]) for node in pairs] == [
        (pytest.approx(1.0), True, node["total"]) for node in pairs
    ]
    assert {len(node["explained"]) for node in pairs} == {EXPLAINED_COUNT}


def test_compare_graph_has_no_edge_from_second_producer_to_first_leaf():
    torch.manual_seed(0)
    model = TwoInputNet().eval()
    result_a, result_b = results(model, copy.deepcopy(model), torch.randn(1, 3, 8, 8))
    graph, _ = compare_graph(result_a, result_b, ("A", "B"), seed=0)
    ids = {node["name"]: node["id"] for node in graph["nodes"]}
    expected = [["input", "a"], ["input", "b"], ["a", "block.left"], ["b", "block.right"]]
    assert graph["edges"] == [[ids[a], ids[b]] for a, b in expected]


def test_compare_gives_model_b_the_input_model_a_received(tmp_path):
    torch.manual_seed(0)
    batch = torch.randn(1, 3, 8, 8)
    original = batch.clone()
    recorder = InputRecorder().eval()
    model_a = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(3, 4, 1)).eval()
    run_compare(tmp_path, model_a, recorder, batch)
    assert (torch.equal(recorder.seen[0], original), torch.equal(batch, original)) == (True, True)


@pytest.mark.parametrize(("shared", "rows", "reason"), [(2, [2, 2], None), (1, [], NO_SHARED)])
def test_pair_needs_two_shared_positions_for_panes(shared, rows, reason):
    model, batch = image2d_model()
    result_a, result_b = results(model, copy.deepcopy(model), batch)
    layer_b = next(layer for layer in result_b.layers if layer.node.name == "1")
    layer_b.data.positions[shared:] += 1000.0
    graph, payloads = compare_graph(result_a, result_b, ("A", "B"), seed=0)
    node = next(node for node in graph["nodes"] if node["name"] == "1")
    panes = payloads.get(node["id"], [])
    assert ([len(pane.data.positions) for pane in panes], node["skipped"]) == (rows, reason)


def test_compare_restores_training_mode_and_weights(tmp_path):
    model, batch = image2d_model()
    model.train()
    before = copy.deepcopy(model.state_dict())
    run_compare(tmp_path, model, copy.deepcopy(model), batch)
    assert model.training and all(module.training for module in model.modules())
    assert torch.is_grad_enabled()
    # torch.equal avoids the one second lazy import of torch.testing.
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_package_compare_resolves_to_comparison_module():
    assert layerviz.compare is compare


def test_compare_options_reject_labels_that_are_not_two():
    with pytest.raises(ValueError, match="two labels"):
        CompareOptions(labels=("only",))


def test_capture_options_apply_to_both_models(tmp_path):
    model, batch = image2d_model()
    capture = CaptureOptions(include=["1"])
    graph, _ = run_compare(tmp_path, model, copy.deepcopy(model), batch, capture=capture)
    expect_paired(graph, ["1"])
    assert graph["meta"]["unpaired"] == {"a": [], "b": []}


def test_shared_rows_pairs_equal_positions_in_a_order():
    pos_a = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
    pos_b = np.array([[2, 0], [5, 5], [0, 0]], dtype=np.float32)
    rows_a, rows_b = shared_rows(pos_a, pos_b)
    assert (rows_a.tolist(), rows_b.tolist()) == ([0, 2], [2, 0])


def test_pair_without_shared_positions_gives_no_panes():
    model, batch = image2d_model()
    layers = captured(model, batch)
    moved = copy.deepcopy(layers["1"])
    moved.data.positions[:] += 1000.0
    assert pair_payloads(layers["1"], moved, seed=0) is None


def test_compare_layer_trimmed_by_layer_mb_writes_every_file_within_it(tmp_path):
    model, batch = wide_volume3d_model()
    capture = CaptureOptions(layer_mb=4096 / MB)
    graph, out = run_compare(tmp_path, model, copy.deepcopy(model), batch, capture=capture)
    node = expect_paired(graph, ["0"])[0]
    sizes = [path.stat().st_size for path in (out / "layers").iterdir()]
    assert (node["kept"] < node["total"], max(sizes) <= 4096) == (True, True)
