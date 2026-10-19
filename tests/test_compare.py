"""
Test the comparison bundle that pairs two models' layers by module name.

Purpose:

- Check that channel permuted, sign flipped copies color byte equal in both panes.
- Check the unpaired names, the CKA per node, and the joint RGB sections.
- Check that layers of different widths are flagged as having no joint colors.
- Check that compare budgets the joint RGB and keeps every layer file within layer_mb.

Special Notes:

- Bundles are written as folders with a fake web folder, so tests read graph.json and the
  layer files directly.
"""

import copy
import json
import struct

import numpy as np
import pytest
import torch
from torch import nn

import layerviz
from layerviz.comparison import CompareOptions, compare, pair_payloads, shared_rows
from layerviz.align import aligned_rgb, joint_rgb
from layerviz.budget import BYTES_PER_POSITION, COMPARE_BYTES_PER_POSITION, MB
from layerviz.pipeline import CaptureOptions, run_pipeline
from layerviz.reduce import standardize
from tests.models import image2d_model, wide_volume3d_model

PERM0 = [2, 0, 3, 1]
SIGN0 = [1.0, -1.0, -1.0, 1.0]
PERM1 = [5, 3, 0, 1, 4, 2]
SIGN1 = [-1.0, 1.0, 1.0, -1.0, 1.0, -1.0]


def make_web(root):
    """Write a minimal web folder with the page template and an entry module."""
    web = root / "web"
    web.mkdir()
    (web / "index.html").write_text("<html><body><!-- layerviz:scripts --></body></html>")
    (web / "app.js").write_text("console.log(1);\n")
    return web


def run_compare(tmp_path, model_a, model_b, batch, **options):
    """Write a comparison folder bundle with CompareOptions fields and return (graph, out)."""
    out = tmp_path / "out"
    opts = CompareOptions(inline_mb=0, web_dir=make_web(tmp_path), **options)
    compare(model_a, model_b, batch, out, opts)
    return json.loads((out / "graph.json").read_text()), out


def read_sections(path):
    """Decode a framed layer file into its header and its sections as NumPy arrays."""
    blob = path.read_bytes()
    (length,) = struct.unpack("<I", blob[:4])
    header = json.loads(blob[4 : 4 + length].decode("utf-8"))
    data = blob[4 + length :]
    sections = {}
    for name, (offset, size, dtype) in header["sections"].items():
        sections[name] = np.frombuffer(data[offset : offset + size], dtype=np.dtype(dtype))
    return header, sections


def pane_sections(out, node):
    """Return the decoded sections of pane a and pane b of one node."""
    return [read_sections(out / node["files"][pane])[1] for pane in ("a", "b")]


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


def paired_nodes(graph):
    """Return the graph nodes that carry two panes."""
    return [node for node in graph["nodes"] if node["files"] is not None]


def other_model(width=6):
    """Build an image2d shaped model with different random weights and a given last width."""
    torch.manual_seed(1)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.Conv2d(4, width, 3, stride=2, padding=1)
    ).eval()


def captured(model, batch):
    """Return the captured layers of a model by module name."""
    return {layer.node.name: layer for layer in run_pipeline(model, batch, CaptureOptions()).layers}


def reference_cka(x, y):
    """Compute linear CKA of standardized features with plain NumPy."""
    def zscore(f):
        f = np.asarray(f, dtype=np.float64)
        std = f.std(axis=0)
        return np.where(std > 0, (f - f.mean(axis=0)) / np.where(std > 0, std, 1), 0)

    a, b = zscore(x), zscore(y)
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    return np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b))


def test_permuted_copy_colors_both_panes_byte_equal(tmp_path):
    model, batch = image2d_model()
    graph, out = run_compare(tmp_path, model, permuted_copy(model), batch)
    nodes = paired_nodes(graph)
    assert [node["name"] for node in nodes] == ["0", "1"]
    for node in nodes:
        pane_a, pane_b = pane_sections(out, node)
        np.testing.assert_array_equal(pane_a["positions"], pane_b["positions"])
        np.testing.assert_array_equal(pane_a["rgb"], pane_b["rgb"])


def test_identical_copy_has_equal_joint_rgb_and_cka_one(tmp_path):
    model, batch = image2d_model()
    graph, out = run_compare(tmp_path, model, copy.deepcopy(model), batch)
    for node in paired_nodes(graph):
        pane_a, pane_b = pane_sections(out, node)
        np.testing.assert_array_equal(pane_a["rgb_joint"], pane_b["rgb_joint"])
        assert node["cka"] == pytest.approx(1.0, abs=1e-6)


def test_cka_per_node_matches_numpy_reference(tmp_path):
    model, batch = image2d_model()
    other = other_model()
    graph, out = run_compare(tmp_path, model, other, batch)
    layers_a, layers_b = captured(model, batch), captured(other, batch)
    for node in paired_nodes(graph):
        expected = reference_cka(layers_a[node["name"]].data.features, layers_b[node["name"]].data.features)
        assert node["cka"] == pytest.approx(expected, abs=1e-4)
        assert all("rgb_joint" in sections for sections in pane_sections(out, node))


def test_joint_rgb_comes_from_one_fit_over_both_models(tmp_path):
    model, batch = image2d_model()
    other = other_model()
    graph, out = run_compare(tmp_path, model, other, batch)
    layers_a, layers_b = captured(model, batch), captured(other, batch)
    for node in paired_nodes(graph):
        z_a = standardize(layers_a[node["name"]].data.features)[0]
        z_b = standardize(layers_b[node["name"]].data.features)[0]
        expected = joint_rgb(z_a, z_b, seed=0)
        for pane, rgb in zip(pane_sections(out, node), expected):
            np.testing.assert_array_equal(pane["rgb_joint"], rgb.reshape(-1))


def test_pane_b_colors_align_to_pane_a(tmp_path):
    model, batch = image2d_model()
    other = other_model()
    graph, out = run_compare(tmp_path, model, other, batch)
    layers_a, layers_b = captured(model, batch), captured(other, batch)
    for node in paired_nodes(graph):
        expected = aligned_rgb(layers_a[node["name"]].reduced, layers_b[node["name"]].reduced)
        np.testing.assert_array_equal(pane_sections(out, node)[1]["rgb"], expected.reshape(-1))


def test_extra_layer_in_b_is_listed_unpaired(tmp_path):
    model, batch = image2d_model()
    longer = nn.Sequential(*copy.deepcopy(model), nn.Conv2d(6, 5, 1)).eval()
    graph, _ = run_compare(tmp_path, model, longer, batch, labels=("base", "long"))
    assert graph["meta"]["unpaired"] == {"a": [], "b": ["2"]}
    assert graph["meta"]["labels"] == ["base", "long"]
    assert graph["meta"]["mode"] == "compare"
    assert [node["name"] for node in paired_nodes(graph)] == ["0", "1"]


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


def test_compare_restores_training_mode_and_weights(tmp_path):
    model, batch = image2d_model()
    model.train()
    before = copy.deepcopy(model.state_dict())
    run_compare(tmp_path, model, copy.deepcopy(model), batch)
    assert model.training and all(module.training for module in model.modules())
    assert torch.is_grad_enabled()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_package_compare_resolves_to_comparison_module():
    assert layerviz.compare is compare


def test_compare_options_reject_labels_that_are_not_two():
    with pytest.raises(ValueError, match="two labels"):
        CompareOptions(labels=("only",))


def test_capture_options_apply_to_both_models(tmp_path):
    model, batch = image2d_model()
    capture = CaptureOptions(include=["1"])
    graph, _ = run_compare(tmp_path, model, copy.deepcopy(model), batch, capture=capture)
    assert [node["name"] for node in paired_nodes(graph)] == ["1"]
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


def test_pair_with_different_widths_has_no_joint_colors():
    model, batch = image2d_model()
    layer_a, layer_b = captured(model, batch)["1"], captured(other_model(width=7), batch)["1"]
    pair = pair_payloads(layer_a, layer_b, seed=0)
    blank = np.zeros((layer_a.data.positions.shape[0], 3), dtype=np.uint8)
    assert pair.joint is False
    np.testing.assert_array_equal(pair.pane_a.rgb_joint, blank)
    np.testing.assert_array_equal(pair.pane_b.rgb_joint, blank)


def test_joint_flag_marks_equal_widths_true_and_differing_widths_false(tmp_path):
    model, batch = image2d_model()
    graph, out = run_compare(tmp_path, model, other_model(width=7), batch)
    node_1 = next(node for node in paired_nodes(graph) if node["name"] == "1")
    assert {node["name"]: node["joint"] for node in paired_nodes(graph)} == {"0": True, "1": False}
    assert all(not pane["rgb_joint"].any() for pane in pane_sections(out, node_1))


def test_compare_budgets_joint_rgb_whatever_bytes_per_position_is_passed(tmp_path):
    model, batch = image2d_model()
    single = CaptureOptions(layer_mb=8000 / MB, bytes_per_position=BYTES_PER_POSITION)
    graph, _ = run_compare(tmp_path, model, copy.deepcopy(model), batch, capture=single)
    joint = CaptureOptions(layer_mb=8000 / MB, bytes_per_position=COMPARE_BYTES_PER_POSITION)
    expected = {layer.node.name: layer.kept for layer in run_pipeline(model, batch, joint).layers}
    assert {node["name"]: node["kept"] for node in paired_nodes(graph)} == expected


def test_compare_layer_trimmed_by_layer_mb_writes_every_file_within_it(tmp_path):
    model, batch = wide_volume3d_model()
    capture = CaptureOptions(layer_mb=4096 / MB)
    graph, out = run_compare(tmp_path, model, copy.deepcopy(model), batch, capture=capture)
    node = paired_nodes(graph)[-1]
    sizes = [path.stat().st_size for path in (out / "layers").iterdir()]
    assert (node["kept"] < node["total"], max(sizes) <= 4096) == (True, True)
