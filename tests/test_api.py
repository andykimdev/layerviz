"""
Test that capture writes graph bundles and schedules snapshots into step folders.

Purpose:

- Check the snapshot helpers: due steps, step folder names, the step list, and the index page.
- Check build_graph ids, edges, shares, and placement, and the files capture writes.

Special Notes:

- Layer files are decoded with the readers in tests/harness.py and tests/models.py.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import layerviz
from layerviz.api import build_graph, capture
from layerviz.budget import MB
from layerviz.pipeline import CaptureOptions, run_pipeline
from layerviz.snapshots import due, is_main_process, list_steps, step_dir, write_index
from tests.harness import read_graph, read_positions
from tests.models import as_eval_pair, image2d_model, make_web

SMALL_LAYER_MB = 0.004


def same_grid_model():
    """Build a conv, relu, strided conv model whose first two layers share a 16 by 16 grid."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 6, 3, stride=2, padding=1)
    )
    return as_eval_pair(model, torch.randn(1, 3, 16, 16))


@pytest.fixture
def web(tmp_path) -> Path:
    """Write a minimal web folder with a page template and an entry module."""
    return make_web(tmp_path)


def count_forwards(model: nn.Module) -> list:
    """Attach a forward pre hook to model and return the list it appends to on each call."""
    calls: list = []
    model.register_forward_pre_hook(lambda module, args: calls.append(1))
    return calls


@pytest.mark.parametrize(
    ("step", "every", "expected"), [(0, 2, True), (4, 2, True), (3, 2, False), (5, 1, True)]
)
def test_due_is_true_only_on_multiples_of_every(step, every, expected):
    assert due(step, every) is expected


@pytest.mark.parametrize("every", [0, -1])
def test_due_rejects_an_interval_below_one(every):
    with pytest.raises(ValueError, match=f"every={every}"):
        due(4, every)


def test_step_dir_pads_the_step_to_six_digits(tmp_path):
    assert step_dir(tmp_path, 42) == tmp_path / "step_000042"


def test_list_steps_returns_only_step_folders_in_order(tmp_path):
    for name in ["step_000010", "step_000002", "other"]:
        (tmp_path / name).mkdir()
    (tmp_path / "step_000003").write_text("not a folder")

    assert list_steps(tmp_path) == [tmp_path / "step_000002", tmp_path / "step_000010"]


def test_list_steps_orders_by_step_number_past_six_digits(tmp_path):
    for step in (1_000_000, 200_000):
        step_dir(tmp_path, step).mkdir()

    assert list_steps(tmp_path) == [tmp_path / "step_200000", tmp_path / "step_1000000"]


def test_list_steps_of_a_missing_folder_is_empty(tmp_path):
    assert list_steps(tmp_path / "missing") == []


def test_write_index_links_every_step_index_page(tmp_path):
    for step in (2, 0):
        step_dir(tmp_path, step).mkdir()

    page = write_index(tmp_path).read_text(encoding="utf-8")

    assert page.index('href="step_000000/index.html"') < page.index(
        'href="step_000002/index.html"'
    )


def test_is_main_process_is_false_only_for_a_nonzero_rank(monkeypatch):
    assert is_main_process() is True
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    assert is_main_process() is False


def test_build_graph_names_nodes_by_execution_index_and_projects_edges():
    model, batch = image2d_model()
    result = run_pipeline(model, batch, CaptureOptions())

    graph = build_graph(result, step=7, label="epoch 1")

    assert graph["meta"] == {
        "model": "Sequential", "step": 7, "sample_index": 0, "label": "epoch 1",
        "labels": None, "mode": "single", "version": layerviz.__version__, "unpaired": None,
    }
    assert graph["input"] == "L0000"
    assert [(n["id"], n["name"], n["parent"], n["leaf"]) for n in graph["nodes"]] == [
        ("L0000", "input", None, True),
        ("L0001", "", None, False),
        ("L0002", "0", "", True),
        ("L0003", "1", "", True),
    ]
    assert graph["edges"] == [["L0000", "L0002"], ["L0002", "L0003"]]


def test_build_graph_starts_container_edges_at_the_last_leaf_inside():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU()), nn.Conv2d(4, 2, 1))
    result = run_pipeline(model.eval(), torch.randn(1, 3, 4, 4), CaptureOptions())

    graph = build_graph(result)

    names = {n["id"]: n["name"] for n in graph["nodes"]}
    assert [[names[a], names[b]] for a, b in graph["edges"]] == [
        ["input", "0.0"], ["0.0", "0.1"], ["0.1", "1"]
    ]
    assert [box["name"] for box in graph["boxes"]] == ["", "0"]


def test_build_graph_fills_layer_fields_and_shares():
    model, batch = image2d_model()
    result = run_pipeline(model, batch, CaptureOptions(layer_mb=SMALL_LAYER_MB))

    nodes = {n["id"]: n for n in build_graph(result)["nodes"]}

    layer = result.layers[0]
    assert nodes["L0002"]["kind"] == "image2d"
    assert nodes["L0002"]["shape"] == [1, 4, 16, 16]
    assert (nodes["L0002"]["kept"], nodes["L0002"]["total"]) == (layer.kept, 256)
    assert layer.kept < 256
    assert nodes["L0002"]["explained"] == pytest.approx(list(layer.reduced.explained[:3]))
    assert nodes["L0002"]["retained"] == pytest.approx(layer.reduced.retained)
    assert nodes["L0001"]["shares"] == "L0003"
    assert nodes["L0000"]["kind"] == "image2d"
    assert nodes["L0000"]["kept"] is None


def test_build_graph_places_leaves_and_boxes_containers():
    model, batch = image2d_model()
    graph = build_graph(run_pipeline(model, batch, CaptureOptions()))

    nodes = {n["id"]: n for n in graph["nodes"]}
    ys = [nodes[i]["y"] for i in ("L0000", "L0002", "L0003")]
    assert len(set(ys)) == 3
    assert [box["name"] for box in graph["boxes"]] == [""]
    box = graph["boxes"][0]
    assert box["x0"] <= nodes["L0001"]["x"] <= box["x1"]
    assert box["y0"] <= nodes["L0001"]["y"] <= box["y1"]


def test_capture_writes_graph_with_kept_and_total_per_layer(tmp_path, web):
    out = tmp_path / "out"
    model, batch = image2d_model()

    index = capture(model, batch, out, web_dir=web, inline_mb=0)

    graph = read_graph(out)
    files = {n["name"]: (n["file"], n["kept"], n["total"]) for n in graph["nodes"] if n["file"]}
    assert index == out / "index.html" and index.exists()
    assert files == {
        "0": ("layers/L0002.bin", 256, 256),
        "1": ("layers/L0003.bin", 64, 64),
    }


def test_capture_keeps_every_layer_file_under_layer_mb(tmp_path, web):
    out = tmp_path / "out"
    model, batch = same_grid_model()

    capture(model, batch, out, web_dir=web, inline_mb=0, layer_mb=SMALL_LAYER_MB)

    sizes = [path.stat().st_size for path in (out / "layers").iterdir()]
    assert len(sizes) == 6
    assert max(sizes) <= SMALL_LAYER_MB * MB
    kept = [n["kept"] for n in read_graph(out)["nodes"] if n["file"]]
    assert kept == [kept[0]] * 3 and kept[0] < 64


def test_capture_trims_layers_so_the_bundle_fits_total_mb(tmp_path, web):
    out = tmp_path / "out"
    model, batch = same_grid_model()
    total_mb = 0.01

    capture(model, batch, out, web_dir=web, inline_mb=0, total_mb=total_mb)

    sizes = [path.stat().st_size for path in (out / "layers").iterdir()]
    assert len(sizes) == 6
    assert sum(sizes) <= total_mb * MB
    nodes = [n for n in read_graph(out)["nodes"] if n["file"]]
    assert all(n["kept"] < n["total"] for n in nodes)


def test_capture_warns_when_one_position_per_layer_passes_total_mb(tmp_path, web):
    model, batch = same_grid_model()

    budget_warning = pytest.warns(UserWarning, match="1 position per layer")
    with budget_warning, pytest.warns(UserWarning, match="over total_mb"):
        capture(model, batch, tmp_path / "out", web_dir=web, inline_mb=0, total_mb=1e-6)


def test_capture_keeps_the_same_positions_for_layers_on_one_grid(tmp_path, web):
    out = tmp_path / "out"
    model, batch = same_grid_model()

    capture(model, batch, out, web_dir=web, inline_mb=0, layer_mb=SMALL_LAYER_MB)

    conv = read_positions((out / "layers" / "L0002.bin").read_bytes())
    relu = read_positions((out / "layers" / "L0003.bin").read_bytes())
    strided = read_positions((out / "layers" / "L0004.bin").read_bytes())
    np.testing.assert_array_equal(conv, relu)
    assert not np.array_equal(conv, strided)


def test_capture_on_a_step_not_due_writes_nothing_and_never_runs_forward(tmp_path, web):
    out = tmp_path / "out"
    model, batch = image2d_model()
    calls = count_forwards(model)

    assert capture(model, batch, out, web_dir=web, step=3, every=2) is None

    assert calls == []
    assert not out.exists()


def test_capture_on_a_due_step_writes_its_folder_and_lists_it(tmp_path, web):
    out = tmp_path / "out"
    model, batch = image2d_model()

    index = capture(model, batch, out, web_dir=web, inline_mb=0, step=4, every=2)

    assert index == out / "step_000004" / "index.html" and index.exists()
    assert read_graph(out / "step_000004")["meta"]["step"] == 4
    page = (out / "index.html").read_text(encoding="utf-8")
    assert 'href="step_000004/index.html"' in page


def test_capture_on_a_nonzero_rank_writes_nothing(tmp_path, web, monkeypatch):
    out = tmp_path / "out"
    model, batch = image2d_model()
    calls = count_forwards(model)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)

    assert capture(model, batch, out, web_dir=web) is None

    assert calls == []
    assert not out.exists()


def test_capture_restores_the_training_and_grad_modes(tmp_path, web):
    out = tmp_path / "out"
    model, batch = image2d_model()
    model.train()

    capture(model, batch, out, web_dir=web)

    assert model.training is True
    assert torch.is_grad_enabled() is True
    assert all(p.requires_grad for p in model.parameters())


def test_step_dir_rejects_a_negative_step(tmp_path):
    with pytest.raises(ValueError, match="step=-5"):
        step_dir(tmp_path, -5)


def test_capture_with_every_but_no_step_raises_and_never_runs_forward(tmp_path, web):
    out = tmp_path / "out"
    model, batch = image2d_model()
    calls = count_forwards(model)

    with pytest.raises(ValueError, match="every=5"):
        capture(model, batch, out, web_dir=web, every=5)

    assert calls == []
    assert not out.exists()


def test_capture_with_a_negative_step_raises_and_writes_no_folder(tmp_path, web):
    out = tmp_path / "out"
    model, batch = image2d_model()

    with pytest.raises(ValueError, match="step=-5"):
        capture(model, batch, out, web_dir=web, step=-5)

    assert not (out / "step_-00005").exists()
    assert list_steps(out) == []


def test_capture_accepts_the_documented_step_and_every_call(tmp_path, web):
    out = tmp_path / "out"
    model, batch = image2d_model()
    step = 1000

    index = capture(model, batch, out, step=step, every=500, web_dir=web, inline_mb=0)
    skipped = capture(model, batch, out, step=step + 250, every=500, web_dir=web)

    assert skipped is None
    assert index == out / "step_001000" / "index.html" and index.exists()
    assert read_graph(out / "step_001000")["meta"]["step"] == step
    assert list_steps(out) == [out / "step_001000"]


def test_capture_rejects_an_unknown_option_by_name(tmp_path, web):
    out = tmp_path / "out"
    model, batch = image2d_model()
    calls = count_forwards(model)

    with pytest.raises(TypeError, match="sample_idx"):
        capture(model, batch, out, web_dir=web, sample_idx=1)

    assert calls == []
    assert not out.exists()
