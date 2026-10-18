"""
Test the comparison page in headless Chromium from multi file comparison bundles.

Purpose:
- Check two labelled panes, the CKA row, and the fit switch between aligned and joint colors.
- Check that a click in pane A recolors pane B from its own scores, and the shared camera (A28).
- Check that a pair without joint colors disables the joint option and shows a note.

Special Notes:
- Colors and cameras are read through the window.layerviz debug hook that compare.js sets.
- Expected colors come from the layer files the page loads, decoded in Python.
"""

import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from layerviz.comparison import CompareOptions, compare
from tests.models import image2d_model, sparse_model
from tests.test_compare import read_sections
from tests.test_page_core import browser, open_bundle, page, serve_dir  # noqa: F401

LABELS = ("base", "copy")
VIRIDIS_LOW = [68, 1, 84]
VIRIDIS_HIGH = [253, 231, 37]
NO_JOINT_NOTE = "Joint colors are off, this pair has no joint fit"
SIMILARITY_NOTE = "Showing PCA until a position is clicked"
CHANNEL_NOTE = "Showing PCA, channels are not compared"


def reseeded(model):
    """Copy a model and redraw every parameter from a fixed seed other than the model's."""
    twin = copy.deepcopy(model)
    torch.manual_seed(1)
    with torch.no_grad():
        for parameter in twin.parameters():
            parameter.copy_(torch.randn_like(parameter))
    return twin


def wider_image2d(model):
    """Build an image2d model shaped like image2d_model whose last layer has 5 channels."""
    torch.manual_seed(2)
    return nn.Sequential(model[0], nn.Conv2d(4, 5, 3, stride=2, padding=1)).eval()


def bundle(tmp_path, builder, twin):
    """Write a comparison bundle of builder()'s model against twin(model) and return its folder."""
    model, batch = builder()
    out = tmp_path / "out"
    compare(model, twin(model), batch, out, CompareOptions(LABELS, inline_mb=0))
    return out


def node_named(out, name):
    """Return the graph node of one module name from the bundle's graph.json."""
    graph = json.loads((out / "graph.json").read_text())
    return next(node for node in graph["nodes"] if node["name"] == name)


def sections(out, node):
    """Return the decoded sections of pane a and pane b of one node."""
    return [read_sections(out / node["files"][pane])[1] for pane in ("a", "b")]


def show(page, node_id, mode="pca"):
    """Click a graph node and wait until both of its panes are drawn in the given mode."""
    page.click(f'#graph-svg g.node[data-id="{node_id}"]')
    page.wait_for_selector(f'#view[data-node="{node_id}"][data-mode="{mode}"]')
    page.wait_for_selector('.layerviz-pane[data-pane="b"] canvas')


def hook_colors(page, pane):
    """Return one pane's drawn colors from the debug hook as a uint8 array."""
    return np.array(page.evaluate(f"window.layerviz.colors({pane})"), dtype=np.uint8)


def drag(page, selector, dx):
    """Drag the mouse horizontally across the center of the element a selector names."""
    box = page.locator(selector).bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + dx, y, steps=8)
    page.mouse.up()


def extreme_positions(pane, click):
    """Return the rows most and least cosine similar, by this pane's scores, to the clicked row."""
    positions = pane["positions"].reshape(-1, len(click)).astype(np.float64)
    row = np.argmin(np.linalg.norm(positions - click, axis=1))
    scores = pane["scores"].astype(np.float64).reshape(len(positions), -1)
    unit = scores / np.linalg.norm(scores, axis=1, keepdims=True)
    similarity = unit @ unit[row]
    return similarity.argmax(), similarity.argmin()


@pytest.fixture
def image_out(tmp_path):
    """Write the bundle of image2d_model against a copy of itself."""
    return bundle(tmp_path, image2d_model, copy.deepcopy)


@pytest.fixture
def sparse_out(tmp_path):
    """Write the bundle of sparse_model against a reseeded copy."""
    return bundle(tmp_path, sparse_model, reseeded)


def test_compare_layer_shows_two_labelled_panes_in_aligned_colors_and_cka(image_out, page):
    node = node_named(image_out, "1")
    with serve_dir(image_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, node["id"])
        titles = page.locator(".layerviz-pane h3").all_inner_texts()
        cka = page.locator('#panel [data-field="cka"]').inner_text()
        colors = [hook_colors(page, 0), hook_colors(page, 1)]
        joint_shown = page.locator("#mode option[value=joint]").get_attribute("hidden")
    assert titles == list(LABELS)
    assert cka == "1.00"
    for drawn, pane in zip(colors, sections(image_out, node)):
        np.testing.assert_array_equal(drawn, pane["rgb"])
    assert joint_shown is None
    assert errors == []


@pytest.mark.parametrize("clicked", ["a", "b"])
def test_click_in_either_pane_recolors_both_by_similarity(image_out, page, clicked):
    node = node_named(image_out, "1")
    with serve_dir(image_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, node["id"])
        canvas = page.locator(f'.layerviz-pane[data-pane="{clicked}"] canvas')
        canvas.click(position={"x": 5, "y": 5})
        page.wait_for_selector(f'#view[data-node="{node["id"]}"][data-mode="similarity"]')
        colors = [hook_colors(page, 0), hook_colors(page, 1)]
        legends = page.locator(".layerviz-legend").all_inner_texts()
    for drawn, pane in zip(colors, sections(image_out, node)):
        assert not np.array_equal(drawn, pane["rgb"])
    # The copy has the same scores, so both panes rank every position alike.
    np.testing.assert_array_equal(colors[0], colors[1])
    assert legends == ["Similarity percentile, low to high"] * 2
    assert errors == []


def test_fit_switch_draws_joint_colors_then_aligned_colors(image_out, page):
    node = node_named(image_out, "1")
    panes = sections(image_out, node)
    with serve_dir(image_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, node["id"])
        page.select_option("#mode", "joint")
        page.wait_for_selector('#view[data-mode="joint"]')
        joint = [hook_colors(page, 0), hook_colors(page, 1)]
        page.select_option("#mode", "pca")
        page.wait_for_selector('#view[data-mode="pca"]')
        aligned = [hook_colors(page, 0), hook_colors(page, 1)]
    assert not np.array_equal(panes[0]["rgb"], panes[0]["rgb_joint"])
    for drawn, pane in zip(joint, panes):
        np.testing.assert_array_equal(drawn, pane["rgb_joint"])
    for drawn, pane in zip(aligned, panes):
        np.testing.assert_array_equal(drawn, pane["rgb"])
    assert errors == []


def test_layer_switch_after_a_click_keeps_similarity_colors_in_both_panes(image_out, page):
    first, second = node_named(image_out, "1"), node_named(image_out, "0")
    panes = sections(image_out, second)
    with serve_dir(image_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, first["id"])
        page.locator('.layerviz-pane[data-pane="a"] canvas').click(position={"x": 5, "y": 5})
        page.wait_for_selector(f'#view[data-node="{first["id"]}"][data-mode="similarity"]')
        click = np.array([float(v) for v in page.locator("#click").inner_text().split(",")])
        show(page, second["id"], mode="similarity")
        colors = [hook_colors(page, 0).reshape(-1, 3), hook_colors(page, 1).reshape(-1, 3)]
        legends = page.locator(".layerviz-legend").all_inner_texts()
    for drawn, pane in zip(colors, panes):
        high, low = extreme_positions(pane, click)
        assert drawn[high].tolist() == VIRIDIS_HIGH
        assert drawn[low].tolist() == VIRIDIS_LOW
    assert legends == ["Similarity percentile, low to high"] * 2
    assert errors == []


def test_channel_mode_draws_aligned_colors_in_both_panes_with_a_note(image_out, page):
    node = node_named(image_out, "1")
    with serve_dir(image_out) as url:
        errors = open_bundle(page, url + "index.html")
        page.select_option("#mode", "channel")
        show(page, node["id"])
        colors = [hook_colors(page, 0), hook_colors(page, 1)]
        status = page.locator("#status").inner_text()
    for drawn, pane in zip(colors, sections(image_out, node)):
        np.testing.assert_array_equal(drawn, pane["rgb"])
    assert status == CHANNEL_NOTE
    assert errors == []


def test_click_in_3d_pane_a_ranks_each_pane_by_its_own_scores(sparse_out, page):
    node = node_named(sparse_out, "1")
    panes = sections(sparse_out, node)
    with serve_dir(sparse_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, node["id"])
        page.locator('.layerviz-pane[data-pane="a"] canvas').click()
        page.wait_for_selector('#view[data-mode="similarity"]')
        click = np.array([float(v) for v in page.locator("#click").inner_text().split(",")])
        colors = [hook_colors(page, 0).reshape(-1, 3), hook_colors(page, 1).reshape(-1, 3)]
    for drawn, pane in zip(colors, panes):
        high, low = extreme_positions(pane, click)
        assert drawn[high].tolist() == VIRIDIS_HIGH
        assert drawn[low].tolist() == VIRIDIS_LOW
    assert not np.array_equal(colors[0], colors[1])
    assert errors == []


def test_rotating_pane_a_moves_the_camera_of_pane_b(sparse_out, page):
    node = node_named(sparse_out, "1")
    with serve_dir(sparse_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, node["id"])
        before = page.evaluate("window.layerviz.camera(1)")
        drag(page, '.layerviz-pane[data-pane="a"] canvas', 120)
        after = [page.evaluate(f"window.layerviz.camera({pane})") for pane in (0, 1)]
    assert before is not None
    assert not np.allclose(after[1], before)
    assert after[0] == after[1]
    assert errors == []


def test_pair_without_joint_colors_disables_the_joint_option_with_a_note(tmp_path, page):
    out = bundle(tmp_path, image2d_model, wider_image2d)
    node = node_named(out, "1")
    with serve_dir(out) as url:
        errors = open_bundle(page, url + "index.html")
        page.select_option("#mode", "joint")
        show(page, node["id"])
        disabled = page.eval_on_selector("#mode option[value=joint]", "el => el.disabled")
        mode = page.input_value("#mode")
        status = page.locator("#status").inner_text()
    assert node["joint"] is False
    assert disabled is True
    assert mode == "pca"
    assert status == NO_JOINT_NOTE
    assert errors == []


def test_pair_without_joint_colors_keeps_the_similarity_note(tmp_path, page):
    out = bundle(tmp_path, image2d_model, wider_image2d)
    node = node_named(out, "1")
    with serve_dir(out) as url:
        errors = open_bundle(page, url + "index.html")
        page.select_option("#mode", "similarity")
        show(page, node["id"])
        status = page.locator("#status").inner_text()
    assert node["joint"] is False
    assert status == SIMILARITY_NOTE
    assert errors == []
