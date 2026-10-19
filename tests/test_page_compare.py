"""
Test the comparison page in headless Chromium from multi file comparison bundles.

Purpose:
- Check two labelled panes, the CKA row, and the fit switch between aligned and joint colors.
- Check that a click in pane A recolors pane B from its own scores, and the shared camera.
- Check that both 3D panes draw a full height canvas of the same width and one legend per mode.
- Check that a pair without joint colors disables the joint option and shows a note.
- Check the note that names both dimensions when a 2D click meets a 3D layer.

Special Notes:
- Colors are read from the drawn pixels of each pane canvas, and the shared camera from
  screenshots of the 3D panes.
- Expected colors come from the layer files the page loads, decoded in Python.
"""

import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from layerviz.comparison import CompareOptions, compare
from tests.harness import browser, open_bundle, page, pane_sections, serve_dir  # noqa: F401
from tests.models import image2d_model, sparse_model

LABELS = ("base", "copy")
VIRIDIS_LOW = [68, 1, 84]
VIRIDIS_HIGH = [253, 231, 37]
NO_JOINT_NOTE = "Joint colors are off, this pair has no joint fit"
SIMILARITY_NOTE = "Showing PCA until a position is clicked"
CHANNEL_NOTE = "Showing PCA, channels are not compared"
DIMENSION_NOTE = "Showing PCA, clicked in 2D, this layer is 3D"
JOINT_LEGEND = "PCA RGB, joint fit"
# The smallest side views3d.js gives a point cloud canvas.
VIEW_3D_PX = 480
# At this width both panes get a whole pixel width, so equal drawings give equal screenshots.
WHOLE_PIXEL_VIEWPORT = {"width": 1281, "height": 720}
CANVAS = '.layerviz-pane[data-pane="{}"] canvas'
# Layer 1 of image2d_model is an 8 by 8 grid, and this cell is row 3, column 5.
GRID_SIDE = 8
KNOWN_CELL = (3, 5)
# Time for the render loop to draw a frame after a drag.
RENDER_WAIT_MS = 200


class SliceNet(nn.Module):
    """
    Run a Conv3d over a volume, then a Conv2d over its depth mean, giving a 3D and a 2D layer.

    """

    def __init__(self) -> None:
        """Create the volume and image convolutions."""
        super().__init__()
        self.volume = nn.Conv3d(2, 4, 3, padding=1)
        self.image = nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a [1, 2, D, H, W] volume to a [1, 4, H, W] image."""
        return self.image(self.volume(x).mean(dim=2))


def slice_model() -> tuple[nn.Module, torch.Tensor]:
    """Build SliceNet with a fixed seed and its [1, 2, 4, 8, 8] batch."""
    torch.manual_seed(3)
    return SliceNet().eval(), torch.randn(1, 2, 4, 8, 8)


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


def show(page, node_id, mode="pca"):
    """Click a graph node and wait until both of its panes are drawn in the given mode."""
    page.click(f'#graph-svg g.node[data-id="{node_id}"]')
    page.wait_for_selector(f'#view[data-node="{node_id}"][data-mode="{mode}"]')
    page.wait_for_selector('.layerviz-pane[data-pane="b"] canvas')


def canvas_rgb(page, pane: str) -> np.ndarray:
    """Return the drawn rgb of every cell of one pane's 2D canvas, flat in row major order."""
    script = """(selector) => {
      const canvas = document.querySelector(selector);
      return Array.from(canvas.getContext('2d').getImageData(0, 0, canvas.width,
        canvas.height).data);
    }"""
    rgba = np.array(page.evaluate(script, CANVAS.format(pane)), dtype=np.uint8)
    return rgba.reshape(-1, 4)[:, :3].reshape(-1)


def pane_image(page, pane: str) -> bytes:
    """Return the screenshot bytes of one pane's canvas, which change when its drawing does."""
    return page.locator(CANVAS.format(pane)).screenshot()


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


@pytest.fixture
def sparse_copy_out(tmp_path):
    """Write the bundle of sparse_model against a copy of itself."""
    return bundle(tmp_path, sparse_model, copy.deepcopy)


@pytest.fixture
def image_pair_out(tmp_path):
    """Write the bundle of image2d_model against a reseeded copy, so the two records differ."""
    return bundle(tmp_path, image2d_model, reseeded)


def test_compare_layer_shows_two_labelled_panes_in_aligned_colors_and_cka(image_out, page):
    node = node_named(image_out, "1")
    with serve_dir(image_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, node["id"])
        titles = page.locator(".layerviz-pane h3").all_inner_texts()
        cka = page.locator('#panel [data-field="cka"]').inner_text()
        colors = [canvas_rgb(page, "a"), canvas_rgb(page, "b")]
        joint_shown = page.locator("#mode option[value=joint]").get_attribute("hidden")
    assert titles == list(LABELS)
    assert cka == "1.00"
    for drawn, pane in zip(colors, pane_sections(image_out, node), strict=True):
        np.testing.assert_array_equal(drawn, pane["rgb"])
    assert joint_shown is None
    assert errors == []


def test_each_pane_of_a_differing_pair_draws_its_own_record_at_a_cell(image_pair_out, page):
    node = node_named(image_pair_out, "1")
    cell = KNOWN_CELL[0] * GRID_SIDE + KNOWN_CELL[1]
    panes = pane_sections(image_pair_out, node)
    expected = [pane["rgb"].reshape(-1, 3)[cell].tolist() for pane in panes]
    with serve_dir(image_pair_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, node["id"])
        drawn = [canvas_rgb(page, pane).reshape(-1, 3)[cell].tolist() for pane in ("a", "b")]
    assert expected[0] != expected[1]
    assert drawn == expected
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
        colors = [canvas_rgb(page, "a"), canvas_rgb(page, "b")]
        legends = page.locator(".layerviz-legend").all_inner_texts()
    for drawn, pane in zip(colors, pane_sections(image_out, node), strict=True):
        assert not np.array_equal(drawn, pane["rgb"])
    # The copy has the same scores, so both panes rank every position alike.
    np.testing.assert_array_equal(colors[0], colors[1])
    assert legends == ["Similarity percentile, low to high"] * 2
    assert errors == []


def test_fit_switch_draws_joint_colors_then_aligned_colors(image_out, page):
    node = node_named(image_out, "1")
    panes = pane_sections(image_out, node)
    with serve_dir(image_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, node["id"])
        page.select_option("#mode", "joint")
        page.wait_for_selector('#view[data-mode="joint"]')
        joint = [canvas_rgb(page, "a"), canvas_rgb(page, "b")]
        page.select_option("#mode", "pca")
        page.wait_for_selector('#view[data-mode="pca"]')
        aligned = [canvas_rgb(page, "a"), canvas_rgb(page, "b")]
    assert not np.array_equal(panes[0]["rgb"], panes[0]["rgb_joint"])
    for drawn, pane in zip(joint, panes, strict=True):
        np.testing.assert_array_equal(drawn, pane["rgb_joint"])
    for drawn, pane in zip(aligned, panes, strict=True):
        np.testing.assert_array_equal(drawn, pane["rgb"])
    assert errors == []


def test_layer_switch_after_a_click_keeps_similarity_colors_in_both_panes(image_out, page):
    first, second = node_named(image_out, "1"), node_named(image_out, "0")
    panes = pane_sections(image_out, second)
    with serve_dir(image_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, first["id"])
        page.locator('.layerviz-pane[data-pane="a"] canvas').click(position={"x": 5, "y": 5})
        page.wait_for_selector(f'#view[data-node="{first["id"]}"][data-mode="similarity"]')
        click = np.array([float(v) for v in page.locator("#click").inner_text().split(",")])
        show(page, second["id"], mode="similarity")
        colors = [canvas_rgb(page, pane).reshape(-1, 3) for pane in ("a", "b")]
        legends = page.locator(".layerviz-legend").all_inner_texts()
    for drawn, pane in zip(colors, panes, strict=True):
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
        colors = [canvas_rgb(page, "a"), canvas_rgb(page, "b")]
        status = page.locator("#status").inner_text()
    for drawn, pane in zip(colors, pane_sections(image_out, node), strict=True):
        np.testing.assert_array_equal(drawn, pane["rgb"])
    assert status == CHANNEL_NOTE
    assert errors == []


def test_click_in_3d_pane_a_ranks_each_pane_by_its_own_scores(sparse_out, page):
    node = node_named(sparse_out, "1")
    page.set_viewport_size(WHOLE_PIXEL_VIEWPORT)
    with serve_dir(sparse_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, node["id"])
        before = [pane_image(page, pane) for pane in ("a", "b")]
        page.locator(CANVAS.format("a")).click()
        page.wait_for_selector('#view[data-mode="similarity"]')
        after = [pane_image(page, pane) for pane in ("a", "b")]
    assert [after[i] != before[i] for i in (0, 1)] == [True, True]
    # Both panes hold the same positions under one camera, so equal images would mean
    # pane b drew the ranks of pane a.
    assert after[0] != after[1]
    assert errors == []


def test_3d_panes_draw_full_height_canvases_of_equal_width(sparse_out, page):
    node = node_named(sparse_out, "1")
    with serve_dir(sparse_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, node["id"])
        boxes = [page.locator(f'.layerviz-pane[data-pane="{pane}"] canvas').bounding_box()
                 for pane in ("a", "b")]
    assert [box["height"] >= VIEW_3D_PX for box in boxes] == [True, True]
    assert boxes[0]["width"] == pytest.approx(boxes[1]["width"], abs=1)
    assert errors == []


def test_rotating_pane_a_moves_the_camera_of_pane_b(sparse_copy_out, page):
    node = node_named(sparse_copy_out, "1")
    page.set_viewport_size(WHOLE_PIXEL_VIEWPORT)
    with serve_dir(sparse_copy_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, node["id"])
        before = pane_image(page, "b")
        drag(page, CANVAS.format("a"), 120)
        page.wait_for_timeout(RENDER_WAIT_MS)
        after = [pane_image(page, pane) for pane in ("a", "b")]
    assert after[1] != before
    # The copy draws the same points in the same colors, so equal images mean one camera.
    assert after[0] == after[1]
    assert errors == []


def test_mode_switch_in_3d_panes_sets_the_legend_of_both(sparse_out, page):
    node = node_named(sparse_out, "1")
    with serve_dir(sparse_out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, node["id"])
        page.select_option("#mode", "joint")
        page.wait_for_selector('#view[data-mode="joint"]')
        legends = page.locator(".layerviz-legend").all_inner_texts()
    assert legends == [JOINT_LEGEND] * 2
    assert errors == []


def test_click_in_a_2d_layer_then_a_3d_layer_names_both_dimensions(tmp_path, page):
    out = bundle(tmp_path, slice_model, copy.deepcopy)
    flat, volume = node_named(out, "image"), node_named(out, "volume")
    with serve_dir(out) as url:
        errors = open_bundle(page, url + "index.html")
        show(page, flat["id"])
        page.locator(CANVAS.format("a")).click(position={"x": 5, "y": 5})
        page.wait_for_selector(f'#view[data-node="{flat["id"]}"][data-mode="similarity"]')
        show(page, volume["id"])
        status = page.locator("#status").inner_text()
    assert status == DIMENSION_NOTE
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
