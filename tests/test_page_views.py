"""
Test that every view kind renders, and that the 2D views draw known colors and pick positions.

Purpose:
- Capture one bundle per tiny model shape and check each view kind renders, recolors after a
  click, and loads channels late.
- Read the pixels, title, and canvas size of the known layers for the image, volume, tokens,
  and scatter views.

Special Notes:
- Bundles use inline_mb=0, so layer and channel files are fetched only when needed.
- The browser fixtures come from tests.harness and run Chromium with software WebGL.
- The 3D views are tested in test_page_views3d.py with the same known bundle.
"""

import pytest

from layerviz import capture
from tests.harness import browser, open_bundle, page, serve_dir  # noqa: F401
from tests.models import ALL_SHAPES
from tests.views_known import (  # noqa: F401
    LABELS, LEGENDS, NO_CLICK, SIMILARITY_LEGEND, VIRIDIS_HIGH, VOLUME_SLICE_RGB, cell_rgb,
    click_text, click_view_center, click_view_corner, known_bundle, legend_text, open_known,
    pixel, select_known, select_node)

WEBGL_KINDS = {"sparse3d", "points"}


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    """Capture every tiny model into its own multi file bundle and serve them over http."""
    root = tmp_path_factory.mktemp("bundles")
    for name, make in ALL_SHAPES.items():
        model, batch = make()
        capture(model, batch, root / name, inline_mb=0)
    with serve_dir(root) as url:
        yield url


def layer_nodes(page):
    """Return the graph of the page's bundle and its leaf nodes that stored a layer file."""
    graph = page.evaluate("fetch('graph.json').then((response) => response.json())")
    return graph, [node for node in graph["nodes"] if node["leaf"] and node["file"]]


def track_requests(page):
    """Return a list that keeps receiving the bundle relative path of every request."""
    paths = []
    page.on("request", lambda request: paths.append(request.url.rsplit("/", 2)[-2:]))
    return paths


def requested(paths, file):
    """Count the requests that fetched a bundle file such as layers/L0002.bin."""
    return paths.count(file.split("/"))


def canvas_kind(page):
    """Return webgl when the first view canvas holds a WebGL context, else 2d."""
    script = """() => {
      const canvas = document.querySelector('#view canvas');
      return canvas.getContext('webgl2') || canvas.getContext('webgl') ? 'webgl' : '2d';
    }"""
    return page.evaluate(script)


def has_pixel(page, color):
    """Return whether the first view canvas holds a pixel of exactly the given color."""
    script = """(color) => {
      const canvas = document.querySelector('#view canvas');
      const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      for (let i = 0; i < data.length; i += 4) {
        if (color.every((value, a) => data[i + a] === value)) return true;
      }
      return false;
    }"""
    return page.evaluate(script, color)


@pytest.mark.parametrize("shape", list(ALL_SHAPES))
def test_every_view_kind_renders_pca_similarity_and_a_picked_channel(page, bundles, shape):
    paths = track_requests(page)
    errors = open_bundle(page, f"{bundles}{shape}/index.html")
    graph, nodes = layer_nodes(page)
    last = nodes[-1]
    assert page.locator("#graph-svg g.node").count() == sum(n["leaf"] for n in graph["nodes"])
    assert requested(paths, last["file"]) == 0
    select_node(page, last["id"])
    assert requested(paths, last["file"]) == 1
    assert canvas_kind(page) == ("webgl" if last["kind"] in WEBGL_KINDS else "2d")
    assert click_text(page) == NO_CLICK
    assert legend_text(page) == LEGENDS["pca"]
    click_view_center(page)
    center = click_text(page)
    assert center != NO_CLICK
    assert legend_text(page) == SIMILARITY_LEGEND
    assert page.locator("#status").inner_text() == ""
    click_view_corner(page)
    page.wait_for_function("c => document.querySelector('#click').textContent !== c", arg=center)
    assert requested(paths, last["channels"]) == 0
    page.select_option("#mode", "channel")
    page.select_option("#channel", "0")
    page.wait_for_selector('#view[data-mode="channel"][data-channel="0"] canvas')
    assert requested(paths, last["channels"]) == 1
    assert page.locator("#status").inner_text() == ""
    assert errors == []


def click_then_switch_to_previous_layer(page, bundles, shape):
    """Click the last layer's center, then switch to the layer before it in similarity mode.

    Returns the page errors, the previous layer's graph node, and the clicked coordinate text.
    """
    errors = open_bundle(page, f"{bundles}{shape}/index.html")
    previous, last = layer_nodes(page)[1][-2:]
    select_node(page, previous["id"])
    select_node(page, last["id"])
    click_view_center(page)
    coordinate = click_text(page)
    select_node(page, previous["id"], mode="similarity")
    return errors, previous, coordinate


@pytest.mark.parametrize("shape", ["image2d", "sparse"])
def test_a_click_survives_a_switch_to_the_previous_layer(page, bundles, shape):
    errors, _, coordinate = click_then_switch_to_previous_layer(page, bundles, shape)
    assert click_text(page) == coordinate
    assert legend_text(page) == SIMILARITY_LEGEND
    assert page.locator("#status").inner_text() == ""
    assert errors == []


@pytest.mark.parametrize("shape", ["image2d"])
def test_a_switch_to_a_previous_2d_layer_draws_the_clicked_cell_in_viridis_high(
    page, bundles, shape
):
    _, previous, _ = click_then_switch_to_previous_layer(page, bundles, shape)
    assert previous["kind"] not in WEBGL_KINDS
    assert has_pixel(page, VIRIDIS_HIGH)


def test_an_image_cell_draws_the_rgb_of_its_record(page, known_bundle):
    open_known(page, known_bundle)
    select_known(page, "image")
    drawn = [pixel(page, r, c) for r in range(4) for c in range(4)]
    assert drawn == [cell_rgb(i) for i in range(16)]


def test_the_clicked_image_cell_draws_the_viridis_high_color(page, known_bundle):
    open_known(page, known_bundle)
    select_known(page, "image")
    page.locator("#view canvas").click(position={"x": 2.5 * 8, "y": 1.5 * 8})
    page.wait_for_selector('#view[data-mode="similarity"] canvas')
    assert click_text(page) == "12, 20"
    assert pixel(page, 1, 2) == VIRIDIS_HIGH
    assert pixel(page, 0, 0) != VIRIDIS_HIGH


def test_the_default_volume_slice_is_the_middle_slice(page, known_bundle):
    open_known(page, known_bundle)
    select_known(page, "volume")
    middle = VOLUME_SLICE_RGB[1]
    assert [pixel(page, r, c, canvas=0) for r in (0, 2) for c in (0, 2)] == [middle] * 4
    assert [pixel(page, z, 0, canvas=1) for z in range(3)] == VOLUME_SLICE_RGB
    sliders = page.locator('#view input[type="range"]')
    assert sliders.evaluate_all("s => s.map(i => i.value)") == ["1", "1", "1"]


def test_the_slice_slider_draws_another_slice(page, known_bundle):
    open_known(page, known_bundle)
    select_known(page, "volume")
    page.locator('#view input[type="range"]').first.fill("2")
    corners = [pixel(page, r, c) for r in (0, 2) for c in (0, 2)]
    assert corners == [VOLUME_SLICE_RGB[2]] * 4


def test_an_8_by_16_image_layer_draws_a_canvas_of_that_aspect(page, known_bundle):
    open_known(page, known_bundle)
    select_known(page, "tall")
    size = page.locator("#view canvas").evaluate("c => [c.width, c.height]")
    box = page.locator("#view canvas").bounding_box()
    assert size == [16, 8]
    assert box["width"] / box["height"] == 2


def test_a_hovered_token_sets_the_canvas_title_to_its_label(page, known_bundle):
    open_known(page, known_bundle)
    select_known(page, "tokens")
    canvas = page.locator("#view canvas")
    canvas.hover(position={"x": 1.5 * 24, "y": 1.5 * 24})
    assert canvas.get_attribute("title") == LABELS[5]
    canvas.hover(position={"x": 3.5 * 24, "y": 2.5 * 24})
    assert canvas.get_attribute("title") == LABELS[11]


@pytest.mark.parametrize(
    ("right", "bottom", "expected"),
    [(False, False, "2"), (True, False, "3"), (False, True, "0"), (True, True, "1")],
)
def test_a_corner_click_in_the_scatter_picks_the_position_nearest_that_corner(
        page, known_bundle, right, bottom, expected):
    open_known(page, known_bundle)
    select_known(page, "scatter")
    click_view_corner(page, right, bottom)
    assert click_text(page) == expected
