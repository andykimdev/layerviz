"""
Test that every view kind renders PCA, similarity after a click, and a picked channel.

Purpose:
- Capture one multi file bundle per tiny model shape and open each over http (A19, A20).
- Check each view kind draws a 2D or WebGL canvas, recolors after a click, and loads channels late.
- Check image2d and sparse3d keep the clicked input coordinate across a layer switch (A21).

Special Notes:
- Bundles use inline_mb=0, so layer and channel files are fetched only when needed.
- Recoloring is read from the canvas pixels, since the page exposes no color debug hook.
- Chromium runs headless with software WebGL, since the test machines have no GPU.
"""

import pytest
from playwright.sync_api import sync_playwright

from layerviz import capture
from tests.models import ALL_SHAPES
from tests.test_page_core import open_bundle, serve_dir

WEBGL_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
WEBGL_KINDS = {"sparse3d", "points"}
SIMILARITY_LEGEND = "Similarity percentile, low to high"
VIRIDIS_HIGH = [253, 231, 37]
NO_CLICK = "none"
CORNER_PX = 1


@pytest.fixture(scope="session")
def browser():
    """Launch one headless Chromium with software WebGL for every test in this module."""
    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch(args=WEBGL_ARGS)
        yield chromium
        chromium.close()


@pytest.fixture(scope="session")
def bundles(tmp_path_factory):
    """Capture every tiny model into its own multi file bundle and serve them all over http."""
    root = tmp_path_factory.mktemp("bundles")
    for name, make in ALL_SHAPES.items():
        model, batch = make()
        capture(model, batch, root / name, inline_mb=0)
    with serve_dir(root) as url:
        yield url


@pytest.fixture
def page(browser):
    """Open a fresh page that closes after the test."""
    page = browser.new_page()
    yield page
    page.close()


def graph_of(page):
    """Return the parsed graph.json of the page's bundle, fetched in the page."""
    return page.evaluate("fetch('graph.json').then((response) => response.json())")


def layer_nodes(graph):
    """Return the leaf nodes that stored a layer file, in graph order."""
    return [node for node in graph["nodes"] if node["leaf"] and node["file"]]


def track_requests(page):
    """Return a list that keeps receiving the bundle relative path of every request."""
    paths = []
    page.on("request", lambda request: paths.append(request.url.rsplit("/", 2)[-2:]))
    return paths


def requested(paths, file):
    """Count the requests that fetched a bundle file such as layers/L0002.bin."""
    return paths.count(file.split("/"))


def open_shape(page, bundles, shape):
    """
    Open a shape's bundle and return its console errors, request log, and graph.

    Args:
        page: A Playwright page.
        bundles: Base URL that serves every bundle folder.
        shape: Key of ALL_SHAPES.

    Returns:
        A tuple (errors, paths, graph).
    """
    paths = track_requests(page)
    errors = open_bundle(page, f"{bundles}{shape}/index.html")
    return errors, paths, graph_of(page)


def select_node(page, node, mode="pca"):
    """Click a graph node and wait until its layer is drawn in the given color mode."""
    page.click(f'#graph-svg g.node[data-id="{node["id"]}"]')
    page.wait_for_selector(f'#view[data-node="{node["id"]}"][data-mode="{mode}"] canvas')


def canvas_kind(page):
    """Return webgl when the first view canvas holds a WebGL context, else 2d."""
    script = """() => {
      const canvas = document.querySelector('#view canvas');
      return canvas.getContext('webgl2') || canvas.getContext('webgl') ? 'webgl' : '2d';
    }"""
    return page.evaluate(script)


def fingerprint(page):
    """Return the screenshot bytes of the first view canvas, which change when colors change."""
    return page.locator("#view canvas").first.screenshot()


def click_view_center(page):
    """Click the center of the first view canvas, which picks the position under it."""
    page.locator("#view canvas").first.click()
    page.wait_for_selector('#view[data-mode="similarity"] canvas')


def click_view_corner(page):
    """Click near the top left of the first view canvas, which picks another position."""
    page.locator("#view canvas").first.click(position={"x": CORNER_PX, "y": CORNER_PX})


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
    errors, paths, graph = open_shape(page, bundles, shape)
    nodes = layer_nodes(graph)
    last = nodes[-1]
    assert page.locator("#graph-svg g.node").count() == sum(n["leaf"] for n in graph["nodes"])
    assert requested(paths, last["file"]) == 0
    select_node(page, last)
    assert requested(paths, last["file"]) == 1
    assert canvas_kind(page) == ("webgl" if last["kind"] in WEBGL_KINDS else "2d")
    assert page.locator("#click").inner_text() == NO_CLICK
    pca = fingerprint(page)
    click_view_center(page)
    assert page.locator("#click").inner_text() != NO_CLICK
    assert page.locator(".layerviz-legend").inner_text() == SIMILARITY_LEGEND
    assert page.locator("#status").inner_text() == ""
    center = page.locator("#click").inner_text()
    assert fingerprint(page) != pca
    similarity = fingerprint(page)
    click_view_corner(page)
    page.wait_for_function("c => document.querySelector('#click').textContent !== c", arg=center)
    assert fingerprint(page) != similarity
    assert requested(paths, last["channels"]) == 0
    page.select_option("#mode", "channel")
    page.select_option("#channel", "0")
    page.wait_for_selector('#view[data-mode="channel"][data-channel="0"] canvas')
    assert requested(paths, last["channels"]) == 1
    assert page.locator("#status").inner_text() == ""
    assert errors == []


@pytest.mark.parametrize("shape", ["image2d", "sparse"])
def test_a_click_survives_a_switch_to_the_previous_layer(page, bundles, shape):
    errors, _, graph = open_shape(page, bundles, shape)
    previous, last = layer_nodes(graph)[-2:]
    select_node(page, previous)
    pca = fingerprint(page)
    select_node(page, last)
    click_view_center(page)
    coordinate = page.locator("#click").inner_text()
    select_node(page, previous, mode="similarity")
    assert page.locator("#click").inner_text() == coordinate
    assert page.locator(".layerviz-legend").inner_text() == SIMILARITY_LEGEND
    assert page.locator("#status").inner_text() == ""
    assert fingerprint(page) != pca
    if previous["kind"] not in WEBGL_KINDS:
        assert has_pixel(page, VIRIDIS_HIGH)
    assert errors == []
