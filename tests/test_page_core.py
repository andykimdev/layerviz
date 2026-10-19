"""
Test the viewer page in headless Chromium from inline and multi file bundles.

Purpose:
- Check the single file page from file:// and the multi file page over http.
- Check the side panel, channel mode, and similarity kept across a layer switch.
- Check the PCA fallback note and that a stale channel paint never overwrites a later click.

Special Notes:
- serve_dir and open_bundle are shared helpers that later page tests import from here.
- Bundles use the real web folder with the vendored three.js, built without a model.
"""

import functools
import http.server
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from playwright.sync_api import sync_playwright

from layerviz.bundle import LayerPayload, write_bundle
from layerviz.reduce import PCAFit, Reduced
from layerviz.shapes.base import LayerData

GRID = (4, 4)
SCALE = 8.0
VIRIDIS_LOW = [68, 1, 84]
VIRIDIS_HIGH = [253, 231, 37]
CLICK_CELL = (1, 2)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files without logging every request to stderr."""

    def log_message(self, *args):
        """Drop the access log line."""


@contextmanager
def serve_dir(path):
    """
    Serve a folder over http on a free local port in a background thread.

    Args:
        path: Folder to serve.

    Returns:
        A context manager yielding the base URL, ending in a slash.
    """
    handler = functools.partial(_QuietHandler, directory=str(path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


def open_bundle(page, url):
    """
    Open a bundle page, wait for the graph to be drawn, and collect console errors.

    Args:
        page: A Playwright page.
        url: The index.html URL, file:// or http.

    Returns:
        A list that keeps receiving console error texts and uncaught page errors.
    """
    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url)
    page.wait_for_selector("#graph-svg g.node")
    return errors


def cell_positions(order):
    """Return the input coordinates of the grid cell centers, listed in the given cell order."""
    cells = [divmod(int(i), GRID[1]) for i in order]
    return np.array([[(r + 0.5) * SCALE, (c + 0.5) * SCALE] for r, c in cells], np.float32)


def make_payload(order, seed, channels=False):
    """Build an image2d payload on the shared grid, its positions in the given cell order."""
    rng = np.random.default_rng(seed)
    n = len(order)
    data = LayerData("image2d", cell_positions(order), rng.normal(size=(n, 5)).astype(np.float32),
                     GRID, (SCALE, SCALE), None)
    reduced = Reduced(
        rgb=rng.integers(0, 256, size=(n, 3), dtype=np.uint8),
        scores=rng.normal(size=(n, 3)).astype(np.float16),
        explained=np.array([0.5, 0.3, 0.1]), retained=0.9,
        fit=PCAFit(np.eye(3), np.ones(3) / 3, np.zeros(5), np.ones(5)),
    )
    chans = (rng.integers(0, 256, size=(n, 2), dtype=np.uint8), [4, 1]) if channels else None
    return LayerPayload(data=data, reduced=reduced, channels=chans)


def make_node(index, name, parent=None, leaf=True, **fields):
    """Build one graph node with the fields the page reads, placed at x 0 and y index."""
    node = {
        "id": f"L{index:04d}", "name": name, "type": "Conv2d", "parent": parent, "index": index,
        "leaf": leaf, "shape": [1, 5, 4, 4], "kind": "image2d", "explained": [0.5, 0.3, 0.1],
        "retained": 0.9, "shares": None, "skipped": None, "cka": None, "x": 0.0,
        "y": float(index),
    }
    node.update(fields)
    return node


def make_graph():
    """Build a graph of an input, two conv layers in box enc, and a skipped head layer."""
    nodes = [
        make_node(0, "input", type="Input", shape=[1, 3, 32, 32]),
        make_node(1, "enc.conv1", parent="enc", total=20),
        make_node(2, "enc.conv2", parent="enc", explained=[0.4, 0.2, 0.1], retained=0.7),
        make_node(3, "head", type="Linear", kind=None, skipped="output is a tuple"),
        make_node(4, "enc", leaf=False, type="Sequential", x=0.0, y=1.5),
    ]
    meta = {"model": "Toy", "step": None, "sample_index": 0, "label": None, "labels": None,
            "mode": "single", "version": "test", "unpaired": None}
    boxes = [{"name": "enc", "x0": -0.3, "y0": 0.7, "x1": 0.3, "y1": 2.3, "depth": 0}]
    edges = [["L0000", "L0001"], ["L0001", "L0002"], ["L0002", "L0003"]]
    return {"meta": meta, "input": "L0000", "nodes": nodes, "boxes": boxes, "edges": edges}


PAYLOADS = {
    "L0001": [make_payload(range(16), seed=1, channels=True)],
    "L0002": [make_payload(range(15, -1, -1), seed=2)],
    "L0004": [make_payload(range(16), seed=4)],
}


def build(out, inline_mb):
    """Write the test bundle with the real web folder and return its index.html path."""
    return write_bundle(out, make_graph(), PAYLOADS, inline_mb=inline_mb)


@pytest.fixture(scope="module")
def browser():
    """Launch one headless Chromium for every test in this module."""
    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch()
        yield chromium
        chromium.close()


@pytest.fixture
def page(browser):
    """Open a fresh page that closes after the test."""
    page = browser.new_page()
    yield page
    page.close()


def show(page, node_id, mode="pca"):
    """Click a graph node and wait until its layer is drawn in the given color mode."""
    page.click(f'#graph-svg g.node[data-id="{node_id}"]')
    page.wait_for_selector(f'#view[data-node="{node_id}"][data-mode="{mode}"] canvas')


def pixel(page, row, col):
    """Return the [r, g, b] of one grid cell of the layer canvas."""
    script = """([row, col]) => {
      const canvas = document.querySelector('#view canvas');
      const data = canvas.getContext('2d').getImageData(col, row, 1, 1).data;
      return [data[0], data[1], data[2]];
    }"""
    return page.evaluate(script, [row, col])


def click_cell(page, row, col):
    """Click the center of one grid cell on the layer canvas."""
    position = {"x": (col + 0.5) * SCALE, "y": (row + 0.5) * SCALE}
    page.locator("#view canvas").click(position=position)


def cosine_extremes(payload, row):
    """Return the cells of the most and least cosine similar positions to one position."""
    scores = payload.reduced.scores.astype(np.float64)
    unit = scores / np.linalg.norm(scores, axis=1, keepdims=True)
    similarity = unit @ unit[row]
    cells = (payload.data.positions // SCALE).astype(int).tolist()
    return tuple(cells[similarity.argmax()]), tuple(cells[similarity.argmin()])


def test_inline_bundle_renders_from_file_url_with_real_vendor_modules(tmp_path, page):
    index = build(tmp_path / "out", inline_mb=16)
    html = index.read_text(encoding="utf-8")
    requests = []
    page.on("request", lambda request: requests.append(request.url))
    errors = open_bundle(page, index.as_uri())
    assert sorted(p.name for p in index.parent.iterdir()) == ["index.html"]
    assert 'from "./' not in html
    assert page.locator("#graph-svg g.node").count() == 4
    assert page.locator("#graph-svg g.box").count() == 1
    assert page.locator("#graph-svg path.edge").count() == 3
    assert page.locator('#graph-svg g.box text').text_content() == "enc"
    assert page.locator('g.node[data-id="L0001"] text').text_content() == "enc.conv1"
    show(page, "L0001")
    rgb = PAYLOADS["L0001"][0].reduced.rgb
    assert pixel(page, 3, 1) == rgb[13].tolist()
    assert requests == [index.as_uri()]
    assert errors == []


def test_multi_file_bundle_fetches_a_layer_only_after_its_node_is_clicked(tmp_path, page):
    out = tmp_path / "out"
    build(out, inline_mb=0)
    assert (out / "graph.json").exists() and (out / "layers" / "L0001.bin").exists()
    with serve_dir(out) as url:
        requests = []
        page.on("request", lambda request: requests.append(request.url.removeprefix(url)))
        errors = open_bundle(page, url + "index.html")
        assert "graph.json" in requests
        assert not [r for r in requests if r.startswith("layers/")]
        assert page.locator("#graph-svg g.node").count() == 4
        show(page, "L0002")
        assert [r for r in requests if r.startswith("layers/")] == ["layers/L0002.bin"]
        rgb = PAYLOADS["L0002"][0].reduced.rgb
        assert pixel(page, 0, 0) == rgb[15].tolist()
    assert errors == []


def test_side_panel_shows_layer_stats(tmp_path, page):
    index = build(tmp_path / "out", inline_mb=16)
    errors = open_bundle(page, index.as_uri())
    show(page, "L0001")
    panel = page.locator("#panel")
    assert panel.locator('[data-field="name"]').inner_text() == "enc.conv1"
    assert panel.locator('[data-field="shape"]').inner_text() == "1 x 5 x 4 x 4"
    assert panel.locator('[data-field="kept"]').inner_text() == "16 of 20"
    assert panel.locator('[data-field="explained"]').inner_text() == "50.0%, 30.0%, 10.0%"
    assert panel.locator('[data-field="retained"]').inner_text() == "90.0%"
    assert panel.locator('[data-field="skipped"]').inner_text() == "none"
    assert errors == []


def test_side_panel_shows_the_skipped_reason_of_a_layer_without_data(tmp_path, page):
    index = build(tmp_path / "out", inline_mb=16)
    errors = open_bundle(page, index.as_uri())
    page.click('#graph-svg g.node[data-id="L0003"]')
    page.wait_for_selector('#panel [data-field="name"]:text-is("head")')
    assert page.locator('#panel [data-field="skipped"]').inner_text() == "output is a tuple"
    assert page.locator("#view canvas").count() == 0
    assert errors == []


def test_box_click_loads_the_layer_of_its_own_node(tmp_path, page):
    out = tmp_path / "out"
    build(out, inline_mb=0)
    with serve_dir(out) as url:
        requests = []
        page.on("request", lambda request: requests.append(request.url.removeprefix(url)))
        errors = open_bundle(page, url + "index.html")
        page.click("#graph-svg g.box rect", position={"x": 4, "y": 4})
        page.wait_for_selector('#view[data-node="L0004"][data-mode="pca"] canvas')
        assert page.locator('#panel [data-field="name"]').inner_text() == "enc"
        assert [r for r in requests if r.startswith("layers/")] == ["layers/L0004.bin"]
        assert pixel(page, 0, 0) == PAYLOADS["L0004"][0].reduced.rgb[0].tolist()
    assert errors == []


def test_channel_mode_loads_the_channel_file_only_when_chosen(tmp_path, page):
    out = tmp_path / "out"
    build(out, inline_mb=0)
    values = PAYLOADS["L0001"][0].channels[0]
    with serve_dir(out) as url:
        requests = []
        page.on("request", lambda request: requests.append(request.url.removeprefix(url)))
        errors = open_bundle(page, url + "index.html")
        show(page, "L0001")
        options = page.locator("#channel option")
        assert options.all_inner_texts() == ["channel 4", "channel 1"]
        assert "layers/L0001.ch.bin" not in requests
        page.select_option("#mode", "channel")
        page.wait_for_selector('#view[data-mode="channel"] canvas')
        assert "layers/L0001.ch.bin" in requests
        assert pixel(page, 2, 3) == [int(values[11, 0])] * 3
        page.select_option("#channel", "1")
        page.wait_for_function(
            "v => document.querySelector('#view').dataset.channel === v", arg="1")
        assert pixel(page, 2, 3) == [int(values[11, 1])] * 3
    assert errors == []


def test_click_then_layer_switch_recolors_at_the_same_input_coordinate(tmp_path, page):
    index = build(tmp_path / "out", inline_mb=16)
    errors = open_bundle(page, index.as_uri())
    show(page, "L0001")
    click_cell(page, *CLICK_CELL)
    page.wait_for_selector('#view[data-node="L0001"][data-mode="similarity"] canvas')
    coordinate = page.locator("#click").inner_text()
    assert coordinate == "12, 20"
    assert pixel(page, *CLICK_CELL) == VIRIDIS_HIGH
    show(page, "L0002", mode="similarity")
    assert page.locator("#click").inner_text() == coordinate
    row = 15 - (CLICK_CELL[0] * GRID[1] + CLICK_CELL[1])
    high, low = cosine_extremes(PAYLOADS["L0002"][0], row)
    assert high == CLICK_CELL
    assert pixel(page, *high) == VIRIDIS_HIGH
    assert pixel(page, *low) == VIRIDIS_LOW
    assert page.locator(".layerviz-legend").inner_text() == "Similarity percentile, low to high"
    assert errors == []


def test_similarity_without_a_click_draws_pca_with_a_note_until_a_click(tmp_path, page):
    index = build(tmp_path / "out", inline_mb=16)
    errors = open_bundle(page, index.as_uri())
    show(page, "L0001")
    page.select_option("#mode", "similarity")
    page.wait_for_selector('#status:text-is("Showing PCA until a position is clicked")')
    assert page.locator("#view").get_attribute("data-mode") == "pca"
    assert page.input_value("#mode") == "similarity"
    click_cell(page, *CLICK_CELL)
    page.wait_for_selector('#view[data-mode="similarity"] canvas')
    assert page.locator("#status").inner_text() == ""
    assert errors == []


def test_a_slow_channel_load_does_not_overwrite_a_later_click(tmp_path, page):
    out = tmp_path / "out"
    build(out, inline_mb=0)
    held = []
    with serve_dir(out) as url:
        page.route("**/layers/L0001.ch.bin", lambda route: held.append(route))
        errors = open_bundle(page, url + "index.html")
        show(page, "L0001")
        with page.expect_request("**/layers/L0001.ch.bin"):
            page.select_option("#mode", "channel")
        click_cell(page, *CLICK_CELL)
        page.wait_for_selector('#view[data-mode="similarity"] canvas')
        assert len(held) == 1
        held[0].continue_()
        # Awaiting the same cached promise, then one task, lets the held paint finish first.
        page.evaluate("""async (path) => {
          const { loadLayer } = await import("./app.js");
          await loadLayer(path);
          await new Promise((resolve) => setTimeout(resolve, 0));
        }""", "layers/L0001.ch.bin")
        assert page.locator("#view").get_attribute("data-mode") == "similarity"
        assert page.input_value("#mode") == "similarity"
        assert pixel(page, *CLICK_CELL) == VIRIDIS_HIGH
    assert errors == []
