"""
Test the viewer page in headless Chromium from inline and multi file bundles.

Purpose:
- Check the single file page from file:// and the multi file page over http.
- Check the side panel, channel mode, the PCA fallback note, and the selected node.
- Check that stale loads and paints never draw or report over a later selection.
- Check that every load error names its file and that a split bundle on disk says to serve it.

Special Notes:
- The browser fixtures come from tests.harness. Other page tests still import them from here.
- Bundles use the real web folder with the vendored three.js, built without a model.
"""

import json
import re

import numpy as np

from layerviz.bundle import LayerPayload, write_bundle
from layerviz.reduce import PCAFit, Reduced
from layerviz.shapes.base import LayerData
from tests.harness import (  # noqa: F401
    browser, open_bundle, page, release, serve_dir)

GRID = (4, 4)
SCALE = 8.0
VIRIDIS_HIGH = [253, 231, 37]
CLICK_CELL = (1, 2)


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


def build(out, inline_mb, graph=None, payloads=None):
    """Write a bundle, by default the test graph, and return its index.html path."""
    return write_bundle(out, graph or make_graph(), payloads or PAYLOADS, inline_mb=inline_mb)


def build_points(out):
    """Write an inline bundle whose one layer is a 16 position points cloud with channels."""
    payload = make_payload(range(16), seed=5, channels=True)
    cloud = np.random.default_rng(5).normal(size=(16, 3)).astype(np.float32)
    payload.data.kind, payload.data.positions, payload.data.grid, payload.data.scale = (
        "points", cloud, None, None)
    graph = make_graph()
    graph["nodes"] = [graph["nodes"][0], make_node(1, "cloud", kind="points", shape=[16, 5])]
    graph.update(boxes=[], edges=[["L0000", "L0001"]])
    return build(out, 16, graph, {"L0001": [payload]})


def wait_for_status(page, text):
    """Wait until the status line holds the given text and return the whole line."""
    page.wait_for_selector(f"#status:has-text({json.dumps(text)})")
    return page.locator("#status").inner_text()


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


def test_inline_bundle_is_one_file_that_requests_nothing_else(tmp_path, page):
    index = build(tmp_path / "out", inline_mb=16)
    requests = []
    page.on("request", lambda request: requests.append(request.url))
    errors = open_bundle(page, index.as_uri())
    show(page, "L0001")
    assert sorted(p.name for p in index.parent.iterdir()) == ["index.html"]
    assert 'from "./' not in index.read_text(encoding="utf-8")
    assert requests == [index.as_uri()]
    assert errors == []


def test_graph_draws_leaves_boxes_and_edges_with_their_names(tmp_path, page):
    errors = open_bundle(page, build(tmp_path / "out", inline_mb=16).as_uri())
    assert page.locator("#graph-svg g.node").count() == 4
    assert page.locator("#graph-svg g.box").count() == 1
    assert page.locator("#graph-svg path.edge").count() == 3
    assert page.locator("#graph-svg g.box text").text_content() == "enc"
    assert page.locator('g.node[data-id="L0001"] text').text_content() == "enc.conv1"
    assert errors == []


def test_inline_layer_draws_its_pca_colors_from_a_file_url(tmp_path, page):
    errors = open_bundle(page, build(tmp_path / "out", inline_mb=16).as_uri())
    show(page, "L0001")
    assert pixel(page, 3, 1) == PAYLOADS["L0001"][0].reduced.rgb[13].tolist()
    assert errors == []


def test_a_module_name_with_markup_renders_as_literal_text(tmp_path, page):
    graph = make_graph()
    graph["nodes"][2]["name"] = "<b>x</b>"
    errors = open_bundle(page, build(tmp_path / "out", 16, graph).as_uri())
    assert page.locator('g.node[data-id="L0002"] text').text_content() == "<b>x</b>"
    show(page, "L0002")
    assert page.locator('#panel [data-field="name"]').inner_text() == "<b>x</b>"
    assert page.locator("#graph-svg b, #panel b").count() == 0
    assert errors == []


def test_only_the_clicked_node_carries_the_selected_class(tmp_path, page):
    errors = open_bundle(page, build(tmp_path / "out", inline_mb=16).as_uri())
    show(page, "L0001")
    show(page, "L0002")
    selected = page.locator("#graph-svg g.selected")
    assert selected.evaluate_all("els => els.map((el) => el.dataset.id)") == ["L0002"]
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
        release(page, held[0], "layers/L0001.ch.bin")
        assert page.locator("#view").get_attribute("data-mode") == "similarity"
        assert page.input_value("#mode") == "similarity"
        assert pixel(page, *CLICK_CELL) == VIRIDIS_HIGH
    assert errors == []


def test_selecting_a_layer_while_another_loads_draws_only_the_later_one(tmp_path, page):
    out = build(tmp_path / "out", inline_mb=0).parent
    held = []
    with serve_dir(out) as url:
        page.route("**/layers/L0001.bin", lambda route: held.append(route))
        errors = open_bundle(page, url + "index.html")
        with page.expect_request("**/layers/L0001.bin"):
            page.click('#graph-svg g.node[data-id="L0001"]')
        show(page, "L0002")
        release(page, held[0], "layers/L0001.bin")
        assert page.locator("#view").get_attribute("data-node") == "L0002"
        assert page.locator('#panel [data-field="name"]').inner_text() == "enc.conv2"
        assert pixel(page, 0, 0) == PAYLOADS["L0002"][0].reduced.rgb[15].tolist()
    assert errors == []


def test_a_stale_failed_load_leaves_the_later_layer_status_empty(tmp_path, page):
    out = build(tmp_path / "out", inline_mb=0).parent
    held = []
    with serve_dir(out) as url:
        page.route("**/layers/L0001.bin", lambda route: held.append(route))
        open_bundle(page, url + "index.html")
        with page.expect_request("**/layers/L0001.bin"):
            page.click('#graph-svg g.node[data-id="L0001"]')
        show(page, "L0002")
        release(page, held[0], "layers/L0001.bin", fail=True)
        assert page.locator("#status").inner_text() == ""
        assert page.locator("#view").get_attribute("data-node") == "L0002"

def test_a_stale_channel_failure_leaves_the_later_layer_status_alone(tmp_path, page):
    out = build(tmp_path / "out", inline_mb=0).parent
    held = []
    with serve_dir(out) as url:
        page.route("**/layers/L0001.ch.bin", lambda route: held.append(route))
        open_bundle(page, url + "index.html")
        show(page, "L0001")
        with page.expect_request("**/layers/L0001.ch.bin"):
            page.select_option("#mode", "channel")
        show(page, "L0002")
        release(page, held[0], "layers/L0001.ch.bin", fail=True)
        assert page.locator("#status").inner_text() == "Showing PCA, this layer has no channels"
        assert page.locator("#view").get_attribute("data-node") == "L0002"


def test_a_failed_first_load_names_the_file_and_loads_on_the_next_click(tmp_path, page):
    out = build(tmp_path / "out", inline_mb=0).parent
    attempts = []

    def fail_once(route):
        attempts.append(route)
        route.abort() if len(attempts) == 1 else route.continue_()

    with serve_dir(out) as url:
        page.route("**/layers/L0002.bin", fail_once)
        open_bundle(page, url + "index.html")
        page.click('#graph-svg g.node[data-id="L0002"]')
        status = wait_for_status(page, "layers/L0002.bin")
        assert status.startswith("layers/L0002.bin could not be fetched")
        show(page, "L0002")
        assert len(attempts) == 2
        assert page.locator("#status").inner_text() == ""
        assert pixel(page, 0, 0) == PAYLOADS["L0002"][0].reduced.rgb[15].tolist()


def test_invalid_graph_json_names_the_file_in_the_status(tmp_path, page):
    out = build(tmp_path / "out", inline_mb=0).parent
    (out / "graph.json").write_text("{not json")
    with serve_dir(out) as url:
        page.goto(url + "index.html")
        assert wait_for_status(page, "graph.json").startswith("graph.json is not valid JSON")


def test_invalid_inline_base64_names_the_layer_file_in_the_status(tmp_path, page):
    index = build(tmp_path / "out", inline_mb=16)
    html = index.read_text(encoding="utf-8")
    broken = re.sub(r'(data-path="layers/L0001.bin">)[^<]*', r"\1@@@@", html)
    assert broken != html
    index.write_text(broken, encoding="utf-8")
    open_bundle(page, index.as_uri())
    page.click('#graph-svg g.node[data-id="L0001"]')
    status = wait_for_status(page, "layers/L0001.bin")
    assert status.startswith("layers/L0001.bin: base64 text is not valid")


def test_a_split_bundle_opened_from_disk_says_to_serve_its_folder(tmp_path, page):
    index = build(tmp_path / "out", inline_mb=0)
    page.goto(index.as_uri())
    status = wait_for_status(page, "python -m layerviz serve")
    assert f'python -m layerviz serve "{index.parent}" and open' in status
    assert page.locator("#graph-svg g.node").count() == 0


def test_a_mode_switch_of_a_3d_layer_recolors_the_same_canvas_and_legend(tmp_path, page):
    errors = open_bundle(page, build_points(tmp_path / "out").as_uri())
    show(page, "L0001")
    page.evaluate("document.querySelector('#view canvas').dataset.mark = 'first'")
    page.select_option("#mode", "channel")
    page.wait_for_selector('#view[data-mode="channel"]')
    assert page.locator("#view canvas").first.get_attribute("data-mark") == "first"
    assert page.locator(".layerviz-legend").inner_text() == "Channel value, low to high"
    assert errors == []
