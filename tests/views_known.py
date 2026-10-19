"""
Share a bundle of layers with known colors and the page helpers that read what it draws.

Purpose:
- Write one bundle whose layers hold exact rgb values, so page tests assert known pixels.
- Select layers, click canvas corners, and read pixels and text from the viewer page.

Special Notes:
- The bundle uses inline_mb=0, so the page fetches layer files over http.
- Each layer is named in LAYERS and its node id is the position in that list.
"""

import numpy as np
import pytest

from layerviz.bundle import LayerPayload, write_bundle
from layerviz.reduce import PCAFit, Reduced
from layerviz.shapes.base import LayerData
from tests.harness import open_bundle, serve_dir

LAYERS = ["input", "image", "volume", "tall", "cloud", "square", "slab", "scatter", "tokens"]
LABELS = ["the", "cat", "sat", "on", "a", "warm", "mat", "by", "the", "old", "red", "door"]
RED, GREEN, BLUE, WHITE = [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]
VOLUME_SLICE_RGB = [[200, 0, 0], [0, 200, 100], [0, 0, 200]]
VIRIDIS_HIGH = [253, 231, 37]
SIMILARITY_LEGEND = "Similarity percentile, low to high"
LEGENDS = {
    "pca": "PCA RGB of PC1 to PC3",
    "similarity": SIMILARITY_LEGEND,
    "channel": "Channel value, low to high",
}
NO_CLICK = "none"
CORNER_PX = 1


def cell_rgb(index):
    """Return the known color of image cell number index, in row major order."""
    return [15 * index, 250 - 15 * index, 5 * index + 10]


def grid_cells(shape, scale):
    """Return the center of every cell of a grid in input coordinates, in row major order."""
    cells = np.indices(shape).reshape(len(shape), -1).T
    return ((cells + 0.5) * scale).astype(np.float32)


def make_payload(kind, positions, rgb, **fields):
    """
    Build a payload whose rgb is exactly the given colors.

    Args:
        kind: Layer kind such as image2d or points.
        positions: One row per position in input coordinates.
        rgb: One [r, g, b] per position.
        **fields: Optional grid, scale, labels, scores, and channels (a bool).

    Returns:
        A LayerPayload whose scores come from a fixed seed unless scores is given.
    """
    positions = np.array(positions, np.float32)
    n = len(positions)
    rng = np.random.default_rng(n)
    data = LayerData(kind, positions, np.zeros((n, 2), np.float32), fields.get("grid"),
                     fields.get("scale"), fields.get("labels"))
    scores = fields.get("scores", rng.normal(size=(n, 3)))
    reduced = Reduced(
        rgb=np.array(rgb, np.uint8), scores=np.array(scores, np.float16),
        explained=np.array([0.5, 0.3, 0.1]), retained=0.9,
        fit=PCAFit(np.eye(3), np.ones(3) / 3, np.zeros(2), np.ones(2)),
    )
    kept = (rng.integers(0, 256, size=(n, 2), dtype=np.uint8), [0, 1])
    channels = kept if fields.get("channels") else None
    return LayerPayload(data=data, reduced=reduced, channels=channels)


def known_payloads():
    """Return the payload of every known layer by node id."""
    z_of_cell = grid_cells((3, 3, 3), 1).astype(int)[:, 0]
    square = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]]
    return {
        "L0001": make_payload("image2d", grid_cells((4, 4), 8), [cell_rgb(i) for i in range(16)],
                              grid=(4, 4), scale=(8, 8)),
        "L0002": make_payload("volume3d", grid_cells((3, 3, 3), 4),
                              [VOLUME_SLICE_RGB[z] for z in z_of_cell],
                              grid=(3, 3, 3), scale=(4, 4, 4)),
        "L0003": make_payload("image2d", grid_cells((8, 16), 4), [WHITE] * 128,
                              grid=(8, 16), scale=(4, 4)),
        "L0004": make_payload("sparse3d", [[0, 0, 100], [0, 0, 0]], [RED, GREEN], channels=True),
        "L0005": make_payload("points", square, [RED, GREEN, BLUE, WHITE], channels=True),
        "L0006": make_payload("sparse3d", [[0, 0, 0], [10, 10, 10]], [RED, GREEN]),
        "L0007": make_payload("vector", [[0], [1], [2], [3]], [RED, GREEN, BLUE, WHITE],
                              scores=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
        "L0008": make_payload("tokens", [[i] for i in range(12)], [WHITE] * 12, labels=LABELS),
    }


def known_graph():
    """Build a graph with an input node and one node per known layer, joined in a chain."""
    fields = {"type": "Layer", "parent": None, "leaf": True, "shape": [1, 2], "kind": "x",
              "explained": [0.5, 0.3, 0.1], "retained": 0.9, "shares": None, "skipped": None,
              "cka": None, "x": 0.0}
    nodes = [{**fields, "id": f"L{i:04d}", "name": name, "index": i, "y": float(i)}
             for i, name in enumerate(LAYERS)]
    meta = {"model": "Known", "step": None, "sample_index": 0, "label": None, "labels": None,
            "mode": "single", "version": "test", "unpaired": None}
    edges = [[f"L{i:04d}", f"L{i + 1:04d}"] for i in range(len(LAYERS) - 1)]
    return {"meta": meta, "input": "L0000", "nodes": nodes, "boxes": [], "edges": edges}


@pytest.fixture(scope="module")
def known_bundle(tmp_path_factory):
    """Write the known bundle and serve it over http, yielding the url of its folder."""
    root = tmp_path_factory.mktemp("known")
    payloads = {node_id: [one] for node_id, one in known_payloads().items()}
    write_bundle(root / "known", known_graph(), payloads, inline_mb=0)
    with serve_dir(root) as url:
        yield f"{url}known/"


def open_known(page, known_bundle):
    """Open the known bundle and return its console errors."""
    return open_bundle(page, f"{known_bundle}index.html")


def select_node(page, node_id, mode="pca"):
    """Click a graph node and wait until its layer is drawn in the given color mode."""
    page.click(f'#graph-svg g.node[data-id="{node_id}"]')
    page.wait_for_selector(f'#view[data-node="{node_id}"][data-mode="{mode}"] canvas')


def select_known(page, name):
    """Select a known layer by name and return its node id."""
    node_id = f"L{LAYERS.index(name):04d}"
    select_node(page, node_id)
    return node_id


def click_view_center(page):
    """Click the center of the first view canvas, which picks the position under it."""
    page.locator("#view canvas").first.click()
    page.wait_for_selector('#view[data-mode="similarity"] canvas')


def click_view_corner(page, right=False, bottom=False):
    """Click CORNER_PX inside one corner of the first view canvas, top left by default."""
    canvas = page.locator("#view canvas").first
    box = canvas.bounding_box()
    # The 1px canvas border is not part of the drawing, so the far edge sits one pixel in.
    x = box["width"] - 1 - CORNER_PX if right else CORNER_PX
    y = box["height"] - 1 - CORNER_PX if bottom else CORNER_PX
    canvas.click(position={"x": x, "y": y})
    page.wait_for_selector('#view[data-mode="similarity"] canvas')


def pixel(page, row, col, canvas=0):
    """Return the [r, g, b] of one pixel of the nth 2D view canvas."""
    script = """([row, col, nth]) => {
      const canvas = document.querySelectorAll('#view canvas')[nth];
      const data = canvas.getContext('2d').getImageData(col, row, 1, 1).data;
      return [data[0], data[1], data[2]];
    }"""
    return page.evaluate(script, [row, col, canvas])


def click_text(page):
    """Return the text of the clicked input coordinate."""
    return page.locator("#click").inner_text()


def legend_text(page):
    """Return the legend line of the view."""
    return page.locator(".layerviz-legend").inner_text()
