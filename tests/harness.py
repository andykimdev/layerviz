"""
Share the layer file readers, node runner, web folder builders, and browser fixtures.

Purpose:

- Read framed layer files, graph.json, and positions back into Python for assertions.
- Run format.js in node and write the fake web folders that bundle tests need.
- Serve bundles over http and provide the module scoped browser and page fixtures.
- Release held page requests and wait until the page has handled them.
- Write comparison bundles and pick out their paired nodes for the compare tests.

Special Notes:

- playwright is imported inside the browser fixture, so modules that never use a browser
  still import this file.
- The browser launches with software WebGL, since the test machines have no GPU.
"""

import functools
import http.server
import json
import shutil
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from layerviz.comparison import CompareOptions, compare
from layerviz.pipeline import CaptureOptions, run_pipeline
from tests.models import make_web, read_frame

FORMAT_JS = Path(__file__).resolve().parents[1] / "layerviz" / "web" / "format.js"
NODE = shutil.which("node")
WEBGL_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
INDEX_HTML = "<html><body><!-- layerviz:scripts --></body></html>"


def read_sections(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """
    Decode a framed layer file into its header and its sections as NumPy arrays.

    Args:
        path: Layer file whose header lists sections as name: (offset, size, dtype).

    Returns:
        The tuple (header dict, dict mapping section name to a 1D array).
    """
    header, data, _ = read_frame(path.read_bytes())
    sections = {}
    for name, (offset, size, dtype) in header["sections"].items():
        sections[name] = np.frombuffer(data[offset : offset + size], dtype=np.dtype(dtype))
    return header, sections


def read_graph(folder: Path) -> dict:
    """
    Load graph.json from a bundle folder.

    Args:
        folder: Bundle folder that holds graph.json.

    Returns:
        The parsed graph dict.
    """
    return json.loads((folder / "graph.json").read_text(encoding="utf-8"))


def read_positions(blob: bytes) -> np.ndarray:
    """
    Decode the positions section of one layer file.

    Args:
        blob: Framed layer file bytes.

    Returns:
        A float32 array [n, dim].
    """
    header, body, _ = read_frame(blob)
    offset, length, _ = header["sections"]["positions"]
    data = np.frombuffer(body[offset : offset + length], dtype="<f4")
    return data.reshape(header["n"], header["dim"])


def run_compare(
    tmp_path: Path, model_a: nn.Module, model_b: nn.Module, batch: Any, **options: Any
) -> tuple[dict, Path]:
    """
    Write a comparison folder bundle and read its graph back.

    Args:
        tmp_path: Folder that receives the web folder and the out bundle folder.
        model_a: Model shown in pane a.
        model_b: Model shown in pane b.
        batch: Input batch given to both models.
        **options: Extra CompareOptions fields such as capture or labels.

    Returns:
        The tuple (parsed graph dict, bundle folder path).
    """
    out = tmp_path / "out"
    opts = CompareOptions(inline_mb=0, web_dir=make_web(tmp_path), **options)
    compare(model_a, model_b, batch, out, opts)
    return read_graph(out), out


def pane_sections(out: Path, node: dict) -> list[dict[str, np.ndarray]]:
    """
    Return the decoded sections of pane a and pane b of one paired node.

    Args:
        out: Bundle folder that holds the layer files.
        node: Paired graph node whose files entry names the pane a and pane b files.

    Returns:
        Two dicts of decoded section arrays, pane a first.
    """
    return [read_sections(out / node["files"][pane])[1] for pane in ("a", "b")]


def expect_paired(graph: dict, names: list[str]) -> list[dict]:
    """
    Assert the exact names of the paired nodes in order and return those nodes.

    Args:
        graph: Parsed comparison graph dict.
        names: Module names that must carry two panes, in graph order.

    Returns:
        The graph nodes that carry two panes.

    Notes:
        Looping over an empty pairing checks nothing, so every compare test that loops over
        paired nodes goes through this assertion first.
    """
    nodes = [node for node in graph["nodes"] if node["files"] is not None]
    assert [node["name"] for node in nodes] == names
    return nodes


def other_model(width: int = 6) -> nn.Module:
    """
    Build an image2d shaped model with different random weights and a given last width.

    Args:
        width: Output channel count of the last convolution.

    Returns:
        A two convolution model in eval mode, seeded differently from the shared test models.
    """
    torch.manual_seed(1)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.Conv2d(4, width, 3, stride=2, padding=1)
    ).eval()


def captured(model: nn.Module, batch: Any) -> dict:
    """
    Return the captured layers of a model by module name.

    Args:
        model: Model to run through the capture pipeline.
        batch: Input batch passed to the model.

    Returns:
        A dict that maps each module name to its captured layer.
    """
    result = run_pipeline(model, batch, CaptureOptions())
    return {layer.node.name: layer for layer in result.layers}


def zscore(features: Any) -> np.ndarray:
    """
    Standardize each column in float64 with plain NumPy, leaving constant columns at zero.

    Args:
        features: Array like of shape [N, C] with one feature per column.

    Returns:
        A float64 array of shape [N, C] with zero mean and unit spread per column.
    """
    f = np.asarray(features, dtype=np.float64)
    std = f.std(axis=0)
    return np.where(std > 0, (f - f.mean(axis=0)) / np.where(std > 0, std, 1), 0)


def percentile_bytes(values: np.ndarray) -> np.ndarray:
    """
    Scale each column from its 1st to 99th percentile onto 0..255 with plain NumPy.

    Args:
        values: Array of shape [N, C] with one channel per column.

    Returns:
        A float array of shape [N, C] holding rounded values from 0 to 255.
    """
    low, high = np.percentile(values, [1, 99], axis=0)
    unit = np.clip((values - low) / np.where(high > low, high - low, 1.0), 0.0, 1.0)
    return np.rint(unit * 255)


def run_node(script: str) -> str:
    """
    Run an ES module script in node.

    Args:
        script: JavaScript module source.

    Returns:
        The standard output of the script.
    """
    assert NODE, "node must be on PATH to cross check the bundle against format.js"
    done = subprocess.run(
        [NODE, "--input-type=module", "-e", script], capture_output=True, text=True, check=True
    )
    return done.stdout


def node_decode(tmp_path: Path, blob: bytes, parser: str) -> dict:
    """
    Decode bytes with a format.js parser in node.

    Args:
        tmp_path: Folder that receives the temporary blob file.
        blob: Bytes to decode.
        parser: Name of the format.js function to call.

    Returns:
        The decoded record as plain JSON, with typed arrays turned into lists.
    """
    path = tmp_path / "blob.bin"
    path.write_bytes(blob)
    script = (
        f'import {{ {parser} }} from "{FORMAT_JS.as_uri()}";\n'
        'import { readFileSync } from "node:fs";\n'
        f"const b = readFileSync({json.dumps(str(path))});\n"
        f"const r = {parser}(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));\n"
        "const out = {};\n"
        "for (const [k, v] of Object.entries(r)) "
        "out[k] = v && v.length !== undefined ? Array.from(v) : v;\n"
        "console.log(JSON.stringify(out));\n"
    )
    return json.loads(run_node(script))


def make_module_web(root: Path) -> Path:
    """
    Write a web folder whose modules share a helper name and import each other.

    Args:
        root: Folder that receives the web subfolder.

    Returns:
        The path of the new web folder.
    """
    web = root / "web"
    (web / "vendor").mkdir(parents=True)
    (web / "index.html").write_text(INDEX_HTML)
    (web / "vendor" / "three.js").write_text("const base = 20;\nexport{base as BASE};\n")
    (web / "b.js").write_text(
        'import { BASE } from "./vendor/three.js";\n'
        "function helper() { return 1; }\n"
        "export function twice(x) { return 2 * x + helper() + BASE; }\n"
    )
    (web / "app.js").write_text(
        'import {\n  twice,\n} from "./b.js";\nfunction helper() { return 0; }\n'
        "console.log(twice(10) + helper());\n"
    )
    (web / "style.css").write_text("body { margin: 0; }")
    (web / "NOTES.md").write_text("map")
    return web


def make_vendor_web(root: Path) -> Path:
    """
    Write a web folder whose vendor modules import each other in minified form.

    Args:
        root: Folder that receives the web subfolder.

    Returns:
        The path of the new web folder.
    """
    web = root / "web"
    (web / "vendor").mkdir(parents=True)
    (web / "index.html").write_text("<!-- layerviz:scripts -->")
    (web / "vendor" / "a.js").write_text("const n=20;export{n as BASE,n as N};")
    (web / "vendor" / "b.js").write_text(
        'import{BASE as t}from"./a.js";import{N}from"./a.js";const o=t+N;export{o as BOTH};'
    )
    (web / "app.js").write_text('import { BOTH } from "./vendor/b.js";\nconsole.log(BOTH + 1);\n')
    return web


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files without logging every request to stderr."""

    def log_message(self, *args: Any) -> None:
        """Drop the access log line."""


@contextmanager
def serve_dir(path: Path) -> Iterator[str]:
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


def open_bundle(page: Any, url: str) -> list[str]:
    """
    Open a bundle page, wait for the graph to be drawn, and collect console errors.

    Args:
        page: A Playwright page.
        url: The index.html URL, file:// or http.

    Returns:
        A list that keeps receiving console error texts and uncaught page errors.
    """
    errors: list[str] = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url)
    page.wait_for_selector("#graph-svg g.node")
    return errors


def release(page: Any, route: Any, path: str, fail: bool = False) -> None:
    """
    Let a held bundle request through or abort it, then wait until every waiter has finished.

    Args:
        page: A Playwright page whose app.js holds the load of path.
        route: The held Playwright route of path.
        path: Bundle relative path of the held file.
        fail: Abort the request instead of letting it through.
    """
    # A failed load leaves the cache, so its promise is taken before the route is released.
    page.evaluate("""async (path) => {
      const { loadLayer } = await import("./app.js");
      window.layervizHeld = loadLayer(path).catch(() => {});
    }""", path)
    route.abort() if fail else route.continue_()
    # Awaiting the same promise, then one task, lets every paint awaiting it finish first.
    page.evaluate("""async () => {
      await window.layervizHeld;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }""")


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    """Launch one headless Chromium with software WebGL for every test in a module."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch(args=WEBGL_ARGS)
        yield chromium
        chromium.close()


@pytest.fixture
def page(browser: Any) -> Iterator[Any]:
    """Open a fresh page that closes after the test."""
    fresh = browser.new_page()
    yield fresh
    fresh.close()
