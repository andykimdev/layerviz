"""
Test the bundle writer: layer and channel framing, inline and multi file bundles, and caps.

Purpose:
- Cross check encode_layer and encode_channels against the browser decoder format.js.
- Check the single file and multi file bundles, the size caps, channels, and rgb_joint.

Special Notes:
- Node tests fail when node is missing, since the full check runs node anyway.
- Bundles use a small fake web folder in tmp_path.
"""

import base64
import json
import re
import shutil
import struct
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from layerviz.bundle import (
    WEB_DIR,
    BundleOptions,
    LayerPayload,
    encode_channels,
    encode_layer,
    write_bundle,
)
from layerviz.budget import MB
from layerviz.reduce import PCAFit, Reduced
from layerviz.shapes.base import LayerData

FORMAT_JS = Path(__file__).resolve().parents[1] / "layerviz" / "web" / "format.js"
NODE = shutil.which("node")
LICENSE = WEB_DIR / "vendor" / "LICENSE-three.txt"


def make_payload(n=4, k=3, joint=False, channels=False, labels=None):
    """Build a deterministic image2d payload with n positions and k stored scores."""
    rng = np.random.default_rng(n + k)
    data = LayerData(
        kind="image2d",
        positions=rng.normal(size=(n, 2)).astype(np.float32),
        features=rng.normal(size=(n, 5)).astype(np.float32),
        grid=(2, n // 2),
        scale=(1.5, 2.0),
        labels=labels,
    )
    fit = PCAFit(np.eye(3), np.ones(3) / 3, np.zeros(5), np.ones(5))
    reduced = Reduced(
        rgb=rng.integers(0, 256, size=(n, 3), dtype=np.uint8),
        scores=rng.normal(size=(n, k)).astype(np.float16),
        explained=np.array([0.5, 0.3, 0.1]),
        retained=0.9,
        fit=fit,
    )
    joint_rgb = rng.integers(0, 256, size=(n, 3), dtype=np.uint8) if joint else None
    chans = (rng.integers(0, 256, size=(n, 2), dtype=np.uint8), [4, 1]) if channels else None
    return LayerPayload(data=data, reduced=reduced, rgb_joint=joint_rgb, channels=chans)


def make_graph(mode="single"):
    """Build a hand made graph with the input node and one leaf layer L0007."""
    node = {"id": "L0007", "name": "conv", "kind": "image2d", "total": 10, "x": 1.0, "y": 2.0}
    return {"meta": {"mode": mode}, "input": "L0000", "nodes": [node], "boxes": [], "edges": []}


def make_web(root):
    """Write a fake web folder whose modules share a helper name and import each other."""
    web = root / "web"
    (web / "vendor").mkdir(parents=True)
    (web / "index.html").write_text("<html><body><!-- layerviz:scripts --></body></html>")
    (web / "vendor" / "three.js").write_text("const base = 20;\nexport{base as BASE};\n")
    (web / "b.js").write_text(
        'import { BASE } from "./vendor/three.js";\n'
        "function helper() { return 1; }\nexport function twice(x) { return 2 * x + helper() + BASE; }\n"
    )
    (web / "app.js").write_text(
        'import {\n  twice,\n} from "./b.js";\nfunction helper() { return 0; }\n'
        "console.log(twice(10) + helper());\n"
    )
    (web / "style.css").write_text("body { margin: 0; }")
    (web / "CLAUDE.md").write_text("map")
    return web


def make_vendor_web(root):
    """Write a web folder whose vendor modules import each other in minified form."""
    web = root / "web"
    (web / "vendor").mkdir(parents=True)
    (web / "index.html").write_text("<!-- layerviz:scripts -->")
    (web / "vendor" / "a.js").write_text("const n=20;export{n as BASE,n as N};")
    (web / "vendor" / "b.js").write_text('import{BASE as t}from"./a.js";import{N}from"./a.js";const o=t+N;export{o as BOTH};')
    (web / "app.js").write_text('import { BOTH } from "./vendor/b.js";\nconsole.log(BOTH + 1);\n')
    return web


def write(tmp_path, payloads, graph=None, inline_mb=16, web=None, **limits):
    """Write a bundle into tmp_path/out with a fake web folder and return index.html."""
    options = BundleOptions(web_dir=web or make_web(tmp_path), **limits)
    return write_bundle(tmp_path / "out", graph or make_graph(), payloads, inline_mb, options)


def bundle_bytes(out):
    """Return the total size of every file written under one bundle folder."""
    return sum(path.stat().st_size for path in out.rglob("*") if path.is_file())


def read_frame(blob):
    """Split framed bytes into the header and the data that follows it."""
    (length,) = struct.unpack("<I", blob[:4])
    return json.loads(blob[4 : 4 + length].decode("utf-8")), blob[4 + length :], length


def run_node(script):
    """Run an ES module script in node and return its standard output."""
    assert NODE, "node must be on PATH to cross check the bundle against format.js"
    done = subprocess.run(
        [NODE, "--input-type=module", "-e", script], capture_output=True, text=True, check=True
    )
    return done.stdout


def node_decode(tmp_path, blob, parser):
    """Decode bytes with a format.js parser in node and return the record as plain JSON."""
    path = tmp_path / "blob.bin"
    path.write_bytes(blob)
    script = (
        f'import {{ {parser} }} from "{FORMAT_JS.as_uri()}";\n'
        'import { readFileSync } from "node:fs";\n'
        f"const b = readFileSync({json.dumps(str(path))});\n"
        f"const r = {parser}(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));\n"
        "const out = {};\n"
        "for (const [k, v] of Object.entries(r)) out[k] = v && v.length !== undefined ? Array.from(v) : v;\n"
        "console.log(JSON.stringify(out));\n"
    )
    return json.loads(run_node(script))


def graph_node(index_html):
    """Return node L0007 from the graph.json written next to index.html."""
    return json.loads((index_html.parent / "graph.json").read_text())["nodes"][0]


@pytest.mark.parametrize("joint", [False, True])
def test_encode_layer_round_trips_through_format_js(tmp_path, joint):
    payload = make_payload(joint=joint, labels=["é", "b", "c", "d"])
    record = node_decode(tmp_path, encode_layer(payload), "parseLayer")
    header = record["header"]
    assert (header["n"], header["dim"], header["k"], header["kind"]) == (4, 2, 3, "image2d")
    assert (header["grid"], header["scale"], header["labels"]) == ([2, 2], [1.5, 2.0], ["é", "b", "c", "d"])
    np.testing.assert_array_equal(np.float32(record["positions"]), payload.data.positions.ravel())
    np.testing.assert_array_equal(record["rgb"], payload.reduced.rgb.ravel())
    np.testing.assert_array_equal(np.float32(record["scores"]), payload.reduced.scores.astype(np.float32).ravel())
    expected_joint = payload.rgb_joint.ravel().tolist() if joint else None
    assert record["rgbJoint"] == expected_joint


def test_encode_channels_round_trips_through_format_js(tmp_path):
    payload = make_payload(channels=True)
    record = node_decode(tmp_path, encode_channels(payload), "parseChannels")
    assert (record["header"]["n"], record["header"]["c"], record["header"]["channels"]) == (4, 2, [4, 1])
    np.testing.assert_array_equal(record["values"], payload.channels[0].ravel())


def test_encode_layer_aligns_header_and_sections_to_four_bytes():
    header, data, length = read_frame(encode_layer(make_payload(n=6, k=3, labels=["é"] * 6)))
    sections = header["sections"]
    assert (4 + length) % 4 == 0
    assert sections == {"positions": [0, 48, "float32"], "rgb": [48, 18, "uint8"], "scores": [68, 36, "float16"]}
    assert len(data) == 104


def test_encode_layer_rejects_labels_not_one_per_position():
    with pytest.raises(ValueError, match="labels"):
        encode_layer(make_payload(labels=["a"]))


def test_small_bundle_is_one_self_contained_index_html(tmp_path):
    payload = make_payload()
    index = write(tmp_path, {"L0007": [payload]})
    html = index.read_text()
    assert sorted(p.name for p in index.parent.iterdir()) == ["index.html"]
    graph = json.loads(re.search(r'<script type="application/json" id="graph">(.*?)</script>', html, re.S).group(1))
    assert graph["nodes"][0]["file"] == "layers/L0007.bin"
    blob = re.search(r'id="L0007" data-path="layers/L0007.bin">(.*?)</script>', html, re.S).group(1)
    assert base64.b64decode(blob) == encode_layer(payload)
    assert "<!-- layerviz:scripts -->" not in html and "import " not in html
    assert "<style>\nbody { margin: 0; }</style>" in html


def test_web_file_that_cannot_inline_raises_in_inline_mode(tmp_path):
    web = make_web(tmp_path)
    (web / "logo.png").write_bytes(b"png")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="logo.png"):
            write(tmp_path, {"L0007": [make_payload()]}, web=web)
    assert not (tmp_path / "out" / "logo.png").exists()


def test_web_file_that_cannot_inline_is_copied_in_folder_mode(tmp_path):
    web = make_web(tmp_path)
    (web / "logo.png").write_bytes(b"png")
    index = write(tmp_path, {"L0007": [make_payload()]}, web=web, inline_mb=0)
    assert (index.parent / "logo.png").read_bytes() == b"png"
    assert graph_node(index)["file"] == "layers/L0007.bin"


def html_comments(html):
    """Return the text inside every HTML comment of a page."""
    return re.findall(r"<!--(.*?)-->", html, re.S)


def test_inline_page_embeds_the_vendor_license_in_an_html_comment(tmp_path):
    web = make_web(tmp_path)
    shutil.copy(LICENSE, web / "vendor" / LICENSE.name)
    index = write(tmp_path, {"L0007": [make_payload()]}, web=web)
    license_text = LICENSE.read_text(encoding="utf-8")
    assert sorted(p.name for p in index.parent.iterdir()) == ["index.html"]
    assert html_comments(index.read_text(encoding="utf-8")) == [
        f" vendor/{LICENSE.name}\n{license_text}\n"
    ]


def test_inline_page_of_the_real_web_folder_carries_the_three_license(tmp_path):
    web = tmp_path / "web"
    shutil.copytree(WEB_DIR, web, ignore=shutil.ignore_patterns("index.html", "__pycache__"))
    (web / "index.html").write_text("<body><!-- layerviz:scripts --></body>")
    index = write(tmp_path, {"L0007": [make_payload()]}, web=web)
    comments = html_comments(index.read_text(encoding="utf-8"))
    assert f" vendor/{LICENSE.name}\n{LICENSE.read_text(encoding='utf-8')}\n" in comments
    assert not (index.parent / "graph.json").exists()


def test_folder_bundle_ships_the_vendor_license_file(tmp_path):
    web = make_web(tmp_path)
    shutil.copy(LICENSE, web / "vendor" / LICENSE.name)
    index = write(tmp_path, {"L0007": [make_payload()]}, web=web, inline_mb=0)
    assert (index.parent / "vendor" / LICENSE.name).read_bytes() == LICENSE.read_bytes()


def test_vendor_modules_in_a_subfolder_import_each_other_minified(tmp_path):
    html = write(tmp_path, {"L0007": [make_payload()]}, web=make_vendor_web(tmp_path)).read_text()
    script = re.search(r'<script type="module">(.*?)</script>', html, re.S).group(1)
    assert run_node(script).strip() == "41"
    assert [m.group(1) for m in re.finditer(r"^const (__lv_\w+) =", script, re.M)] == [
        "__lv_vendor_a_js", "__lv_vendor_b_js", "__lv_app_js",
    ]


@pytest.mark.parametrize("extra, inlined", [(0, True), (-1, False)])
def test_inline_choice_measures_the_built_page(tmp_path, extra, inlined):
    payload = make_payload(n=600)
    folder_size = bundle_bytes(write(tmp_path / "folder", {"L0007": [payload]}, inline_mb=0).parent)
    page_size = write(tmp_path / "page", {"L0007": [payload]}).stat().st_size
    assert page_size > folder_size
    index = write(tmp_path, {"L0007": [payload]}, inline_mb=(page_size + extra) / MB)
    assert (index.parent / "graph.json").exists() is not inlined


def test_inline_scripts_run_in_dependency_order_in_one_module(tmp_path):
    html = write(tmp_path, {"L0007": [make_payload()]}).read_text()
    script = re.search(r'<script type="module">(.*?)</script>', html, re.S).group(1)
    assert run_node(script).strip() == "41"


def test_inline_mb_zero_writes_graph_layers_and_web_files(tmp_path):
    payload = make_payload()
    index = write(tmp_path, {"L0007": [payload]}, inline_mb=0)
    out = index.parent
    names = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert names == [
        "app.js", "b.js", "graph.json", "index.html", "layers/L0007.bin", "style.css", "vendor/three.js",
    ]
    assert (out / "layers" / "L0007.bin").read_bytes() == encode_layer(payload)
    assert index.read_text() == (
        '<html><body><link rel="stylesheet" href="style.css">\n'
        '<script type="module" src="app.js"></script></body></html>'
    )


def test_graph_records_kept_total_and_file_fields(tmp_path):
    node = graph_node(write(tmp_path, {"L0007": [make_payload(n=6)]}, inline_mb=0))
    assert {k: node[k] for k in ("kept", "total", "file", "channels", "channel_list", "files")} == {
        "kept": 6, "total": 10, "file": "layers/L0007.bin", "channels": None, "channel_list": None, "files": None,
    }


def test_layer_at_layer_mb_is_written(tmp_path):
    payload = make_payload(n=6)
    size = len(encode_layer(payload))
    index = write(tmp_path, {"L0007": [payload]}, inline_mb=0, layer_mb=size / MB)
    assert (index.parent / "layers" / "L0007.bin").stat().st_size == size


def test_layer_over_layer_mb_raises_naming_the_layer(tmp_path):
    with pytest.raises(ValueError, match="L0007"):
        write(tmp_path, {"L0007": [make_payload(n=6)]}, inline_mb=0, layer_mb=0.0001)


def test_bundle_over_total_mb_warns_counting_channel_files(tmp_path):
    payload = make_payload(n=6, channels=True)
    size = len(encode_layer(payload)) + len(encode_channels(payload))
    with pytest.warns(UserWarning, match="total_mb"):
        write(tmp_path, {"L0007": [payload]}, inline_mb=0, total_mb=(size - 1) / MB)


def test_bundle_at_total_mb_does_not_warn(tmp_path):
    payload = make_payload(n=6, channels=True)
    size = len(encode_layer(payload)) + len(encode_channels(payload))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(tmp_path, {"L0007": [payload]}, inline_mb=0, total_mb=size / MB)


def test_channel_file_and_list_follow_the_payload(tmp_path):
    payload = make_payload(channels=True)
    index = write(tmp_path, {"L0007": [payload]}, inline_mb=0)
    node = graph_node(index)
    assert (node["channels"], node["channel_list"]) == ("layers/L0007.ch.bin", [4, 1])
    assert (index.parent / "layers" / "L0007.ch.bin").read_bytes() == encode_channels(payload)


def test_compare_bundle_writes_both_panes_with_joint_rgb(tmp_path):
    panes = [make_payload(joint=True), make_payload(n=6, joint=True)]
    index = write(tmp_path, {"L0007": panes}, graph=make_graph("compare"), inline_mb=0)
    node = graph_node(index)
    assert node["files"] == {"a": "layers/L0007a.bin", "b": "layers/L0007b.bin"}
    assert node["file"] == "layers/L0007a.bin"
    assert {k: node[k] for k in ("kept", "total", "kept_b", "total_b")} == {
        "kept": 4, "total": 10, "kept_b": 6, "total_b": 6,
    }
    for pane, path in zip(panes, ("L0007a.bin", "L0007b.bin")):
        blob = (index.parent / "layers" / path).read_bytes()
        assert blob == encode_layer(pane)
        assert "rgb_joint" in read_frame(blob)[0]["sections"]


def test_compare_bundle_without_joint_rgb_raises(tmp_path):
    with pytest.raises(ValueError, match="rgb_joint"):
        write(tmp_path, {"L0007": [make_payload(), make_payload()]}, graph=make_graph("compare"))


def test_payload_count_must_match_the_mode(tmp_path):
    with pytest.raises(ValueError, match="L0007"):
        write(tmp_path, {"L0007": [make_payload(), make_payload()]})
