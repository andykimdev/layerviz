"""
Test the bundle writer: inline and multi file bundles, shipped web files, escaping, and caps.

Purpose:
- Check the single file and multi file bundles, the size caps, channels, and rgb_joint.
- Check that only page assets ship and that the inline block escapes every script ending.

Special Notes:
- Node tests fail when node is missing, since the full check runs node anyway.
- Bundles use a small fake web folder in tmp_path. Frame encoding is tested in test_framing.
"""

import base64
import json
import re
import shutil
import warnings

import numpy as np
import pytest

from layerviz.budget import MB
from layerviz.bundle import WEB_DIR, BundleOptions, LayerPayload, write_bundle
from layerviz.framing import encode_channels, encode_layer
from layerviz.inline import inline_block
from layerviz.reduce import PCAFit, Reduced
from layerviz.shapes.base import LayerData
from tests.harness import make_module_web, make_vendor_web, node_decode, run_node
from tests.models import read_frame

LICENSE = WEB_DIR / "vendor" / "LICENSE-three.txt"
MODULE_RE = re.compile(r'<script type="module">\n(.*)</script>$', re.S)
GRAPH_RE = re.compile(r'<script type="application/json" id="graph">(.*?)</script>', re.S)
FOLDER_FILES = ["app.js", "b.js", "graph.json", "index.html", "layers/L0007.bin", "style.css",
                "vendor/three.js"]


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


def layer_bytes(payload):
    """Encode a payload's layer file with the framing encoder."""
    return encode_layer(payload.data, payload.reduced, payload.rgb_joint)


def channel_bytes(payload):
    """Encode a payload's channel file with the framing encoder."""
    values, kept = payload.channels
    return encode_channels(values, np.asarray(kept), len(values))


def make_graph(mode="single"):
    """Build a hand made graph with the input node and one leaf layer L0007."""
    node = {"id": "L0007", "name": "conv", "kind": "image2d", "total": 10, "x": 1.0, "y": 2.0}
    return {"meta": {"mode": mode}, "input": "L0000", "nodes": [node], "boxes": [], "edges": []}


def write(tmp_path, payloads, graph=None, inline_mb=16, web=None, **limits):
    """Write a bundle into tmp_path/out with a fake web folder and return index.html."""
    options = BundleOptions(web_dir=web or make_module_web(tmp_path), **limits)
    return write_bundle(tmp_path / "out", graph or make_graph(), payloads, inline_mb, options)


def written_files(out):
    """Return the sorted relative posix paths of every file under one bundle folder."""
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())


def bundle_bytes(out):
    """Return the total size of every file written under one bundle folder."""
    return sum(path.stat().st_size for path in out.rglob("*") if path.is_file())


def graph_node(index_html):
    """Return node L0007 from the graph.json written next to index.html."""
    return json.loads((index_html.parent / "graph.json").read_text())["nodes"][0]


def html_comments(html):
    """Return the text inside every HTML comment of a page."""
    return re.findall(r"<!--(.*?)-->", html, re.S)


def module_script(block):
    """Return the module script text of an inline block or page body."""
    return MODULE_RE.search(block.removesuffix("</body></html>")).group(1)


def test_bundle_layer_file_decodes_in_format_js_with_grid_scale_and_labels(tmp_path):
    payload = make_payload(labels=["é", "b", "c", "d"])
    index = write(tmp_path, {"L0007": [payload]}, inline_mb=0)
    blob = (index.parent / "layers" / "L0007.bin").read_bytes()
    record = node_decode(tmp_path, blob, "parseLayer")
    header = record["header"]
    assert (header["n"], header["dim"], header["k"], header["kind"]) == (4, 2, 3, "image2d")
    labels = ["é", "b", "c", "d"]
    assert (header["grid"], header["scale"], header["labels"]) == ([2, 2], [1.5, 2.0], labels)
    np.testing.assert_array_equal(np.float32(record["positions"]), payload.data.positions.ravel())
    assert record["rgbJoint"] is None


def test_labels_not_one_per_position_raise_naming_labels(tmp_path):
    with pytest.raises(ValueError, match="labels"):
        write(tmp_path, {"L0007": [make_payload(labels=["a"])]})


def test_small_bundle_writes_only_index_html(tmp_path):
    index = write(tmp_path, {"L0007": [make_payload()]})
    assert written_files(index.parent) == ["index.html"]


def test_small_bundle_embeds_the_graph_with_its_file_field(tmp_path):
    html = write(tmp_path, {"L0007": [make_payload()]}).read_text()
    graph = json.loads(GRAPH_RE.search(html).group(1))
    assert graph["nodes"][0]["file"] == "layers/L0007.bin"


def test_small_bundle_embeds_the_layer_file_as_base64(tmp_path):
    payload = make_payload()
    html = write(tmp_path, {"L0007": [payload]}).read_text()
    blob = re.search(r'id="L0007" data-path="layers/L0007.bin">(.*?)</script>', html, re.S).group(1)
    assert base64.b64decode(blob) == layer_bytes(payload)


def test_small_bundle_replaces_the_marker_with_styles_and_import_free_code(tmp_path):
    html = write(tmp_path, {"L0007": [make_payload()]}).read_text()
    assert "<!-- layerviz:scripts -->" not in html
    assert "import " not in html
    assert "<style>\nbody { margin: 0; }</style>" in html


def add_stray_files(web):
    """Add an editor folder file and a source map that no page needs."""
    (web / ".DS_Store").write_bytes(b"\0\1junk")
    (web / "app.js.map").write_text('{"version": 3, "mappings": "MAPTEXT"}')


def test_inline_page_skips_ds_store_and_source_maps(tmp_path):
    web = make_module_web(tmp_path)
    add_stray_files(web)
    index = write(tmp_path, {"L0007": [make_payload()]}, web=web)
    assert written_files(index.parent) == ["index.html"]
    assert "MAPTEXT" not in index.read_text()


def test_folder_bundle_copies_no_ds_store_or_source_map(tmp_path):
    web = make_module_web(tmp_path)
    add_stray_files(web)
    index = write(tmp_path, {"L0007": [make_payload()]}, web=web, inline_mb=0)
    assert written_files(index.parent) == FOLDER_FILES


def test_shipped_html_file_that_cannot_inline_raises_in_inline_mode(tmp_path):
    web = make_module_web(tmp_path)
    (web / "help.html").write_text("<p>help</p>")
    with pytest.raises(ValueError, match="help.html"):
        write(tmp_path, {"L0007": [make_payload()]}, web=web)
    assert not (tmp_path / "out" / "help.html").exists()


def test_shipped_html_file_that_cannot_inline_is_copied_in_folder_mode(tmp_path):
    web = make_module_web(tmp_path)
    (web / "help.html").write_text("<p>help</p>")
    index = write(tmp_path, {"L0007": [make_payload()]}, web=web, inline_mb=0)
    assert (index.parent / "help.html").read_text() == "<p>help</p>"


@pytest.mark.parametrize("ending", ["</script>", "</SCRIPT>", "</ScRiPt >"])
def test_inline_block_escapes_a_script_end_in_any_letter_case(ending):
    web = {"app.js": f"console.log({json.dumps(ending)});\n".encode()}
    script = module_script(inline_block("{}", {}, web))
    assert ending not in script
    assert f'"<\\/{ending[2:]}"' in script


def test_inline_block_escapes_a_comment_open_in_the_module_script():
    web = {"app.js": b'console.log("<!--");\n'}
    script = module_script(inline_block("{}", {}, web))
    assert "<!--" not in script
    assert '"<\\!--"' in script


@pytest.mark.parametrize("ending", ["</script>", "</SCRIPT>"])
def test_inline_script_with_an_escaped_end_still_prints_the_original_text(ending):
    web = {"app.js": f"console.log({json.dumps(ending + '<!--')});\n".encode()}
    assert run_node(module_script(inline_block("{}", {}, web))) == ending + "<!--\n"


def test_inline_block_writes_lt_before_slash_or_bang_in_the_graph_as_unicode_escape():
    graph_json = json.dumps({"name": "a</SCRIPT><!--b<c"})
    text = GRAPH_RE.search(inline_block(graph_json, {}, {})).group(1)
    assert text == '{"name": "a\\u003c/SCRIPT>\\u003c!--b<c"}'
    assert json.loads(text) == {"name": "a</SCRIPT><!--b<c"}


@pytest.mark.parametrize(
    "source",
    [
        'export { x } from "./b.js";\n',
        'export{x as y}from"./b.js";',
        'const z=1;export{x}from"./b.js";',
    ],
)
def test_inline_block_rejects_an_export_from_clause_naming_the_file(source):
    web = {"app.js": source.encode(), "b.js": b"export const x = 1;\n"}
    with pytest.raises(ValueError, match="app.js"):
        inline_block("{}", {}, web)


def test_inline_page_embeds_the_vendor_license_in_an_html_comment(tmp_path):
    web = make_module_web(tmp_path)
    shutil.copy(LICENSE, web / "vendor" / LICENSE.name)
    index = write(tmp_path, {"L0007": [make_payload()]}, web=web)
    expected = f" vendor/{LICENSE.name}\n{LICENSE.read_text(encoding='utf-8')}\n"
    assert html_comments(index.read_text(encoding="utf-8")) == [expected]


def test_inline_page_of_the_real_web_folder_carries_the_three_license(tmp_path):
    web = tmp_path / "web"
    shutil.copytree(WEB_DIR, web, ignore=shutil.ignore_patterns("index.html", "__pycache__"))
    (web / "index.html").write_text("<body><!-- layerviz:scripts --></body>")
    index = write(tmp_path, {"L0007": [make_payload()]}, web=web)
    comments = html_comments(index.read_text(encoding="utf-8"))
    assert f" vendor/{LICENSE.name}\n{LICENSE.read_text(encoding='utf-8')}\n" in comments


def test_folder_bundle_ships_the_vendor_license_file(tmp_path):
    web = make_module_web(tmp_path)
    shutil.copy(LICENSE, web / "vendor" / LICENSE.name)
    index = write(tmp_path, {"L0007": [make_payload()]}, web=web, inline_mb=0)
    assert (index.parent / "vendor" / LICENSE.name).read_bytes() == LICENSE.read_bytes()


def test_vendor_modules_in_a_subfolder_run_after_their_imports(tmp_path):
    html = write(tmp_path, {"L0007": [make_payload()]}, web=make_vendor_web(tmp_path)).read_text()
    assert run_node(module_script(html)) == "41\n"


def test_vendor_modules_are_wrapped_vendor_first_in_import_order(tmp_path):
    html = write(tmp_path, {"L0007": [make_payload()]}, web=make_vendor_web(tmp_path)).read_text()
    names = re.findall(r"^const (__lv_\w+) =", module_script(html), re.M)
    assert names == ["__lv_vendor_a_js", "__lv_vendor_b_js", "__lv_app_js"]


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
    assert run_node(module_script(html)) == "41\n"


def test_inline_mb_zero_writes_graph_layers_and_web_files(tmp_path):
    index = write(tmp_path, {"L0007": [make_payload()]}, inline_mb=0)
    assert written_files(index.parent) == FOLDER_FILES


def test_inline_mb_zero_writes_the_layer_file_bytes(tmp_path):
    payload = make_payload()
    index = write(tmp_path, {"L0007": [payload]}, inline_mb=0)
    assert (index.parent / "layers" / "L0007.bin").read_bytes() == layer_bytes(payload)


def test_inline_mb_zero_page_links_styles_and_the_entry_module(tmp_path):
    index = write(tmp_path, {"L0007": [make_payload()]}, inline_mb=0)
    assert index.read_text() == (
        '<html><body><link rel="stylesheet" href="style.css">\n'
        '<script type="module" src="app.js"></script></body></html>'
    )


def test_graph_records_kept_total_and_file_fields(tmp_path):
    node = graph_node(write(tmp_path, {"L0007": [make_payload(n=6)]}, inline_mb=0))
    assert {k: node[k] for k in ("kept", "total", "file", "channels", "channel_list", "files")} == {
        "kept": 6, "total": 10, "file": "layers/L0007.bin", "channels": None, "channel_list": None,
        "files": None,
    }


def test_layer_at_layer_mb_is_written(tmp_path):
    payload = make_payload(n=6)
    size = len(layer_bytes(payload))
    index = write(tmp_path, {"L0007": [payload]}, inline_mb=0, layer_mb=size / MB)
    assert (index.parent / "layers" / "L0007.bin").stat().st_size == size


def test_layer_over_layer_mb_raises_naming_the_layer(tmp_path):
    with pytest.raises(ValueError, match="L0007"):
        write(tmp_path, {"L0007": [make_payload(n=6)]}, inline_mb=0, layer_mb=0.0001)


def test_bundle_over_total_mb_warns_counting_channel_files(tmp_path):
    payload = make_payload(n=6, channels=True)
    size = len(layer_bytes(payload)) + len(channel_bytes(payload))
    with pytest.warns(UserWarning, match="total_mb"):
        write(tmp_path, {"L0007": [payload]}, inline_mb=0, total_mb=(size - 1) / MB)


def test_bundle_at_total_mb_does_not_warn(tmp_path):
    payload = make_payload(n=6, channels=True)
    size = len(layer_bytes(payload)) + len(channel_bytes(payload))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(tmp_path, {"L0007": [payload]}, inline_mb=0, total_mb=size / MB)


def test_channel_file_and_list_follow_the_payload(tmp_path):
    index = write(tmp_path, {"L0007": [make_payload(channels=True)]}, inline_mb=0)
    node = graph_node(index)
    assert (node["channels"], node["channel_list"]) == ("layers/L0007.ch.bin", [4, 1])


def test_channel_file_holds_the_kept_channel_values(tmp_path):
    payload = make_payload(channels=True)
    index = write(tmp_path, {"L0007": [payload]}, inline_mb=0)
    assert (index.parent / "layers" / "L0007.ch.bin").read_bytes() == channel_bytes(payload)


def compare_panes():
    """Build the two panes of a compare node, of 4 and 6 positions, both with joint rgb."""
    return [make_payload(joint=True), make_payload(n=6, joint=True)]


def test_compare_node_records_both_pane_files_and_counts(tmp_path):
    index = write(tmp_path, {"L0007": compare_panes()}, graph=make_graph("compare"), inline_mb=0)
    node = graph_node(index)
    assert {k: node[k] for k in ("file", "files", "kept", "total", "kept_b", "total_b")} == {
        "file": "layers/L0007a.bin", "files": {"a": "layers/L0007a.bin", "b": "layers/L0007b.bin"},
        "kept": 4, "total": 10, "kept_b": 6, "total_b": 6,
    }


@pytest.mark.parametrize("pane, name", [(0, "L0007a.bin"), (1, "L0007b.bin")])
def test_compare_pane_file_holds_its_layer_with_joint_rgb(tmp_path, pane, name):
    panes = compare_panes()
    index = write(tmp_path, {"L0007": panes}, graph=make_graph("compare"), inline_mb=0)
    blob = (index.parent / "layers" / name).read_bytes()
    assert blob == layer_bytes(panes[pane])
    assert list(read_frame(blob)[0]["sections"]) == ["positions", "rgb", "scores", "rgb_joint"]


def test_compare_bundle_without_joint_rgb_raises(tmp_path):
    with pytest.raises(ValueError, match="rgb_joint"):
        write(tmp_path, {"L0007": [make_payload(), make_payload()]}, graph=make_graph("compare"))


def test_payload_count_must_match_the_mode(tmp_path):
    with pytest.raises(ValueError, match="L0007"):
        write(tmp_path, {"L0007": [make_payload(), make_payload()]})
