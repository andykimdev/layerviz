"""
Write a layerviz bundle: graph.json, binary layer and channel files, web files, and index.html.

Purpose:
- Frame each layer and channel file in the little endian layout format.js decodes.
- Inline everything into one index.html when the built page fits inline_mb, else write a folder.

Special Notes:
- The only module that writes files. Encoding and the inline transform in inline.py are pure.
- Inline mode raises ValueError for a web file that is not .js, .css, or .txt.
"""

import copy
import json
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from layerviz.budget import MB
from layerviz.inline import inline_block
from layerviz.reduce import Reduced
from layerviz.shapes.base import LayerData

WEB_DIR = Path(__file__).resolve().parent / "web"
MARKER = "<!-- layerviz:scripts -->"
ENTRY = "app.js"
PANES = {"single": ("",), "compare": ("a", "b")}


@dataclass
class LayerPayload:
    """
    Hold one layer's positions, reduction, optional joint RGB, and optional channel values.

    """

    data: LayerData
    reduced: Reduced
    rgb_joint: Optional[np.ndarray] = None
    channels: Optional[tuple] = None


@dataclass(frozen=True)
class BundleOptions:
    """
    Hold the per file and total size caps in megabytes and the folder of web files.

    """

    layer_mb: float = 16
    total_mb: float = 1024
    web_dir: Path = field(default=WEB_DIR)


def _plain_json(value):
    """Convert a NumPy scalar or array to plain Python for json.dumps, else raise TypeError."""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"cannot write {type(value).__name__} to JSON")


def _frame(header: dict, sections: list) -> bytes:
    """
    Frame a JSON header and (name, array, dtype) sections as length, header, then data.

    The header is space padded by UTF-8 byte count so data starts at a multiple of 4, and each
    section offset counts from the header end and is a multiple of 4.
    """
    specs, chunks, offset = {}, [], 0
    for name, array, dtype in sections:
        raw = np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<")).tobytes()
        specs[name] = [offset, len(raw), dtype]
        chunks.append(raw + b"\0" * (-len(raw) % 4))
        offset += len(chunks[-1])
    return _frame_header(header, specs) + b"".join(chunks)


def _frame_header(header: dict, specs: dict) -> bytes:
    """Return the length prefix and the JSON header padded so the data starts at a multiple of 4."""
    text = json.dumps({**header, "sections": specs}, ensure_ascii=False, default=_plain_json)
    raw_header = text.encode("utf-8")
    raw_header += b" " * (-(4 + len(raw_header)) % 4)
    return struct.pack("<I", len(raw_header)) + raw_header


def _check_rows(name: str, array, n: int, cols: Optional[int] = None) -> None:
    """Raise ValueError naming the field unless the array has n rows and, if given, cols columns."""
    shape = np.shape(array)
    if shape[0] != n or (cols is not None and (len(shape) != 2 or shape[1] != cols)):
        want = f"[{n}, {cols}]" if cols is not None else f"{n} rows"
        raise ValueError(f"{name} has shape {list(shape)}, expected {want}")


def encode_layer(payload: LayerPayload) -> bytes:
    """
    Encode one layer as framed positions, rgb, scores, and rgb_joint when given.

    Args:
        payload: The layer to encode.

    Returns:
        The layer file bytes that format.js parseLayer decodes.

    Raises:
        ValueError: When an array or the labels do not have one row per position.
    """
    data, reduced = payload.data, payload.reduced
    n = data.positions.shape[0]
    _check_rows("rgb", reduced.rgb, n, 3)
    _check_rows("scores", reduced.scores, n)
    if data.labels is not None:
        _check_rows("labels", data.labels, n)
    header = {
        "n": n, "dim": data.positions.shape[1], "k": reduced.scores.shape[1], "kind": data.kind,
        "grid": data.grid, "scale": data.scale, "labels": data.labels,
    }
    sections = [
        ("positions", data.positions, "float32"),
        ("rgb", reduced.rgb, "uint8"),
        ("scores", reduced.scores, "float16"),
    ]
    if payload.rgb_joint is not None:
        _check_rows("rgb_joint", payload.rgb_joint, n, 3)
        sections.append(("rgb_joint", payload.rgb_joint, "uint8"))
    return _frame(header, sections)


def encode_channels(payload: LayerPayload) -> bytes:
    """
    Encode one layer's kept channel values as a framed channel file.

    Args:
        payload: A layer whose channels field holds (uint8 [N, C] values, kept indices).

    Returns:
        The channel file bytes that format.js parseChannels decodes.

    Raises:
        ValueError: When the payload has no channels or the values do not match the indices.
    """
    if payload.channels is None:
        raise ValueError("payload has no channels to encode")
    values, kept = payload.channels
    _check_rows("channel values", values, payload.data.positions.shape[0], len(kept))
    header = {"n": len(values), "c": len(kept), "channels": [int(i) for i in kept]}
    return _frame(header, [("values", values, "uint8")])


def channel_overhead(n: int, n_channels: int) -> int:
    """
    Return the most bytes a channel file of n rows adds beyond its channel values.

    Args:
        n: Rows, one per kept position.
        n_channels: Channels of the layer, an upper bound on the channels kept.

    Returns:
        The length prefix, the padded header listing every channel, and 3 bytes of
        section padding. A file keeping any subset of the channels is never larger.
    """
    header = {"n": n, "c": n_channels, "channels": list(range(n_channels))}
    return len(_frame_header(header, {"values": [0, n * n_channels, "uint8"]})) + 3


def layer_overhead(data: LayerData, k: int = 32) -> int:
    """
    Return the most bytes a layer file of any subset of data adds beyond its position bytes.

    Args:
        data: The full layer before subsampling.
        k: Most score columns a layer stores, the reduce_layer default.

    Returns:
        The length prefix, the padded header, and 3 bytes of padding for each of the rgb,
        scores, and rgb_joint sections. It bounds single and compare mode files alike.

    Notes:
        Every section offset and size is set to the full file size, so a smaller subset never
        writes more digits, and the full labels bound the labels of any subset.
    """
    n, dim = data.positions.shape
    full = n * (dim * 4 + 3 + k * 2 + 3) + 12
    header = {
        "n": n, "dim": dim, "k": k, "kind": data.kind,
        "grid": data.grid, "scale": data.scale, "labels": data.labels,
    }
    dtypes = {"positions": "float32", "rgb": "uint8", "scores": "float16", "rgb_joint": "uint8"}
    specs = {name: [full, full, dtype] for name, dtype in dtypes.items()}
    return len(_frame_header(header, specs)) + 3 * 3


def _node_files(node_id: str, panes: list, mode: str) -> tuple:
    """
    Encode one node's panes and channel file, returning (path to bytes, node fields).

    Channels come from the first pane, kept counts pane a and kept_b pane b. Raises ValueError
    when the pane count does not match the mode or a compare pane lacks rgb_joint.
    """
    suffixes = PANES[mode]
    if len(panes) != len(suffixes):
        raise ValueError(f"node {node_id} has {len(panes)} payloads, {mode} mode needs {len(suffixes)}")
    files = {}
    for suffix, pane in zip(suffixes, panes):
        if mode == "compare" and pane.rgb_joint is None:
            raise ValueError(f"node {node_id} pane {suffix} has no rgb_joint, a comparison needs it")
        files[f"layers/{node_id}{suffix}.bin"] = encode_layer(pane)
    paths, first = list(files), panes[0]
    fields = {
        "file": paths[0], "files": dict(zip(suffixes, paths)) if mode == "compare" else None,
        "channels": None, "channel_list": None,
    }
    fields.update(zip(("kept", "kept_b"), (int(pane.data.positions.shape[0]) for pane in panes)))
    if first.channels is not None:
        path = f"layers/{node_id}{suffixes[0]}.ch.bin"
        files[path] = encode_channels(first)
        fields.update(channels=path, channel_list=[int(i) for i in first.channels[1]])
    return files, fields


def _fill_graph(graph: dict, payloads: dict) -> tuple:
    """
    Encode every payload and return (graph copy with file fields filled, path to bytes).

    Nodes without payloads get null file fields. Raises ValueError when a payload names a
    node the graph lacks.
    """
    graph = copy.deepcopy(graph)
    mode = graph.get("meta", {}).get("mode", "single")
    nodes = {node["id"]: node for node in graph["nodes"]}
    missing = sorted(set(payloads) - set(nodes))
    if missing:
        raise ValueError(f"payloads name nodes missing from the graph: {missing}")
    files = {}
    for node_id, node in nodes.items():
        for key in ("file", "files", "channels", "channel_list"):
            node.setdefault(key, None)
        if node_id in payloads:
            encoded, fields = _node_files(node_id, payloads[node_id], mode)
            files.update(encoded)
            node.update(fields)
            for key in ("kept", "kept_b")[: len(payloads[node_id])]:
                node.setdefault(key.replace("kept", "total"), fields[key])
    return graph, files


def _check_sizes(files: dict, options: BundleOptions) -> None:
    """
    Raise ValueError naming any file over layer_mb, and warn when all exceed total_mb.

    The total only warns, since the budget keeps at least 1 position per layer.
    """
    for path, blob in files.items():
        if len(blob) > options.layer_mb * MB:
            raise ValueError(f"{path} is {len(blob)} bytes, over layer_mb={options.layer_mb:g}")
    total = sum(len(blob) for blob in files.values())
    if total > options.total_mb * MB:
        message = f"layer and channel files total {total} bytes, over total_mb={options.total_mb:g}"
        warnings.warn(message, UserWarning, stacklevel=3)


def _read_web(web_dir: Path) -> dict:
    """Read every shipped web file as relative posix path to bytes, skipping maps and the template."""
    return {
        path.relative_to(web_dir).as_posix(): path.read_bytes()
        for path in sorted(web_dir.rglob("*"))
        if path.is_file() and path.suffix != ".md" and path.name != "index.html"
        and "__pycache__" not in path.parts
    }


def _fill_template(template: str, block: str) -> str:
    """Replace the scripts marker of the page template, raising ValueError when it is absent."""
    if MARKER not in template:
        raise ValueError(f"index.html template lacks the marker {MARKER}")
    return template.replace(MARKER, block)


def _write_files(out: Path, blobs: dict) -> None:
    """Write each relative path to bytes mapping under out, creating folders."""
    for rel, blob in blobs.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(blob)


def write_bundle(
    out: Path, graph: dict, payloads: dict, inline_mb: float = 16,
    options: Optional[BundleOptions] = None,
) -> Path:
    """
    Write a bundle as one self contained index.html when it fits, else as separate files.

    Args:
        out: Output folder, created when missing.
        graph: The graph.json dict. A copy gets file, files, channels, channel_list, kept.
        payloads: Node id to one LayerPayload in single mode or two in compare mode.
        inline_mb: Largest index.html, in MB, written as one self contained page.
        options: Size caps and web folder, BundleOptions() when None.

    Returns:
        Path of the written index.html.

    Raises:
        ValueError: When a file exceeds layer_mb, payloads mismatch the graph, or the
            bundle fits inline_mb but a web file is not .js, .css, or .txt, or a module or
            text file cannot inline.
    """
    options = options or BundleOptions()
    graph, files = _fill_graph(graph, payloads)
    _check_sizes(files, options)
    graph_json = json.dumps(graph, ensure_ascii=False, default=_plain_json)
    template = (options.web_dir / "index.html").read_text(encoding="utf-8")
    web = _read_web(options.web_dir)
    out.mkdir(parents=True, exist_ok=True)
    page, limit = "", inline_mb * MB
    # Base64 and escaping only grow the data, so raw data over the limit never fits inline.
    if len(graph_json.encode("utf-8")) + sum(map(len, files.values())) <= limit:
        page = _fill_template(template, inline_block(graph_json, files, web))
    if not page or len(page.encode("utf-8")) > limit:
        _write_files(out, {"graph.json": graph_json.encode("utf-8"), **files, **web})
        links = "".join(f'<link rel="stylesheet" href="{rel}">\n' for rel in web if rel.endswith(".css"))
        page = _fill_template(template, f'{links}<script type="module" src="{ENTRY}"></script>')
    index = out / "index.html"
    index.write_text(page, encoding="utf-8")
    return index
