"""
Frame layer and channel files in the little endian layout format.js decodes.

Purpose:

- Encode one layer's positions, rgb, scores, and joint rgb, or its kept channel values.
- Bound the bytes a layer or channel file adds beyond its data, for the size budget.

Special Notes:

- Pure. Writes no files and imports no bundle, pipeline, or api module.
- Every section starts at a multiple of 4 bytes, so typed arrays view it without copying.
"""

import json
import struct

import numpy as np

from layerviz.reduce import SCORE_COUNT, Reduced
from layerviz.shapes.base import LayerData

# Sections are padded to a multiple of 4, so each adds at most 3 zero bytes.
SECTION_SLACK = 3


def _plain_json(value: object) -> object:
    """Convert a NumPy scalar or array to plain Python for json.dumps, else raise TypeError."""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"cannot write {type(value).__name__} to JSON")


def _frame(header: dict, sections: list[tuple[str, object, str]]) -> bytes:
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


def _check_rows(name: str, array: object, n: int, cols: int | None = None) -> None:
    """Raise ValueError naming the field unless the array has n rows and, if given, cols columns."""
    shape = np.shape(array)
    if shape[0] != n or (cols is not None and (len(shape) != 2 or shape[1] != cols)):
        want = f"[{n}, {cols}]" if cols is not None else f"{n} rows"
        raise ValueError(f"{name} has shape {list(shape)}, expected {want}")


def encode_layer(data: LayerData, reduced: Reduced, rgb_joint: np.ndarray | None = None) -> bytes:
    """
    Encode one layer as framed positions, rgb, scores, and rgb_joint when given.

    Args:
        data: The layer whose positions, kind, grid, scale, and labels are written.
        reduced: The reduction whose uint8 [N, 3] rgb and [N, K] scores are written.
        rgb_joint: Optional uint8 [N, 3] colors from the joint PCA of compare mode.

    Returns:
        The layer file bytes that format.js parseLayer decodes.

    Raises:
        ValueError: When an array or the labels do not have one row per position.
    """
    n = data.positions.shape[0]
    _check_rows("rgb", reduced.rgb, n, 3)
    _check_rows("scores", reduced.scores, n)
    if data.labels is not None:
        _check_rows("labels", data.labels, n)
    header = {
        "n": n, "dim": data.positions.shape[1], "k": reduced.scores.shape[1], "kind": data.kind,
        "grid": data.grid, "scale": data.scale, "labels": data.labels,
    }
    sections: list[tuple[str, object, str]] = [
        ("positions", data.positions, "float32"),
        ("rgb", reduced.rgb, "uint8"),
        ("scores", reduced.scores, "float16"),
    ]
    if rgb_joint is not None:
        _check_rows("rgb_joint", rgb_joint, n, 3)
        sections.append(("rgb_joint", rgb_joint, "uint8"))
    return _frame(header, sections)


def encode_channels(values: np.ndarray, kept: np.ndarray, n: int) -> bytes:
    """
    Encode one layer's kept channel values as a framed channel file.

    Args:
        values: uint8 [N, C] values, one column per kept channel.
        kept: The C indices of the kept channels in the layer.
        n: Positions of the layer, the rows values must have.

    Returns:
        The channel file bytes that format.js parseChannels decodes.

    Raises:
        ValueError: When the values do not have n rows and one column per kept index.
    """
    _check_rows("channel values", values, n, len(kept))
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
    specs = {"values": [0, n * n_channels, "uint8"]}
    return len(_frame_header(header, specs)) + SECTION_SLACK


def layer_overhead(data: LayerData, k: int = SCORE_COUNT) -> int:
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
    return len(_frame_header(header, specs)) + 3 * SECTION_SLACK
