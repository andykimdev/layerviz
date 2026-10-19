"""
Test the framing module against format.js and its overhead bounds, and the capture options.

Purpose:

- Check that every section of a layer file starts at a multiple of 4 and decodes in node.
- Check that layer_overhead and channel_overhead cover the header and padding of real files.

Special Notes:

- read_frame, run_node, and node_decode come from tests.models and tests.harness.
"""

import numpy as np
import pytest

from layerviz.framing import channel_overhead, encode_channels, encode_layer, layer_overhead
from layerviz.options import CaptureOptions
from layerviz.reduce import reduce_layer
from layerviz.shapes.base import LayerData
from tests.harness import node_decode
from tests.models import read_frame


def make_layer(n, dim=1, k=1):
    """Build a deterministic sequence layer of n positions, its reduction, and a joint rgb."""
    rng = np.random.default_rng(n)
    data = LayerData(
        kind="sequence", positions=rng.normal(size=(n, dim)).astype(np.float32),
        features=rng.normal(size=(n, 6)).astype(np.float32), grid=None, scale=None, labels=None,
    )
    joint = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    return data, reduce_layer(data.features, k=k), joint


def padding_of(header):
    """Return the zero bytes the file adds after each section to reach a multiple of 4."""
    return sum(-size % 4 for _, size, _ in header["sections"].values())


def test_encode_layer_starts_every_section_at_a_multiple_of_four():
    data, reduced, joint = make_layer(5, k=3)
    header, payload, length = read_frame(encode_layer(data, reduced, joint))
    sections = header["sections"]
    assert sections["rgb"][1] == 15
    # Scores of 30 bytes put rgb_joint at 2 mod 4 if sections were padded only to 2.
    assert sections["scores"][1] == 30
    assert [(4 + length + offset) % 4 for offset, _, _ in sections.values()] == [0, 0, 0, 0]
    assert sections == {
        "positions": [0, 20, "float32"], "rgb": [20, 15, "uint8"],
        "scores": [36, 30, "float16"], "rgb_joint": [68, 15, "uint8"],
    }
    assert len(payload) == 84


@pytest.mark.parametrize("label", ["a", "ab", "abc", "é"])
def test_encode_layer_pads_the_header_to_a_multiple_of_four(label):
    data, reduced, _ = make_layer(2)
    data.labels = [label, label]
    blob = encode_layer(data, reduced)
    header, payload, length = read_frame(blob)
    assert (4 + length) % 4 == 0
    assert header["labels"] == [label, label]
    assert len(payload) == 8 + 8 + 4


def test_encode_layer_round_trips_through_format_js(tmp_path):
    data, reduced, joint = make_layer(5, k=3)
    record = node_decode(tmp_path, encode_layer(data, reduced, joint), "parseLayer")
    header = record["header"]
    assert (header["n"], header["dim"], header["k"], header["kind"]) == (5, 1, 3, "sequence")
    assert (header["grid"], header["scale"], header["labels"]) == (None, None, None)
    np.testing.assert_array_equal(np.float32(record["positions"]), data.positions.ravel())
    np.testing.assert_array_equal(record["rgb"], reduced.rgb.ravel())
    expected_scores = reduced.scores.astype(np.float32).ravel()
    np.testing.assert_array_equal(np.float32(record["scores"]), expected_scores)
    assert record["rgbJoint"] == joint.ravel().tolist()


def test_encode_layer_without_joint_omits_its_section():
    data, reduced, _ = make_layer(4, k=2)
    header, _, _ = read_frame(encode_layer(data, reduced))
    assert list(header["sections"]) == ["positions", "rgb", "scores"]


def test_encode_layer_rejects_rgb_not_one_row_per_position():
    data, reduced, _ = make_layer(4, k=2)
    with pytest.raises(ValueError, match=r"rgb has shape \[4, 3\], expected \[5, 3\]"):
        encode_layer(LayerData(**{**vars(data), "positions": np.zeros((5, 1))}), reduced)


@pytest.mark.parametrize("n", range(1, 9))
def test_layer_overhead_covers_header_and_padding_of_the_file(n):
    data, reduced, joint = make_layer(n)
    blob = encode_layer(data, reduced, joint)
    header, payload, length = read_frame(blob)
    raw = sum(size for _, size, _ in header["sections"].values())
    assert len(payload) == raw + padding_of(header)
    assert layer_overhead(data, k=reduced.scores.shape[1]) >= 4 + length + padding_of(header)


@pytest.mark.parametrize("n", range(1, 9))
def test_channel_overhead_covers_header_and_padding_of_the_file(n):
    values = np.arange(n * 3, dtype=np.uint8).reshape(n, 3)
    blob = encode_channels(values, np.array([0, 1, 2]), n)
    header, payload, length = read_frame(blob)
    assert len(payload) == n * 3 + padding_of(header)
    assert channel_overhead(n, 3) >= 4 + length + padding_of(header)


def test_encode_channels_round_trips_through_format_js(tmp_path):
    values = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint8)
    record = node_decode(tmp_path, encode_channels(values, np.array([4, 1]), 3), "parseChannels")
    assert (record["header"]["n"], record["header"]["c"], record["header"]["channels"]) == (
        3, 2, [4, 1],
    )
    assert record["values"] == [1, 2, 3, 4, 5, 6]


def test_encode_channels_rejects_values_not_matching_the_indices():
    with pytest.raises(ValueError, match=r"channel values has shape \[3, 2\], expected \[3, 3\]"):
        encode_channels(np.zeros((3, 2), dtype=np.uint8), np.array([0, 1, 2]), 3)


def test_capture_options_leave_bytes_per_position_to_the_mode():
    options = CaptureOptions()
    assert options.bytes_per_position is None
    assert (options.sample_index, options.layer_mb, options.total_mb, options.seed) == (
        0, 16, 1024, 0,
    )
    assert (options.include, options.exclude, options.shapes, options.forward) == (
        None, None, None, None,
    )
