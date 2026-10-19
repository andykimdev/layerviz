"""
Test that total_mb bounds the layer files and the channel files together.

Purpose:

- Check that a wide model fits total_mb in single mode and as a whole compare bundle.
- Check that a squeezed channel file keeps its highest variance channels.
- Check that a model whose channel files fit keeps every channel.
- Check the byte sharing across channel files and the empty cap that writes no file.

Special Notes:

- Bundles are written as folders with a fake web folder, so tests read the layer and
  channel files and graph.json directly.
"""

import json
import warnings

import numpy as np
import pytest
import torch
from torch import nn

from layerviz.api import capture
from layerviz.budget import MB, plan_channel_caps
from layerviz.bundle import channel_overhead
from layerviz.comparison import CompareOptions, compare
from layerviz.pipeline import CaptureOptions, captured_layer, with_channels
from layerviz.shapes.base import LayerData
from tests.models import as_eval_pair, image2d_model
from tests.test_bundle import read_frame

TOTAL_MB = 0.5
# Holds all 2 x 4096 positions with about 400 KB left, short of 1 MB of full channel files.
SQUEEZE_MB = 1.0


def make_web(root):
    """Write a minimal web folder with the page template and an entry module."""
    web = root / "web"
    web.mkdir()
    (web / "index.html").write_text("<html><body><!-- layerviz:scripts --></body></html>")
    (web / "app.js").write_text("console.log(1);\n")
    return web


def wide_model(seed: int = 0):
    """Build two 128 channel Conv2d layers on a 64x64 input, 4096 positions each."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Conv2d(3, 128, 3, padding=1), nn.Conv2d(128, 128, 3, padding=1))
    return as_eval_pair(model, torch.randn(1, 3, 64, 64))


def bundle_bytes(out):
    """Return the total size of every layer and channel file of a bundle."""
    return sum(path.stat().st_size for path in (out / "layers").iterdir())


def captured_nodes(out):
    """Return the graph.json nodes that have a layer file."""
    graph = json.loads((out / "graph.json").read_text())
    return [node for node in graph["nodes"] if node["file"]]


def test_wide_capture_fits_total_mb_counting_channel_files(tmp_path, recwarn):
    model, batch = wide_model()
    out = tmp_path / "out"

    capture(model, batch, out, web_dir=make_web(tmp_path), inline_mb=0, total_mb=TOTAL_MB)

    assert bundle_bytes(out) <= TOTAL_MB * MB
    assert not [w for w in recwarn if "total_mb" in str(w.message)]
    nodes = captured_nodes(out)
    # Positions keep priority, so the 3318 kept of 4096 match the cap without channels.
    assert [node["kept"] for node in nodes] == [3318, 3318]
    assert [len(node["channel_list"]) for node in nodes] == [3, 3]


def test_wide_compare_fits_total_mb_for_the_whole_bundle(tmp_path):
    model_a, batch = wide_model(0)
    model_b, _ = wide_model(1)
    out = tmp_path / "out"
    capture_opts = CaptureOptions(total_mb=TOTAL_MB)
    opts = CompareOptions(inline_mb=0, web_dir=make_web(tmp_path), capture=capture_opts)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compare(model_a, model_b, batch, out, opts)

    assert bundle_bytes(out) <= TOTAL_MB * MB
    nodes = captured_nodes(out)
    # Each pane budgets half of total_mb, so 1598 of 4096 positions per layer and pane.
    assert [(node["kept"], node["kept_b"]) for node in nodes] == [(1598, 1598)] * 2
    assert [len(node["channel_list"]) for node in nodes] == [3, 3]


def test_squeezed_channel_file_keeps_its_highest_variance_channels(tmp_path):
    model, batch = wide_model()
    out = tmp_path / "out"

    capture(model, batch, out, web_dir=make_web(tmp_path), inline_mb=0, total_mb=SQUEEZE_MB)

    assert bundle_bytes(out) <= SQUEEZE_MB * MB
    nodes = captured_nodes(out)
    assert [node["kept"] for node in nodes] == [4096, 4096]
    node = nodes[-1]
    header, _, _ = read_frame((out / node["channels"]).read_bytes())
    kept = header["channels"]
    assert kept == node["channel_list"] == sorted(kept)
    assert 0 < len(kept) < 128
    with torch.no_grad():
        features = model(batch)[0].reshape(128, -1).T.numpy()
    variance = features.var(axis=0)
    dropped = np.setdiff1d(np.arange(128), kept)
    assert variance[kept].min() >= variance[dropped].max()


def test_channel_files_that_fit_keep_every_channel(tmp_path):
    model, batch = image2d_model()
    out = tmp_path / "out"

    capture(model, batch, out, web_dir=make_web(tmp_path), inline_mb=0)

    widths = {4, 6}
    lists = [node["channel_list"] for node in captured_nodes(out) if node["channels"]]
    assert sorted(len(channels) for channels in lists) == sorted(widths)
    assert all(channels == list(range(len(channels))) for channels in lists)


def test_channel_caps_water_fill_what_the_layer_files_leave():
    need = {"small": 100, "big": 5000, "huge": 9000}
    # 4000 spare bytes: small keeps its 100, and big and huge split the other 3900.
    caps = plan_channel_caps(need, layer_mb=3000 / MB, total_mb=10000 / MB, used=6000)
    assert caps == {"small": 100, "big": 1950, "huge": 1950}
    capped = plan_channel_caps(need, layer_mb=1500 / MB, total_mb=10000 / MB, used=6000)
    assert capped == {"small": 100, "big": 1500, "huge": 1500}


def test_channel_caps_are_zero_when_layer_files_fill_total_mb():
    caps = plan_channel_caps({"a": 100, "b": 200}, layer_mb=1, total_mb=500 / MB, used=600)
    assert caps == {"a": 0, "b": 0}


def test_cap_below_one_channel_writes_no_channel_file():
    # Column 1 has the largest spread, so it is the one channel kept.
    features = (np.arange(10)[:, None] * np.array([1.0, 4.0, 2.0, 3.0])).astype(np.float32)
    positions = np.zeros((10, 3), dtype=np.float32)
    data = LayerData("points", positions, features, grid=None, scale=None, labels=None)
    layer = captured_layer(None, data, 10, seed=0)
    floor = channel_overhead(10, 4)

    assert with_channels(layer, floor + 9).channels is None
    values, kept = with_channels(layer, floor + 10).channels
    assert values.shape == (10, 1) and list(kept) == [1]
