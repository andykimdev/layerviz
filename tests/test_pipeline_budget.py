"""
Test that run_pipeline budgets each layer within the per layer and total size caps.

Purpose:

- Check the kept positions, the channel cap, and the bytes per position option.
- Check that each layer is priced by its own coordinate count, so 6 coordinates still fit.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from layerviz.api import capture
from layerviz.budget import BYTES_PER_POSITION, COMPARE_BYTES_PER_POSITION, MB
from layerviz.framing import encode_channels, encode_layer, layer_overhead
from layerviz.pipeline import CaptureOptions, run_pipeline
from layerviz.reduce import SCORE_COUNT
from tests.models import image2d_model, wide_volume3d_model

# Float32 coordinates, 3 rgb bytes, and SCORE_COUNT float16 scores per kept position.
IMAGE2D_POSITION = 2 * 4 + 3 + SCORE_COUNT * 2
VECTOR_POSITION = 1 * 4 + 3 + SCORE_COUNT * 2
# 20000 points of 6 coordinates take 1.7 MB of positions, over a 1 MB layer cap.
SIX_COORD_POINTS = 20000


class WidePointsNet(nn.Module):
    """
    Map the feats of a (coords, feats) point batch to SCORE_COUNT channels, so all scores are kept.

    """

    def __init__(self) -> None:
        """Create the Linear layer."""
        super().__init__()
        self.wide = nn.Linear(4, SCORE_COUNT)

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return wide(feats), ignoring the coords."""
        _coords, feats = batch
        return self.wide(feats)


def six_coordinate_points_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a (coords [1, 20000, 6], feats [1, 20000, 4]) point batch."""
    torch.manual_seed(0)
    return torch.randn(1, SIX_COORD_POINTS, 6), torch.randn(1, SIX_COORD_POINTS, 4)


def full_overheads(model, batch) -> list[int]:
    """Return the layer file overhead of each of the model's full, untrimmed layers."""
    layers = run_pipeline(model, batch, CaptureOptions()).layers
    return [layer_overhead(layer.data) for layer in layers]


def image2d_layer_mb(positions: int) -> float:
    """Return the layer_mb that holds this many 2D positions past the largest image2d overhead."""
    model, batch = image2d_model()
    return (positions * IMAGE2D_POSITION + max(full_overheads(model, batch))) / MB


def test_points_with_six_coordinates_write_every_layer_file_within_layer_mb(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    model, batch = WidePointsNet().eval(), six_coordinate_points_batch()
    capture(model, batch, tmp_path, layer_mb=1, inline_mb=0)
    sizes = {path.name: path.stat().st_size for path in (tmp_path / "layers").glob("*.bin")}
    assert sorted(sizes) == ["L0002.bin", "L0002.ch.bin"]
    assert max(sizes.values()) <= MB


def test_budget_keeps_the_2d_positions_the_layer_cap_holds() -> None:
    model, batch = image2d_model()
    result = run_pipeline(model, batch, CaptureOptions(layer_mb=image2d_layer_mb(10)))
    assert [(layer.total, layer.kept) for layer in result.layers] == [(256, 10), (64, 10)]


def test_budget_reduces_and_quantizes_only_the_kept_rows() -> None:
    model, batch = image2d_model()
    first = run_pipeline(model, batch, CaptureOptions(layer_mb=image2d_layer_mb(10))).layers[0]
    shapes = [first.data.positions.shape, first.data.features.shape, first.reduced.rgb.shape]
    assert shapes == [(10, 2), (10, 4), (10, 3)]
    assert (first.channels[0].dtype, first.channels[0].shape) == (np.uint8, (10, 4))
    np.testing.assert_array_equal(first.channels[1], np.arange(4))


def test_layer_files_and_their_overheads_fit_total_mb() -> None:
    # 2 x 20 positions of 75 bytes plus both overheads, so each layer keeps 20.
    model, batch = image2d_model()
    total_mb = (2 * 20 * IMAGE2D_POSITION + sum(full_overheads(model, batch))) / MB
    result = run_pipeline(model, batch, CaptureOptions(total_mb=total_mb))
    assert [layer.kept for layer in result.layers] == [20, 20]


def test_vector_layer_keeps_the_positions_of_its_1d_size_under_the_cap() -> None:
    torch.manual_seed(0)
    model, batch = nn.Sequential(nn.Linear(10, 100)), torch.randn(64, 10)
    layer = run_pipeline(model, batch, CaptureOptions(layer_mb=4096 / MB)).layers[0]
    assert layer.kept == (4096 - full_overheads(model, batch)[0]) // VECTOR_POSITION


def test_channel_file_with_its_header_fits_the_layer_cap() -> None:
    torch.manual_seed(0)
    model, batch = nn.Sequential(nn.Linear(10, 100)), torch.randn(64, 10)
    layer = run_pipeline(model, batch, CaptureOptions(layer_mb=4096 / MB)).layers[0]
    size = len(encode_channels(*layer.channels, layer.kept))
    assert (size <= 4096, 0 < len(layer.channels[1]) < 100) == (True, True)


def test_layer_trimmed_to_the_cap_encodes_within_layer_mb() -> None:
    model, batch = wide_volume3d_model()
    layer = run_pipeline(model, batch, CaptureOptions(layer_mb=4096 / MB)).layers[-1]
    size = len(encode_layer(layer.data, layer.reduced))
    assert (layer.kept < layer.total, size <= 4096) == (True, True)


def test_channels_under_the_layer_cap_are_all_kept() -> None:
    torch.manual_seed(0)
    model, batch = nn.Sequential(nn.Linear(10, 6)), torch.randn(4, 10)
    data, kept = run_pipeline(model, batch, CaptureOptions(layer_mb=1)).layers[0].channels
    assert (data.shape, kept.tolist()) == ((4, 6), list(range(6)))


@pytest.mark.parametrize(
    ("bytes_per_position", "kept"),
    [(None, 31), (BYTES_PER_POSITION, 31), (COMPARE_BYTES_PER_POSITION, 30)],
)
def test_bytes_per_position_option_sets_the_kept_positions(
    bytes_per_position: int | None, kept: int
) -> None:
    # 2360 bytes past the file overhead hold 31 2D positions of 75 bytes but 30 of 78.
    model, batch = image2d_model()
    layer_mb = (2360 + max(full_overheads(model, batch))) / MB
    opts = CaptureOptions(layer_mb=layer_mb, bytes_per_position=bytes_per_position)
    assert [layer.kept for layer in run_pipeline(model, batch, opts).layers] == [kept, kept]


def test_layer_cap_below_one_compare_position_names_its_size() -> None:
    model, batch = image2d_model()
    opts = CaptureOptions(layer_mb=80 / MB, bytes_per_position=COMPARE_BYTES_PER_POSITION)
    with pytest.raises(ValueError, match=r"at least 7\.82e-05 MB"):
        run_pipeline(model, batch, opts)


def test_layer_cap_below_one_position_raises() -> None:
    model, batch = image2d_model()
    with pytest.raises(ValueError, match=r"layer_mb=3\.81"):
        run_pipeline(model, batch, CaptureOptions(layer_mb=40 / MB))
