"""
Test the joint colors, the pane alignment, and the byte budget of the comparison bundle.

Purpose:

- Check that joint RGB comes from one PCA fit over both models, against a NumPy reference.
- Check that pane b colors follow pane a's components, against a NumPy and SciPy reference.
- Check that layers of different widths are flagged as having no joint colors.
- Check that compare budgets the joint RGB with the compare bytes per position.

Special Notes:

- References use plain NumPy and SciPy, never the align or reduce functions under test.
- Bundles are written as folders with a fake web folder, so tests read the layer files.
"""

import copy

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from layerviz.budget import BYTES_PER_POSITION, COMPARE_BYTES_PER_POSITION, MB
from layerviz.comparison import pair_payloads
from layerviz.pipeline import CaptureOptions
from tests.harness import (
    captured, expect_paired, other_model, pane_sections, percentile_bytes, run_compare,
    zscore,
)
from tests.models import image2d_model

BYTE_TOLERANCE = 2


def byte_gap(got, expected):
    """Return the largest byte difference to the expected column, or to its inversion."""
    return min(np.abs(got - expected).max(), np.abs(got - (255 - expected)).max())


def reference_joint_bytes(features_a, features_b):
    """Color the stacked standardized features of both models by one SVD with plain NumPy."""
    stacked = np.concatenate([zscore(features_a), zscore(features_b)])
    _, _, vt = np.linalg.svd(stacked, full_matrices=False)
    return percentile_bytes(stacked @ vt[:3].T)


def reference_aligned_bytes(scores_a, scores_b):
    """Match b's score columns to a's by Hungarian assignment and color the first three."""
    scores_a, scores_b = scores_a.astype(np.float64), scores_b.astype(np.float64)
    k = min(scores_a.shape[1], scores_b.shape[1])
    corr = zscore(scores_a[:, :k]).T @ zscore(scores_b[:, :k]) / len(scores_a)
    _, perm = linear_sum_assignment(-np.abs(corr))
    signs = np.where(corr[np.arange(k), perm] < 0, -1.0, 1.0)
    return percentile_bytes(scores_b[:, perm[:3]] * signs[:3])


def test_joint_rgb_comes_from_one_fit_over_both_models(tmp_path):
    model, batch = image2d_model()
    other = other_model()
    graph, out = run_compare(tmp_path, model, other, batch)
    layers_a, layers_b = captured(model, batch), captured(other, batch)
    for node in expect_paired(graph, ["0", "1"]):
        expected = reference_joint_bytes(
            layers_a[node["name"]].data.features, layers_b[node["name"]].data.features
        )
        pane_a, pane_b = pane_sections(out, node)
        got = np.concatenate([pane_a["rgb_joint"], pane_b["rgb_joint"]]).reshape(-1, 3)
        gaps = [byte_gap(got[:, j].astype(int), expected[:, j]) for j in range(3)]
        assert max(gaps) <= BYTE_TOLERANCE


def test_pane_b_colors_align_to_pane_a(tmp_path):
    model, batch = image2d_model()
    other = other_model()
    graph, out = run_compare(tmp_path, model, other, batch)
    layers_a, layers_b = captured(model, batch), captured(other, batch)
    for node in expect_paired(graph, ["0", "1"]):
        expected = reference_aligned_bytes(
            layers_a[node["name"]].reduced.scores, layers_b[node["name"]].reduced.scores
        )
        got = pane_sections(out, node)[1]["rgb"].reshape(-1, 3).astype(int)
        # Sign matching is part of the alignment, so an inverted column is a failure here.
        assert np.abs(got - expected).max() <= BYTE_TOLERANCE


def test_pair_with_different_widths_has_no_joint_colors():
    model, batch = image2d_model()
    layer_a, layer_b = captured(model, batch)["1"], captured(other_model(width=7), batch)["1"]
    pair = pair_payloads(layer_a, layer_b, seed=0)
    blank = np.zeros((layer_a.data.positions.shape[0], 3), dtype=np.uint8)
    assert pair.joint is False
    np.testing.assert_array_equal(pair.pane_a.rgb_joint, blank)
    np.testing.assert_array_equal(pair.pane_b.rgb_joint, blank)


def test_joint_flag_marks_equal_widths_true_and_differing_widths_false(tmp_path):
    model, batch = image2d_model()
    graph, out = run_compare(tmp_path, model, other_model(width=7), batch)
    nodes = expect_paired(graph, ["0", "1"])
    assert {node["name"]: node["joint"] for node in nodes} == {"0": True, "1": False}
    assert all(not pane["rgb_joint"].any() for pane in pane_sections(out, nodes[1]))


@pytest.mark.parametrize("bytes_per_position", [BYTES_PER_POSITION, COMPARE_BYTES_PER_POSITION])
def test_compare_budgets_joint_rgb_whatever_bytes_per_position_is_passed(
    tmp_path, bytes_per_position
):
    model, batch = image2d_model()
    capture = CaptureOptions(layer_mb=8000 / MB, bytes_per_position=bytes_per_position)
    graph, _ = run_compare(tmp_path, model, copy.deepcopy(model), batch, capture=capture)
    nodes = expect_paired(graph, ["0", "1"])
    # A 2D position with joint rgb costs 78 bytes, so 8000 bytes less the header keep 99 of
    # layer 0's 256 positions. Single mode pricing at 74 bytes would keep more.
    assert {node["name"]: node["kept"] for node in nodes} == {"0": 99, "1": 64}
