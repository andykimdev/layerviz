"""
Capture two models on one batch and write one comparison bundle of their paired layers.

Purpose:

- Pair the captured layers of two models by module name and list the unpaired names.
- Color pane B aligned to pane A, fit joint colors for both, and score each pair by CKA.

Special Notes:

- Everything but compare is pure, so graph and payload building run without files.
- The graph follows model A. A layer only model A has stays in the graph without panes.
- Panes keep only the positions both layers share, in model A's order.
- Layers of different widths have no joint colors. Their node says joint false and their
  rgb_joint sections hold black placeholders, since the compare format needs the section.
"""

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from torch import nn

from layerviz.align import aligned_rgb, joint_rgb, linear_cka
from layerviz.budget import COMPARE_BYTES_PER_POSITION
from layerviz.bundle import WEB_DIR, BundleOptions, LayerPayload, write_bundle
from layerviz.graph import annotate_nodes, graph_body, graph_meta
from layerviz.pipeline import CaptureOptions, CapturedLayer, PipelineResult, run_pipeline
from layerviz.reduce import standardize

MIN_SHARED_ROWS = 2
NO_SHARED = "no positions shared between the two models"


class Pair(NamedTuple):
    """
    Hold both panes of one paired layer, their CKA, and whether joint colors were fit.

    """

    pane_a: LayerPayload
    pane_b: LayerPayload
    cka: float
    joint: bool


@dataclass(frozen=True)
class CompareOptions:
    """
    Choose the pane labels, the capture options both models share, and the bundle output.

    """

    labels: tuple[str, str] = ("A", "B")
    capture: CaptureOptions = field(default_factory=CaptureOptions)
    inline_mb: float = 16
    web_dir: Path | None = None

    def __post_init__(self) -> None:
        """Raise ValueError when labels does not hold exactly two names."""
        if len(self.labels) != 2:
            raise ValueError(f"compare needs two labels, got {list(self.labels)}")


def shared_rows(pos_a: np.ndarray, pos_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the rows of two position arrays that hold the same coordinates.

    Args:
        pos_a: Model A positions of shape [Na, D].
        pos_b: Model B positions of shape [Nb, D'].

    Returns:
        Row indices into pos_a and into pos_b, matched pairwise, in pos_a order.
    """
    rows_b = {row.tobytes(): i for i, row in enumerate(np.asarray(pos_b, dtype=np.float32))}
    pairs = [
        (i, rows_b[row.tobytes()])
        for i, row in enumerate(np.asarray(pos_a, dtype=np.float32))
        if row.tobytes() in rows_b
    ]
    rows = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    return rows[:, 0], rows[:, 1]


def take_rows(layer: CapturedLayer, rows: np.ndarray) -> LayerPayload:
    """
    Cut one captured layer down to the given rows as a pane without joint colors.

    Args:
        layer: The captured layer.
        rows: Row indices to keep, in pane order.

    Returns:
        LayerPayload whose positions, features, labels, colors, scores, and channels follow rows.
    """
    data = layer.data
    labels = None if data.labels is None else [data.labels[i] for i in rows]
    data = replace(
        data, positions=data.positions[rows], features=data.features[rows], labels=labels
    )
    reduced = replace(layer.reduced, rgb=layer.reduced.rgb[rows], scores=layer.reduced.scores[rows])
    channels = None if layer.channels is None else (layer.channels[0][rows], layer.channels[1])
    return LayerPayload(data=data, reduced=reduced, channels=channels)


def joint_colors(
    z_a: np.ndarray, z_b: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Fit joint colors of both panes, or return None when the layers differ in width.

    Args:
        z_a: Model A standardized features [N, Ca].
        z_b: Model B standardized features [N, Cb].
        seed: Seed of the joint PCA fit.

    Returns:
        Joint colors of pane A and pane B, each uint8 [N, 3], or None when Ca differs from Cb.

    Notes:
        A joint PCA needs one channel space, which layers of different widths lack.
    """
    if z_a.shape[1] != z_b.shape[1]:
        return None
    return joint_rgb(z_a, z_b, seed=seed)


def pair_payloads(layer_a: CapturedLayer, layer_b: CapturedLayer, seed: int) -> Pair | None:
    """
    Build both panes of one paired layer and their linear CKA over the shared positions.

    Args:
        layer_a: Model A captured layer.
        layer_b: Model B captured layer with the same module name.
        seed: Seed of the joint PCA fit.

    Returns:
        Pair with pane B colored to align with pane A, or None when fewer than 2 positions
        match. Without joint colors, joint is False and both rgb_joint arrays are zeros.
    """
    rows_a, rows_b = shared_rows(layer_a.data.positions, layer_b.data.positions)
    if len(rows_a) < MIN_SHARED_ROWS:
        return None
    pane_a, pane_b = take_rows(layer_a, rows_a), take_rows(layer_b, rows_b)
    pane_b.reduced = replace(pane_b.reduced, rgb=aligned_rgb(pane_a.reduced, pane_b.reduced))
    z_a = standardize(pane_a.data.features)[0]
    z_b = standardize(pane_b.data.features)[0]
    joint = joint_colors(z_a, z_b, seed)
    # Black placeholders keep the format valid without passing off own colors as joint ones.
    blank = (np.zeros_like(pane_a.reduced.rgb), np.zeros_like(pane_b.reduced.rgb))
    pane_a.rgb_joint, pane_b.rgb_joint = blank if joint is None else joint
    return Pair(pane_a, pane_b, linear_cka(z_a, z_b), joint is not None)


def unpaired_names(names_a: list[str], names_b: list[str]) -> dict[str, list[str]]:
    """
    List the captured layer names that only one model has, in execution order.

    Args:
        names_a: Model A captured layer names.
        names_b: Model B captured layer names.

    Returns:
        Dict with key a for names only in A and key b for names only in B.
    """
    set_a, set_b = set(names_a), set(names_b)
    return {
        "a": [name for name in names_a if name not in set_b],
        "b": [name for name in names_b if name not in set_a],
    }


def pair_nodes(
    by_name: dict[str, dict[str, Any]], layers_a: list[CapturedLayer],
    layers_b: dict[str, CapturedLayer], seed: int,
) -> dict[str, list[LayerPayload]]:
    """
    Fill the pair fields of every layer both models captured and build its two panes.

    Args:
        by_name: Model A graph nodes by module name, changed in place.
        layers_a: Model A captured layers in execution order.
        layers_b: Model B captured layers by module name.
        seed: Seed of the joint PCA fits.

    Returns:
        Node id to [pane A, pane B] for every pair with at least MIN_SHARED_ROWS shared
        positions. A pair with fewer gets the NO_SHARED skip reason and no panes.
    """
    payloads = {}
    for layer in layers_a:
        name = layer.node.name
        if name not in layers_b:
            continue
        pair = pair_payloads(layer, layers_b[name], seed)
        node = by_name[name]
        if pair is None:
            node["skipped"] = NO_SHARED
            continue
        payloads[node["id"]] = [pair.pane_a, pair.pane_b]
        node.update(cka=pair.cka, total_b=int(layers_b[name].total), joint=pair.joint)
    return payloads


def compare_graph(
    result_a: PipelineResult, result_b: PipelineResult, labels: tuple[str, str], seed: int,
    sample_index: int = 0,
) -> tuple[dict[str, Any], dict[str, list[LayerPayload]]]:
    """
    Build the comparison graph and the two panes of every paired layer.

    Args:
        result_a: Model A pipeline result, which the graph follows.
        result_b: Model B pipeline result.
        labels: Display names of model A and model B.
        seed: Seed of the joint PCA fits.
        sample_index: The sample of the batch that was captured.

    Returns:
        (graph.json dict in compare mode, node id to [pane A, pane B]). Paired nodes carry
        cka, total_b, and joint, which is False when the pair has no joint colors.

    Notes:
        Nodes, boxes, edges, and meta come from layerviz.graph, so they equal the capture
        graph of model A apart from the pair fields and the labels, mode, and unpaired meta.
    """
    body = graph_body(result_a.trace, {layer.node.name: layer for layer in result_a.layers})
    annotate_nodes(body["nodes"], result_a)
    layers_b = {layer.node.name: layer for layer in result_b.layers}
    unpaired = unpaired_names([layer.node.name for layer in result_a.layers], list(layers_b))
    by_name = {node["name"]: node for node in body["nodes"]}
    for name in unpaired["a"]:
        by_name[name]["skipped"] = f"only in {labels[0]}"
    payloads = pair_nodes(by_name, result_a.layers, layers_b, seed)
    meta = graph_meta(result_a, list(labels), None, None, sample_index)
    meta["unpaired"] = unpaired
    return {"meta": meta, **body}, payloads


def compare(
    model_a: nn.Module, model_b: nn.Module, batch: Any, out: Path,
    options: CompareOptions | None = None,
) -> Path:
    """
    Capture two models on one batch and write one bundle of their layers paired by name.

    Args:
        model_a: The model the graph follows.
        model_b: The model compared against it.
        batch: The input both models run on. It is left unchanged.
        out: Output folder.
        options: Labels, shared capture options, inline_mb, and web_dir. Defaults when None.

    Returns:
        Path of the written index.html.

    Notes:
        Each model runs on its own deep copy of batch, so an in place first module of model
        A cannot change what model B sees. Both captures budget COMPARE_BYTES_PER_POSITION,
        whatever bytes_per_position the capture options hold, since every pane also stores
        joint RGB. Each pane budgets half of total_mb, so the whole bundle fits total_mb.
    """
    opts = options or CompareOptions()
    capture = opts.capture
    pane = replace(
        capture, bytes_per_position=COMPARE_BYTES_PER_POSITION, total_mb=capture.total_mb / 2
    )
    result_a = run_pipeline(model_a, copy.deepcopy(batch), pane)
    result_b = run_pipeline(model_b, copy.deepcopy(batch), pane)
    graph, payloads = compare_graph(
        result_a, result_b, opts.labels, capture.seed, capture.sample_index
    )
    # The bundle check measures both panes together against the full total_mb.
    bundle = BundleOptions(capture.layer_mb, capture.total_mb, opts.web_dir or WEB_DIR)
    return write_bundle(Path(out), graph, payloads, opts.inline_mb, bundle)
