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

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np
from torch import nn

from layerviz import __version__
from layerviz.align import aligned_rgb, joint_rgb, linear_cka
from layerviz.budget import COMPARE_BYTES_PER_POSITION
from layerviz.bundle import WEB_DIR, BundleOptions, LayerPayload, write_bundle
from layerviz.layout import ancestors, layout_graph
from layerviz.pipeline import CaptureOptions, CapturedLayer, PipelineResult, run_pipeline
from layerviz.reduce import standardize
from layerviz.trace import INPUT, Trace

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
    web_dir: Optional[Path] = None

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
    data = replace(data, positions=data.positions[rows], features=data.features[rows], labels=labels)
    reduced = replace(layer.reduced, rgb=layer.reduced.rgb[rows], scores=layer.reduced.scores[rows])
    channels = None if layer.channels is None else (layer.channels[0][rows], layer.channels[1])
    return LayerPayload(data=data, reduced=reduced, channels=channels)


def joint_colors(
    z_a: np.ndarray, z_b: np.ndarray, seed: int
) -> Optional[tuple[np.ndarray, np.ndarray]]:
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


def pair_payloads(layer_a: CapturedLayer, layer_b: CapturedLayer, seed: int) -> Optional[Pair]:
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


def layout_leaves(trace: Trace) -> list[str]:
    """
    List the nodes drawn as graph leaves, the input first, in execution order.

    Args:
        trace: The traced forward pass of model A.

    Returns:
        Names of the input and every module with no executed child.
    """
    return [node.name for node in trace.nodes if not trace.children.get(node.name)]


def leaf_edges(trace: Trace, leaves: list[str]) -> list[tuple[str, str]]:
    """
    Project the trace edges onto leaves, a container leaving from its last leaf.

    Args:
        trace: The traced forward pass of model A.
        leaves: The leaf names in execution order.

    Returns:
        Unique (source, target) leaf pairs in trace order. An edge into a container enters
        its first leaf.
    """
    parents = {node.name: node.parent for node in trace.nodes}
    first: dict[str, str] = {}
    last: dict[str, str] = {}
    for leaf in leaves:
        for container in ancestors(leaf, parents):
            first.setdefault(container, leaf)
            last[container] = leaf
    edges: dict[tuple[str, str], None] = {}
    for source, target in trace.edges:
        pair = (last.get(source, source), first.get(target, target))
        if pair[0] in leaves and pair[1] in leaves:
            edges[pair] = None
    return list(edges)


def node_id(index: int) -> str:
    """Return the graph id of the node with the given execution index."""
    return f"L{index:04d}"


def layer_fields(layer: Optional[CapturedLayer]) -> dict[str, Any]:
    """
    Describe one model A layer for its graph node, with nulls when it has none.

    Args:
        layer: The captured layer, or None.

    Returns:
        Dict with kind, explained padded to 3 floats, retained, and total.
    """
    if layer is None:
        return {"kind": None, "explained": None, "retained": None, "total": None}
    explained = [float(v) for v in layer.reduced.explained[:3]]
    return {
        "kind": layer.data.kind,
        "explained": explained + [0.0] * (3 - len(explained)),
        "retained": float(layer.reduced.retained),
        "total": int(layer.total),
    }


def graph_nodes(result: PipelineResult, points: dict[str, tuple[float, float]]) -> list[dict]:
    """
    Build one node per traced module of model A, without pair fields.

    Args:
        result: Model A pipeline result.
        points: Name to (x, y) for leaves and box centers for containers.

    Returns:
        Node dicts in execution order with ids, layer fields, and coordinates.
    """
    ids = {node.name: node_id(node.index) for node in result.trace.nodes}
    layers = {layer.node.name: layer for layer in result.layers}
    nodes = []
    for node in result.trace.nodes:
        entry = {
            "id": ids[node.name], "name": node.name, "type": node.type, "parent": node.parent,
            "index": node.index, "leaf": not result.trace.children.get(node.name),
            "shape": node.output_shape, "kept": None,
            "shares": ids.get(node.shares), "skipped": result.skipped.get(node.name),
            "cka": None, **layer_fields(layers.get(node.name)),
        }
        if node.name == INPUT and result.input is not None:
            entry["kind"] = result.input.kind
        entry["x"], entry["y"] = points.get(node.name, (0.0, 0.0))
        nodes.append(entry)
    return nodes


def graph_layout(trace: Trace) -> tuple[dict, list[dict], list[tuple[str, str]]]:
    """
    Place model A's leaves and boxes and project its edges onto leaves.

    Args:
        trace: The traced forward pass of model A.

    Returns:
        (name to (x, y), box dicts outer first, leaf edges by name).
    """
    leaves = layout_leaves(trace)
    edges = leaf_edges(trace, leaves)
    placed, boxes = layout_graph(leaves, edges, {node.name: node.parent for node in trace.nodes})
    points = {box.name: ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2) for box in boxes}
    points.update({leaf.name: (leaf.x, leaf.y) for leaf in placed})
    box_dicts = [
        {"name": b.name, "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1, "depth": b.depth}
        for b in boxes
    ]
    return points, box_dicts, edges


def compare_graph(
    result_a: PipelineResult, result_b: PipelineResult, labels: tuple[str, str], seed: int
) -> tuple[dict, dict[str, list[LayerPayload]]]:
    """
    Build the comparison graph and the two panes of every paired layer.

    Args:
        result_a: Model A pipeline result, which the graph follows.
        result_b: Model B pipeline result.
        labels: Display names of model A and model B.
        seed: Seed of the joint PCA fits.

    Returns:
        (graph.json dict in compare mode, node id to [pane A, pane B]). Paired nodes carry
        cka, total_b, and joint, which is False when the pair has no joint colors.
    """
    points, boxes, edges = graph_layout(result_a.trace)
    nodes = graph_nodes(result_a, points)
    ids = {node["name"]: node["id"] for node in nodes}
    layers_b = {layer.node.name: layer for layer in result_b.layers}
    unpaired = unpaired_names([layer.node.name for layer in result_a.layers], list(layers_b))
    by_name = {node["name"]: node for node in nodes}
    for name in unpaired["a"]:
        by_name[name]["skipped"] = f"only in {labels[0]}"
    payloads = {}
    for layer in result_a.layers:
        name = layer.node.name
        pair = pair_payloads(layer, layers_b[name], seed) if name in layers_b else None
        if name in layers_b and pair is None:
            by_name[name]["skipped"] = NO_SHARED
        if pair is not None:
            payloads[ids[name]] = [pair.pane_a, pair.pane_b]
            by_name[name].update(
                cka=pair.cka, total_b=int(layers_b[name].total), joint=pair.joint
            )
    meta = {
        "model": f"{result_a.model_name} vs {result_b.model_name}", "step": None,
        "sample_index": None, "label": None, "labels": list(labels), "mode": "compare",
        "version": __version__, "unpaired": unpaired,
    }
    graph = {
        "meta": meta, "input": ids[INPUT], "nodes": nodes, "boxes": boxes,
        "edges": [[ids[source], ids[target]] for source, target in edges],
    }
    return graph, payloads


def compare(
    model_a: nn.Module, model_b: nn.Module, batch: Any, out: Path,
    options: Optional[CompareOptions] = None,
) -> Path:
    """
    Capture two models on one batch and write one bundle of their layers paired by name.

    Args:
        model_a: The model the graph follows.
        model_b: The model compared against it.
        batch: The input both models run on.
        out: Output folder.
        options: Labels, shared capture options, inline_mb, and web_dir. Defaults when None.

    Returns:
        Path of the written index.html.

    Notes:
        Both captures budget COMPARE_BYTES_PER_POSITION, whatever bytes_per_position the
        capture options hold, since every pane also stores joint RGB. Each pane budgets half
        of total_mb, so the whole bundle fits total_mb.
    """
    opts = options or CompareOptions()
    capture = opts.capture
    pane = replace(
        capture, bytes_per_position=COMPARE_BYTES_PER_POSITION, total_mb=capture.total_mb / 2
    )
    result_a = run_pipeline(model_a, batch, pane)
    result_b = run_pipeline(model_b, batch, pane)
    graph, payloads = compare_graph(result_a, result_b, opts.labels, capture.seed)
    graph["meta"]["sample_index"] = capture.sample_index
    # The bundle check measures both panes together against the full total_mb.
    bundle = BundleOptions(capture.layer_mb, capture.total_mb, opts.web_dir or WEB_DIR)
    return write_bundle(Path(out), graph, payloads, opts.inline_mb, bundle)
