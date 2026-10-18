"""
Capture one model and batch as a viewable bundle and build its graph dict.

Purpose:

- Turn a pipeline result and its layout into the graph.json dict that app.js reads.
- Run the pipeline and write the bundle, into a step folder when capturing during training.

Special Notes:

- A step that is not due returns before the model is touched, and only rank 0 captures.
- Edges from a container start at its last executed leaf, and edges into a container are
  dropped, since its first child already has the same inputs.
"""

from pathlib import Path
from typing import Any, Optional

from torch import nn

from layerviz import __version__
from layerviz.bundle import WEB_DIR, BundleOptions, LayerPayload, write_bundle
from layerviz.layout import Box, Placed, layout_graph
from layerviz.pipeline import CapturedLayer, CaptureOptions, PipelineResult, run_pipeline
from layerviz.snapshots import due, is_main_process, step_dir, write_index
from layerviz.trace import INPUT, Trace, TraceNode

EXPLAINED_COUNT = 3


def node_id(index: int) -> str:
    """Return the graph id of the node at an execution index, such as L0007."""
    return f"L{index:04d}"


def layout_leaves(trace: Trace) -> list[str]:
    """Return the nodes with no executed children in execution order, the input first."""
    return [node.name for node in trace.nodes if not trace.children.get(node.name)]


def exit_leaf(name: str, trace: Trace) -> str:
    """Follow last executed children down from name to the leaf that produced its output."""
    while trace.children.get(name):
        name = trace.children[name][-1]
    return name


def leaf_edges(trace: Trace, leaves: list[str]) -> list[tuple[str, str]]:
    """
    Project the trace edges onto leaves, without duplicates and in trace order.

    Args:
        trace: The traced forward pass.
        leaves: The layout leaves of the trace.

    Returns:
        (source, target) leaf pairs. A container source becomes its exit leaf and an edge
        into a container is dropped.
    """
    targets = set(leaves)
    edges = (
        (exit_leaf(source, trace), target)
        for source, target in trace.edges
        if target in targets
    )
    return list(dict.fromkeys(edges))


def explained_triple(layer: CapturedLayer) -> list[float]:
    """Return the first three explained variance ratios, padded with zeros."""
    values = [float(v) for v in layer.reduced.explained[:EXPLAINED_COUNT]]
    return values + [0.0] * (EXPLAINED_COUNT - len(values))


def layer_fields(layer: Optional[CapturedLayer]) -> dict:
    """Return the kind, counts, and reduction fields of a node, null when not captured."""
    if layer is None:
        return {"kind": None, "explained": None, "retained": None, "kept": None, "total": None}
    return {
        "kind": layer.data.kind,
        "explained": explained_triple(layer),
        "retained": float(layer.reduced.retained),
        "kept": layer.kept,
        "total": layer.total,
    }


def node_dict(
    node: TraceNode, ids: dict[str, str], is_leaf: bool, layer: Optional[CapturedLayer]
) -> dict:
    """
    Build the graph.json entry of one traced node, without its coordinates and files.

    Args:
        node: The traced node.
        ids: Node name to graph id.
        is_leaf: Whether the node is placed as a leaf rather than drawn as a box.
        layer: The captured layer of the node, or None.

    Returns:
        The node dict with ids, tree fields, shape, and layer fields.
    """
    return {
        "id": ids[node.name], "name": node.name, "type": node.type,
        "parent": node.parent, "index": node.index, "leaf": is_leaf,
        "shape": node.output_shape, **layer_fields(layer),
        "file": None, "channels": None, "channel_list": None,
        "shares": ids.get(node.shares) if node.shares else None,
        "skipped": None, "cka": None, "files": None,
    }


def coordinates(placed: list[Placed], boxes: list[Box]) -> dict[str, tuple[float, float]]:
    """Map each leaf to its placement and each container to the center of its box."""
    spots = {box.name: ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2) for box in boxes}
    spots.update({leaf.name: (leaf.x, leaf.y) for leaf in placed})
    return spots


def graph_meta(result: PipelineResult, labels: Optional[list[str]], step: Optional[int],
               label: Optional[str], sample_index: int) -> dict:
    """Return the meta block of graph.json, in compare mode when two labels are given."""
    return {
        "model": result.model_name, "step": step, "sample_index": sample_index,
        "label": label, "labels": labels, "mode": "compare" if labels else "single",
        "version": __version__, "unpaired": None,
    }


def build_graph(
    result: PipelineResult, labels: Optional[list[str]] = None, *, step: Optional[int] = None,
    label: Optional[str] = None, sample_index: int = 0,
) -> dict:
    """
    Build the graph.json dict of nodes, boxes, edges, and meta from a pipeline result.

    Args:
        result: The pipeline result.
        labels: The two run labels of a comparison, or None for a single capture.
        step: The training step, or None.
        label: A label for this capture, or None.
        sample_index: The sample of the batch that was captured.

    Returns:
        The graph dict with file fields null, which write_bundle fills.
    """
    trace = result.trace
    ids = {node.name: node_id(node.index) for node in trace.nodes}
    leaves = layout_leaves(trace)
    edges = leaf_edges(trace, leaves)
    parents = {node.name: node.parent for node in trace.nodes}
    placed, boxes = layout_graph(leaves, edges, parents)
    spots = coordinates(placed, boxes)
    layers = {layer.node.name: layer for layer in result.layers}
    nodes = []
    for node in trace.nodes:
        entry = node_dict(node, ids, node.name in leaves, layers.get(node.name))
        entry["skipped"] = result.skipped.get(node.name)
        entry["x"], entry["y"] = spots[node.name]
        if node.name == INPUT and result.input is not None:
            entry["kind"] = result.input.kind
        nodes.append(entry)
    return {
        "meta": graph_meta(result, labels, step, label, sample_index),
        "input": ids[INPUT],
        "nodes": nodes,
        "boxes": [vars(box).copy() for box in boxes],
        "edges": [[ids[a], ids[b]] for a, b in edges],
    }


def layer_payloads(result: PipelineResult) -> dict[str, list[LayerPayload]]:
    """Return node id to the single pane payload of every captured layer."""
    return {
        node_id(layer.node.index): [
            LayerPayload(layer.data, layer.reduced, channels=layer.channels)
        ]
        for layer in result.layers
    }


def capture(
    model: nn.Module, batch: Any, out: Path | str, *, step: Optional[int] = None,
    every: Optional[int] = None, label: Optional[str] = None, inline_mb: float = 16,
    web_dir: Path = WEB_DIR, **options: Any,
) -> Optional[Path]:
    """
    Capture one forward pass of model on batch as a bundle under out.

    Args:
        model: The module to capture, possibly wrapped by DDP or torch.compile.
        batch: The model input.
        out: The bundle folder, or the snapshot root when step is given.
        step: The training step, which writes into out/step_<step> and updates out/index.html.
        every: The capture interval. With step, a step that is not a multiple is skipped.
        label: A label shown with the capture.
        inline_mb: Largest index.html written as one self contained page.
        web_dir: Folder of the page files copied into the bundle.
        **options: CaptureOptions fields, such as sample_index, include, layer_mb, total_mb.

    Returns:
        Path of the written index.html, or None when the step is not due or the rank is not 0.

    Notes:
        The model keeps its training and grad modes, since run_pipeline restores them.
    """
    if step is not None and every is not None and not due(step, every):
        return None
    if not is_main_process():
        return None
    opts = CaptureOptions(**options)
    result = run_pipeline(model, batch, opts)
    graph = build_graph(result, step=step, label=label, sample_index=opts.sample_index)
    target = step_dir(Path(out), step) if step is not None else Path(out)
    limits = BundleOptions(opts.layer_mb, opts.total_mb, Path(web_dir))
    index = write_bundle(target, graph, layer_payloads(result), inline_mb, limits)
    if step is not None:
        write_index(Path(out))
    return index
