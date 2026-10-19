"""
Build the graph.json nodes, boxes, edges, and meta from a trace and its captured layers.

Purpose:

- Give node ids, the layout leaves, and the leaf edges that single and compare bundles share.
- Turn each traced node and its captured layer into a graph.json entry with coordinates.

Special Notes:

- Edges from a container start at its last executed leaf, and edges into a container are
  dropped, since its first child already has the same inputs.
"""

from typing import Any

from layerviz import __version__
from layerviz.layout import Box, Placed, layout_graph
from layerviz.pipeline import CapturedLayer, PipelineResult
from layerviz.trace import INPUT, Trace, TraceNode

EXPLAINED_COUNT = 3


def node_id(index: int) -> str:
    """
    Return the graph id of the node at an execution index, such as L0007.

    Args:
        index: The execution index of the node.

    Returns:
        The id, L followed by the index padded to four digits.
    """
    return f"L{index:04d}"


def layout_leaves(trace: Trace) -> list[str]:
    """
    Return the nodes with no executed children in execution order, the input first.

    Args:
        trace: The traced forward pass.

    Returns:
        The leaf names.
    """
    return [node.name for node in trace.nodes if not trace.children.get(node.name)]


def exit_leaf(name: str, trace: Trace) -> str:
    """
    Follow last executed children down from name to the leaf that produced its output.

    Args:
        name: A node name, leaf or container.
        trace: The traced forward pass.

    Returns:
        The leaf name, which is name itself when it has no children.
    """
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
    """
    Return the first three explained variance ratios, padded with zeros.

    Args:
        layer: The captured layer.

    Returns:
        Exactly EXPLAINED_COUNT ratios, so a 2 channel layer ends in 0.0.
    """
    values = [float(v) for v in layer.reduced.explained[:EXPLAINED_COUNT]]
    return values + [0.0] * (EXPLAINED_COUNT - len(values))


def layer_fields(layer: CapturedLayer | None) -> dict[str, Any]:
    """
    Return the kind, counts, and reduction fields of a node, null when not captured.

    Args:
        layer: The captured layer of the node, or None.

    Returns:
        The kind, explained, retained, kept, and total fields.
    """
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
    node: TraceNode, ids: dict[str, str], is_leaf: bool, layer: CapturedLayer | None
) -> dict[str, Any]:
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
    """
    Map each leaf to its placement and each container to the center of its box.

    Args:
        placed: The placed leaves.
        boxes: The container boxes.

    Returns:
        Node name to (x, y).
    """
    spots = {box.name: ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2) for box in boxes}
    spots.update({leaf.name: (leaf.x, leaf.y) for leaf in placed})
    return spots


def graph_meta(result: PipelineResult, labels: list[str] | None, step: int | None,
               label: str | None, sample_index: int) -> dict[str, Any]:
    """
    Return the meta block of graph.json, in compare mode when two labels are given.

    Args:
        result: The pipeline result.
        labels: The two run labels of a comparison, or None for a single capture.
        step: The training step, or None.
        label: A label for this capture, or None.
        sample_index: The sample of the batch that was captured.

    Returns:
        The meta dict with model, step, mode, and version fields.
    """
    return {
        "model": result.model_name, "step": step, "sample_index": sample_index,
        "label": label, "labels": labels, "mode": "compare" if labels else "single",
        "version": __version__, "unpaired": None,
    }


def graph_body(trace: Trace, layers: dict[str, CapturedLayer]) -> dict[str, Any]:
    """
    Build the input id, placed nodes, boxes, and leaf edges of graph.json from a trace.

    Args:
        trace: The traced forward pass.
        layers: Node name to captured layer, for the nodes that were captured.

    Returns:
        The input, nodes, boxes, and edges entries. Nodes are in execution order with x and
        y set and skipped null.
    """
    ids = {node.name: node_id(node.index) for node in trace.nodes}
    leaves = layout_leaves(trace)
    edges = leaf_edges(trace, leaves)
    placed, boxes = layout_graph(leaves, edges, {node.name: node.parent for node in trace.nodes})
    spots = coordinates(placed, boxes)
    nodes = []
    for node in trace.nodes:
        entry = node_dict(node, ids, node.name in leaves, layers.get(node.name))
        entry["x"], entry["y"] = spots[node.name]
        nodes.append(entry)
    return {
        "input": ids[INPUT],
        "nodes": nodes,
        "boxes": [vars(box).copy() for box in boxes],
        "edges": [[ids[a], ids[b]] for a, b in edges],
    }


def annotate_nodes(nodes: list[dict[str, Any]], result: PipelineResult) -> None:
    """
    Fill each node's skip reason and the input node's kind from a pipeline result.

    Args:
        nodes: The graph.json nodes of the result's trace, changed in place.
        result: The pipeline result the nodes were built from.
    """
    for entry in nodes:
        entry["skipped"] = result.skipped.get(entry["name"])
        if entry["name"] == INPUT and result.input is not None:
            entry["kind"] = result.input.kind
