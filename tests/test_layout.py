"""
Test the graph layout that turns leaf names and edges into node coordinates and boxes.

Purpose:

- Check ranks, centered x values, unreachable nodes, and reused modules for placement.
- Check the root box and that nested containers give nested boxes with larger outer margins.

Special Notes:

- Hand made traces mirror trace_forward: an input node at the root, then a root container
  named with the empty string that holds the top level modules.
"""

import pytest

from layerviz.layout import Placed, layout_graph
from layerviz.trace import Trace, TraceNode

ROOT = ""


def make_trace(leaves, containers, edges):
    """
    Build a Trace by hand with an input node, a root container, and the given modules.

    Args:
        leaves: List of (name, parent) pairs in execution order, after the input.
        containers: List of (name, parent) pairs for modules with children, below the root.
        edges: List of (source, target) leaf name pairs.

    Returns:
        A Trace whose nodes list the input, the root, the containers, then the leaves, and
        whose children map the root and each container to its child names.
    """
    nodes = [
        TraceNode(name="input", type="Tensor", parent=None, index=0),
        TraceNode(name=ROOT, type="Net", parent=None, index=1),
    ]
    nodes += [
        TraceNode(name=name, type="Box", parent=parent, index=len(nodes))
        for name, parent in containers
    ]
    nodes += [
        TraceNode(name=name, type="Leaf", parent=parent, index=len(nodes) + i)
        for i, (name, parent) in enumerate(leaves)
    ]
    children = {ROOT: [], **{name: [] for name, _ in containers}}
    for name, parent in [*containers, *leaves]:
        children[parent].append(name)
    return Trace(nodes=nodes, edges=list(edges), children=children)


def layout_inputs(trace):
    """
    Turn a Trace into the leaf names, edges, and parents that layout_graph takes.

    Args:
        trace: A Trace built by make_trace.

    Returns:
        A tuple of leaf names in execution order, edges, and a name to parent dict.

    Notes:
        The leaves are the nodes that are not keys of trace.children, since make_trace keys
        only the root and the containers.
    """
    ordered = sorted(trace.nodes, key=lambda n: n.index)
    leaf_names = [node.name for node in ordered if node.name not in trace.children]
    parents = {node.name: node.parent for node in trace.nodes}
    return leaf_names, trace.edges, parents


def flat_trace(names, edges):
    """
    Build a Trace whose modules all sit directly under the root container.

    Args:
        names: Leaf names in execution order, after the input.
        edges: List of (source, target) leaf name pairs.

    Returns:
        A Trace with one top level leaf per name.
    """
    return make_trace([(name, ROOT) for name in names], [], edges)


def box_tuple(box):
    """
    Return a box as a plain tuple so its extents compare with pytest.approx.

    Args:
        box: A Box.

    Returns:
        A tuple of name, depth, and the four extents.
    """
    return box.name, box.depth, (box.x0, box.y0, box.x1, box.y1)


def boxes_overlap(first, second):
    """
    Tell whether two boxes share any area.

    Args:
        first: A Box.
        second: A Box.

    Returns:
        True when the two boxes overlap on both axes.
    """
    return (first.x0 < second.x1 and second.x0 < first.x1
            and first.y0 < second.y1 and second.y0 < first.y1)


def strictly_inside(inner, outer):
    """
    Tell whether one box lies inside another without touching its edges.

    Args:
        inner: The Box that should be enclosed.
        outer: The Box that should enclose it.

    Returns:
        True when every edge of inner is strictly within outer.
    """
    return (outer.x0 < inner.x0 and inner.x1 < outer.x1
            and outer.y0 < inner.y0 and inner.y1 < outer.y1)


def deep_tree_with_sibling():
    """
    Build a tree three containers deep with a sibling container at the next rank.

    Returns:
        A Trace where b1.c1.d1 holds leaf a, b1 holds leaf b, and sibling b2 holds leaf c.
    """
    leaves = [("a", "b1.c1.d1"), ("b", "b1"), ("c", "b2")]
    containers = [("b1", ROOT), ("b1.c1", "b1"), ("b1.c1.d1", "b1.c1"), ("b2", ROOT)]
    return make_trace(leaves, containers, [("input", "a"), ("a", "b"), ("b", "c")])


def test_linear_chain_gives_consecutive_ranks_at_x_zero():
    trace = flat_trace(["a", "b", "c"], [("input", "a"), ("a", "b"), ("b", "c")])

    placed, _ = layout_graph(*layout_inputs(trace))

    assert placed == [
        Placed("input", 0.0, 0.0, 0),
        Placed("a", 0.0, 1.0, 1),
        Placed("b", 0.0, 2.0, 2),
        Placed("c", 0.0, 3.0, 3),
    ]


def test_skip_edge_keeps_the_longer_path_rank():
    trace = flat_trace(["a", "b"], [("input", "a"), ("input", "b"), ("a", "b")])

    placed, _ = layout_graph(*layout_inputs(trace))

    assert placed == [
        Placed("input", 0.0, 0.0, 0),
        Placed("a", 0.0, 1.0, 1),
        Placed("b", 0.0, 2.0, 2),
    ]


def test_parallel_branches_share_a_rank_centered_in_execution_order():
    edges = [("input", "a"), ("input", "b"), ("a", "c"), ("b", "c")]
    trace = flat_trace(["a", "b", "c"], edges)

    placed, _ = layout_graph(*layout_inputs(trace))

    assert placed == [
        Placed("input", 0.0, 0.0, 0),
        Placed("a", -0.5, 1.0, 1),
        Placed("b", 0.5, 1.0, 1),
        Placed("c", 0.0, 2.0, 2),
    ]


def test_unreachable_node_takes_rank_zero_after_the_reachable_ones():
    trace = flat_trace(["stray", "a"], [("input", "a")])

    placed, _ = layout_graph(*layout_inputs(trace))

    assert placed == [
        Placed("input", -0.5, 0.0, 0),
        Placed("stray", 0.5, 0.0, 0),
        Placed("a", 0.0, 1.0, 1),
    ]


def test_reused_module_back_edge_is_ignored_for_ranking():
    # Mirrors TwiceNet, where act feeds l2 a second time after l2 first ran.
    edges = [("input", "l1"), ("l1", "l2"), ("l2", "act"), ("act", "l2")]
    trace = flat_trace(["l1", "l2", "act"], edges)

    placed, _ = layout_graph(*layout_inputs(trace))

    assert placed == [
        Placed("input", 0.0, 0.0, 0),
        Placed("l1", 0.0, 1.0, 1),
        Placed("l2", 0.0, 2.0, 2),
        Placed("act", 0.0, 3.0, 3),
    ]


def test_root_box_spans_top_level_leaves_at_depth_zero_without_the_input():
    trace = flat_trace(["a", "b"], [("input", "a"), ("a", "b")])

    _, boxes = layout_graph(*layout_inputs(trace))

    assert [box_tuple(box) for box in boxes] == [
        (ROOT, 0, pytest.approx((-0.3, 0.7, 0.3, 2.3))),
    ]


def test_nested_containers_give_nested_boxes_with_larger_outer_margin():
    leaves = [("a", "outer.inner"), ("b", "outer.inner"), ("c", "outer")]
    containers = [("outer", ROOT), ("outer.inner", "outer")]
    trace = make_trace(leaves, containers, [("input", "a"), ("a", "b"), ("b", "c")])

    _, boxes = layout_graph(*layout_inputs(trace))

    assert [box_tuple(box) for box in boxes] == [
        (ROOT, 0, pytest.approx((-0.45, 0.55, 0.45, 3.45))),
        ("outer", 1, pytest.approx((-0.4, 0.6, 0.4, 3.4))),
        ("outer.inner", 2, pytest.approx((-0.3, 0.7, 0.3, 2.3))),
    ]


def deep_tree_boxes():
    """Return the boxes of the deep tree with a sibling, keyed by container name."""
    _, boxes = layout_graph(*layout_inputs(deep_tree_with_sibling()))
    return {box.name: box for box in boxes}


@pytest.mark.parametrize("name, parent", [
    ("b1", ROOT), ("b1.c1", "b1"), ("b1.c1.d1", "b1.c1"), ("b2", ROOT),
])
def test_nested_box_lies_strictly_inside_its_parent_box(name, parent):
    boxes = deep_tree_boxes()

    assert strictly_inside(boxes[name], boxes[parent])


@pytest.mark.parametrize("other", ["b1", "b1.c1", "b1.c1.d1"])
def test_sibling_container_box_does_not_overlap_the_other_branch(other):
    boxes = deep_tree_boxes()

    assert not boxes_overlap(boxes["b2"], boxes[other])


def test_container_margin_counts_only_the_container_levels_inside_it():
    _, boxes = layout_graph(*layout_inputs(deep_tree_with_sibling()))

    assert [box_tuple(box) for box in boxes] == [
        (ROOT, 0, pytest.approx((-0.475, 0.525, 0.475, 3.475))),
        ("b1", 1, pytest.approx((-0.45, 0.55, 0.45, 2.45))),
        ("b2", 1, pytest.approx((-0.3, 2.7, 0.3, 3.3))),
        ("b1.c1", 2, pytest.approx((-0.4, 0.6, 0.4, 1.4))),
        ("b1.c1.d1", 3, pytest.approx((-0.3, 0.7, 0.3, 1.3))),
    ]


def test_container_without_placed_descendants_gets_no_box():
    trace = make_trace([("a", ROOT)], [("empty", ROOT)], [("input", "a")])

    _, boxes = layout_graph(*layout_inputs(trace))

    assert [box.name for box in boxes] == [ROOT]


def test_empty_graph_gives_no_nodes_and_no_boxes():
    assert layout_graph([], [], {}) == ([], [])


def test_edge_order_does_not_change_the_layout():
    edges = [("input", "a"), ("input", "b"), ("a", "c"), ("b", "c")]
    names, _, parents = layout_inputs(flat_trace(["a", "b", "c"], edges))

    assert layout_graph(names, edges, parents) == layout_graph(names, edges[::-1], parents)


def test_edge_to_an_unknown_node_raises_with_its_name():
    with pytest.raises(ValueError, match="ghost"):
        layout_graph(["input", "a"], [("input", "ghost")], {})
