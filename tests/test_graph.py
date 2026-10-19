"""
Test the shared graph builder: node ids, leaf edges, boxes, and explained variance fields.

Purpose:

- Check node ids by execution index and the leaf edge rule on hand built traces.
- Check boxes and coordinates against written out values.
- Check that a 2 channel layer pads explained to three.
"""

import json

import pytest
import torch
from torch import nn

from layerviz.graph import graph_body, layout_leaves, leaf_edges
from layerviz.pipeline import CaptureOptions, run_pipeline
from layerviz.trace import INPUT, Trace, TraceNode
from tests.models import as_eval_pair

CHILDREN = {"": ["block", "head"], "block": ["block.a", "block.b"]}
PARENTS = {INPUT: None, "": None, "block": "", "block.a": "block", "block.b": "block", "head": ""}
INDICES = {INPUT: 0, "": 1, "block": 2, "block.a": 3, "block.b": 4, "head": 12}


def nested_trace(edges: list[tuple[str, str]]) -> Trace:
    """Build a trace of a root holding a two leaf block and a head, with the given edges."""
    nodes = [
        TraceNode(name, "Module", PARENTS[name], index, output_shape=[1, 4])
        for name, index in INDICES.items()
    ]
    return Trace(nodes, edges, dict(CHILDREN))


def chain_trace() -> Trace:
    """Build the nested trace with the input feeding block.a and block feeding the head."""
    return nested_trace([(INPUT, "block.a"), ("block.a", "block.b"), ("block", "head")])


def test_graph_body_ids_nodes_by_execution_index():
    body = graph_body(chain_trace(), {})

    ids = [(node["name"], node["id"]) for node in body["nodes"]]
    assert ids == [
        (INPUT, "L0000"), ("", "L0001"), ("block", "L0002"),
        ("block.a", "L0003"), ("block.b", "L0004"), ("head", "L0012"),
    ]
    assert body["input"] == "L0000"


def test_leaf_edges_drop_an_edge_into_a_container():
    trace = nested_trace([(INPUT, "block"), (INPUT, "block.a"), ("block.a", "block.b")])

    edges = leaf_edges(trace, layout_leaves(trace))

    assert edges == [(INPUT, "block.a"), ("block.a", "block.b")]


def test_leaf_edges_start_a_container_edge_at_its_last_leaf():
    trace = chain_trace()

    edges = leaf_edges(trace, layout_leaves(trace))

    assert edges == [(INPUT, "block.a"), ("block.a", "block.b"), ("block.b", "head")]


def test_graph_body_edges_use_node_ids():
    body = graph_body(chain_trace(), {})

    assert body["edges"] == [["L0000", "L0003"], ["L0003", "L0004"], ["L0004", "L0012"]]


def test_graph_body_boxes_wrap_each_container_with_its_margin():
    # Leaves stack one unit apart from y 0. Margins are 0.3 for block and 0.4 for the root above it.
    body = graph_body(chain_trace(), {})

    assert body["boxes"] == [
        {"name": "", "x0": -0.4, "y0": 0.6, "x1": 0.4, "y1": 3.4, "depth": 0},
        {"name": "block", "x0": -0.3, "y0": 0.7, "x1": 0.3, "y1": 2.3, "depth": 1},
    ]
    assert [box["name"] for box in body["boxes"]] == ["", "block"]


def test_graph_body_places_containers_at_box_centers_and_leaves_as_laid_out():
    body = graph_body(chain_trace(), {})

    assert {node["name"]: (node["x"], node["y"]) for node in body["nodes"]} == {
        INPUT: (0.0, 0.0), "block.a": (0.0, 1.0), "block.b": (0.0, 2.0), "head": (0.0, 3.0),
        "": (0.0, 2.0), "block": (0.0, 1.5),
    }


def test_two_channel_layer_has_three_explained_values_in_graph_json():
    torch.manual_seed(0)
    model, batch = as_eval_pair(nn.Sequential(nn.Conv2d(3, 2, 3, padding=1)),
                                torch.randn(1, 3, 8, 8))
    result = run_pipeline(model, batch, CaptureOptions())
    layers = {layer.node.name: layer for layer in result.layers}

    graph = json.loads(json.dumps(graph_body(result.trace, layers)))

    conv = next(node for node in graph["nodes"] if node["name"] == "0")
    assert len(conv["explained"]) == 3
    assert conv["explained"] == pytest.approx([0.57685, 0.42315, 0.0], abs=1e-4)
    assert conv["explained"][0] + conv["explained"][1] == pytest.approx(1.0, abs=1e-5)
    assert conv["explained"][2] == 0.0
