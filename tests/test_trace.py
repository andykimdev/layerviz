"""
Test that trace_forward records order, provenance edges, the module tree, and outputs.

Purpose:

- Check residual and cat edges, data dependent branches, container sharing, and cleanup.

Special Notes:

- Models live in this file because tests/models.py lands in the same wave.
"""

import io
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from layerviz.trace import trace_forward


class ResidualBlock(nn.Module):
    """
    Add the block input back onto two convolutions, then apply a final activation.

    """

    def __init__(self, channels: int, final_act: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act2 = nn.ReLU() if final_act else None

    def forward(self, x):
        """Return the residual sum, passed through act2 when it exists."""
        summed = self.conv2(self.act1(self.conv1(x))) + x
        return self.act2(summed) if self.act2 is not None else summed


class SkipNet(nn.Module):
    """
    Run a stem, a residual block, a downsample, an upsample, and a cat into a head.

    """

    def __init__(self, final_act: bool = True):
        super().__init__()
        self.stem = nn.Conv2d(1, 4, 3, padding=1)
        self.block = ResidualBlock(4, final_act)
        self.down = nn.Conv2d(4, 4, 3, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.head = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        """Return the head output over the cat of the block and upsampled features."""
        skip = self.block(self.stem(x))
        upsampled = self.up(self.down(skip))
        return self.head(torch.cat([skip, upsampled], dim=1))


class BranchNet(nn.Module):
    """
    Pick a layer with a Python if on a tensor value, which torch.fx cannot trace.

    """

    def __init__(self):
        super().__init__()
        self.pos = nn.Linear(2, 2)
        self.neg = nn.Linear(2, 2)

    def forward(self, x):
        """Return pos(x) when x sums above zero, else neg(x)."""
        return self.pos(x) if x.sum() > 0 else self.neg(x)


class Boom(nn.Module):
    """
    Raise inside forward to exercise cleanup on failure.

    """

    def forward(self, x):
        """Raise a RuntimeError."""
        raise RuntimeError("boom")


class SparseProducer(nn.Module):
    """
    Return an object that carries its tensor in a features attribute.

    """

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def forward(self, x):
        """Wrap lin(x) in a namespace with a features attribute."""
        return SimpleNamespace(features=self.lin(x))


class SparseConsumer(nn.Module):
    """
    Read the features tensor out of a sparse like object.

    """

    def forward(self, sparse):
        """Return the features tensor doubled."""
        return sparse.features * 2


class ReturnsParam(nn.Module):
    """
    Return a parameter object unchanged, so the tracer tags a long lived tensor.

    """

    def __init__(self, param: nn.Parameter):
        super().__init__()
        self.param = param

    def forward(self, x):
        """Return the stored parameter."""
        return self.param


class UsesParam(nn.Module):
    """
    Add a parameter to the input before a linear layer.

    """

    def __init__(self, param: nn.Parameter):
        super().__init__()
        self.param = param
        self.lin = nn.Linear(2, 2)

    def forward(self, x):
        """Return lin(x + param)."""
        return self.lin(x + self.param)


class TwiceNet(nn.Module):
    """
    Call l2 and act twice each, so one module runs more than once.

    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 2)
        self.act = nn.ReLU()

    def forward(self, x):
        """Return act(l2(act(l2(l1(x)))))."""
        x = self.l1(x)
        for _ in range(2):
            x = self.act(self.l2(x))
        return x


def make_skipnet(final_act: bool = True) -> SkipNet:
    """Build a SkipNet with fixed weights."""
    torch.manual_seed(0)
    return SkipNet(final_act).eval()


def image_batch() -> torch.Tensor:
    """Return a fixed one image batch of shape 1x1x8x8."""
    torch.manual_seed(1)
    return torch.randn(1, 1, 8, 8)


SKIPNET_EDGES = {
    ("input", ""),
    ("input", "stem"),
    ("stem", "block"),
    ("stem", "block.conv1"),
    ("block.conv1", "block.act1"),
    ("block.act1", "block.conv2"),
    ("block.conv2", "block.act2"),
    ("stem", "block.act2"),
    ("block", "down"),
    ("down", "up"),
    ("block", "head"),
    ("up", "head"),
}


@pytest.mark.parametrize("context", [torch.enable_grad, torch.no_grad, torch.inference_mode])
def test_residual_and_cat_model_yields_exactly_the_provenance_edges(context):
    model, batch = make_skipnet(), image_batch()

    with context():
        trace = trace_forward(model, batch)

    assert set(trace.edges) == SKIPNET_EDGES
    assert len(trace.edges) == len(SKIPNET_EDGES)


def test_nodes_record_name_type_parent_index_and_leaf_in_execution_order():
    trace = trace_forward(make_skipnet(), image_batch())

    rows = [(n.name, n.type, n.parent, n.index, n.leaf) for n in trace.nodes]

    assert rows == [
        ("input", "Tensor", None, 0, True),
        ("", "SkipNet", None, 1, False),
        ("stem", "Conv2d", "", 2, True),
        ("block", "ResidualBlock", "", 3, False),
        ("block.conv1", "Conv2d", "block", 4, True),
        ("block.act1", "ReLU", "block", 5, True),
        ("block.conv2", "Conv2d", "block", 6, True),
        ("block.act2", "ReLU", "block", 7, True),
        ("down", "Conv2d", "", 8, True),
        ("up", "Upsample", "", 9, True),
        ("head", "Conv2d", "", 10, True),
    ]


def test_children_map_each_executed_module_to_its_children_in_order():
    trace = trace_forward(make_skipnet(), image_batch())

    assert trace.children == {
        "": ["stem", "block", "down", "up", "head"],
        "stem": [],
        "block": ["block.conv1", "block.act1", "block.conv2", "block.act2"],
        "block.conv1": [],
        "block.act1": [],
        "block.conv2": [],
        "block.act2": [],
        "down": [],
        "up": [],
        "head": [],
    }


def test_data_dependent_branch_records_only_the_branch_that_ran():
    torch.manual_seed(0)
    model = BranchNet()

    trace = trace_forward(model, torch.ones(1, 2))

    assert [n.name for n in trace.nodes] == ["input", "", "pos"]
    assert set(trace.edges) == {("input", ""), ("input", "pos")}


def test_every_node_records_its_output_and_shape():
    model, batch = make_skipnet(), image_batch()

    trace = trace_forward(model, batch)
    by_name = {n.name: n for n in trace.nodes}

    assert by_name["input"].output is batch
    assert torch.equal(by_name["stem"].output, model.stem(batch))
    assert torch.equal(by_name[""].output, model(batch))
    assert {n.name: n.output_shape for n in trace.nodes} == {
        "input": [1, 1, 8, 8],
        "": [1, 2, 8, 8],
        "stem": [1, 4, 8, 8],
        "block": [1, 4, 8, 8],
        "block.conv1": [1, 4, 8, 8],
        "block.act1": [1, 4, 8, 8],
        "block.conv2": [1, 4, 8, 8],
        "block.act2": [1, 4, 8, 8],
        "down": [1, 4, 4, 4],
        "up": [1, 4, 8, 8],
        "head": [1, 2, 8, 8],
    }


def test_container_returning_its_last_child_output_shares_that_child():
    trace = trace_forward(make_skipnet(), image_batch())

    shares = {n.name: n.shares for n in trace.nodes if n.shares is not None}

    assert shares == {"block": "block.act2", "": "head"}


def test_container_returning_a_new_tensor_shares_nothing():
    trace = trace_forward(make_skipnet(final_act=False), image_batch())

    shares = {n.name: n.shares for n in trace.nodes if n.shares is not None}

    assert shares == {"": "head"}


def test_forward_that_raises_leaves_no_hooks_and_no_function_mode():
    model = nn.Sequential(nn.Linear(2, 2), Boom())

    with pytest.raises(RuntimeError, match="boom"):
        trace_forward(model, torch.ones(1, 2))

    leftover = [
        name for name, m in model.named_modules() if m._forward_hooks or m._forward_pre_hooks
    ]
    tagged = torch.ones(1)
    tagged._layerviz_tags = frozenset({"x"})
    assert leftover == []
    assert not hasattr(tagged + 1, "_layerviz_tags")


def test_custom_forward_and_dict_batch_tag_the_batch_as_input():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 2))
    batch = {"x": torch.ones(1, 2)}

    trace = trace_forward(model, batch, forward=lambda m, b: m(b["x"]))

    assert set(trace.edges) == {("input", ""), ("input", "0")}


def test_object_with_features_tensor_carries_provenance_between_modules():
    torch.manual_seed(0)
    model = nn.Sequential(SparseProducer(), SparseConsumer())

    trace = trace_forward(model, torch.ones(1, 2))

    assert set(trace.edges) == {("input", ""), ("input", "0"), ("input", "0.lin"), ("0", "1")}
    assert {n.name: n.output_shape for n in trace.nodes}["0"] == [1, 2]


def test_custom_plain_tensors_decides_which_tensors_carry_tags():
    torch.manual_seed(0)
    model = nn.Sequential(SparseProducer(), SparseConsumer())

    trace = trace_forward(model, torch.ones(1, 2), plain_tensors=_only_raw_tensors)

    assert set(trace.edges) == {("input", ""), ("input", "0"), ("input", "0.lin")}


def _only_raw_tensors(obj):
    """Return obj in a list when it is a tensor, else an empty list."""
    return [obj] if isinstance(obj, torch.Tensor) else []


def test_trace_leaves_no_tags_on_batch_or_node_outputs():
    batch = image_batch()

    trace = trace_forward(make_skipnet(), batch)

    tagged = [n.name for n in trace.nodes if hasattr(n.output, "_layerviz_tags")]
    assert tagged == []
    assert not hasattr(batch, "_layerviz_tags")


def test_traced_batch_saves_and_loads_with_weights_only():
    batch = image_batch()
    trace_forward(make_skipnet(), batch)
    buffer = io.BytesIO()

    torch.save(batch, buffer)
    buffer.seek(0)

    assert torch.equal(torch.load(buffer, weights_only=True), batch)


def test_second_trace_reusing_a_returned_parameter_has_no_stale_edge():
    torch.manual_seed(0)
    shared = nn.Parameter(torch.ones(1, 2))
    trace_forward(nn.Sequential(ReturnsParam(shared)), torch.ones(1, 2))

    trace = trace_forward(UsesParam(shared), torch.ones(1, 2))

    assert set(trace.edges) == {("input", ""), ("input", "lin")}


def test_module_called_twice_appears_once_with_last_output_and_cyclic_edges():
    torch.manual_seed(0)

    trace = trace_forward(TwiceNet(), torch.ones(1, 2))
    by_name = {n.name: n for n in trace.nodes}

    assert [n.name for n in trace.nodes] == ["input", "", "l1", "l2", "act"]
    assert set(trace.edges) == {
        ("input", ""),
        ("input", "l1"),
        ("l1", "l2"),
        ("l2", "act"),
        ("act", "l2"),
    }
    assert by_name["act"].output is by_name[""].output
