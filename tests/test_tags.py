"""
Test the tag helpers and that trace_forward leaves no tag or hook behind.

Purpose:

- Check that tags_of, set_tags, and clear_tags read, replace, and remove producer tags.
- Check that a trace after a raising forward, a buffer write, or a shared parameter
  records only what ran in that trace.

Special Notes:

- Leftovers are probed through public behavior only, never through private attributes.
"""

import pytest
import torch
from torch import nn

from layerviz.tags import clear_tags, set_tags, tags_of
from layerviz.trace import trace_forward


class FailsWhenArmed(nn.Module):
    """
    Raise inside forward while armed, else double the input.

    """

    def __init__(self, armed: bool):
        super().__init__()
        self.armed = armed

    def forward(self, x):
        """Raise a RuntimeError when armed, else return x doubled."""
        if self.armed:
            raise RuntimeError("boom")
        return x * 2


class Memory(nn.Module):
    """
    Read a buffer written by the previous call, then overwrite it with this call's lin output.

    """

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 2)
        self.lin = nn.Linear(2, 2)
        self.register_buffer("memory", torch.zeros(1, 2))

    def forward(self, x):
        """Return head(x + memory) and store lin of that result in memory."""
        out = self.head(x + self.memory)
        self.memory.copy_(self.lin(out))
        return out


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


def make_memory_net(armed: bool) -> nn.Sequential:
    """Build a Memory followed by a FailsWhenArmed with fixed weights."""
    torch.manual_seed(0)
    return nn.Sequential(Memory(), FailsWhenArmed(armed))


def test_set_tags_replaces_tags_of_unions_and_clear_tags_removes():
    a, b, untagged = torch.ones(1), torch.ones(1), torch.ones(1)
    set_tags([a], frozenset({"old"}))

    set_tags([a], frozenset({"x"}))
    set_tags([b], frozenset({"y", "z"}))
    union = tags_of([a, b, untagged])
    clear_tags([a, b, untagged])

    assert union == frozenset({"x", "y", "z"})
    assert tags_of([a, b, untagged]) == frozenset()


def test_raising_forward_leaves_no_hooks_and_retraces_like_a_fresh_model():
    model, batch = make_memory_net(armed=True), torch.ones(1, 2)
    with torch.no_grad(), pytest.raises(RuntimeError, match="boom"):
        trace_forward(model, batch)
    model[1].armed = False

    with torch.no_grad():
        # A leftover forward hook would tag the output of this untraced call.
        plain = model(batch)
        trace = trace_forward(model, batch)
        fresh = trace_forward(make_memory_net(armed=False), batch)

    assert tags_of([plain]) == frozenset()
    assert tags_of([batch]) == frozenset()
    assert [n.name for n in trace.nodes] == ["input", "", "0", "0.head", "0.lin", "1"]
    assert trace.edges == fresh.edges


def test_module_writing_a_buffer_gives_the_same_edges_when_traced_twice():
    torch.manual_seed(0)
    model = Memory()

    with torch.no_grad():
        first = trace_forward(model, torch.ones(1, 2))
        second = trace_forward(model, torch.ones(1, 2))

    expected = [("input", ""), ("input", "head"), ("head", "lin")]
    assert first.edges == expected
    assert second.edges == expected


def test_second_trace_reusing_a_returned_parameter_has_no_stale_edge():
    torch.manual_seed(0)
    shared = nn.Parameter(torch.ones(1, 2))
    trace_forward(nn.Sequential(ReturnsParam(shared)), torch.ones(1, 2))

    trace = trace_forward(UsesParam(shared), torch.ones(1, 2))

    assert set(trace.edges) == {("input", ""), ("input", "lin")}
