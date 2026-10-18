"""
Trace one forward pass into execution order, provenance edges, a module tree, and outputs.

Purpose:

- Record every module call with forward hooks and every tensor op with a TorchFunctionMode.
- Build provenance edges from producer tags that ride on tensors as a Python attribute.

Special Notes:

- Tags live in the _layerviz_tags attribute, which works under no_grad and inference_mode.
- Objects with a features tensor join provenance, so sparse outputs need no spconv import.
- The tracer keeps no tensor references beyond the outputs stored on its nodes.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

INPUT = "input"
TAG_ATTR = "_layerviz_tags"

PlainTensors = Callable[[Any], list]
Forward = Callable[[nn.Module, Any], Any]


@dataclass
class TraceNode:
    """
    One executed module, or the input batch, with its place in the tree and its output.

    """

    name: str
    type: str
    parent: str | None
    index: int
    output: Any = None
    output_shape: list[int] | None = None
    leaf: bool = True
    shares: str | None = None


@dataclass
class Trace:
    """
    The nodes in execution order, the provenance edges, and the executed module tree.

    """

    nodes: list[TraceNode]
    edges: list[tuple[str, str]]
    children: dict[str, list[str]] = field(default_factory=dict)


def flatten_tensors(obj: Any) -> list[torch.Tensor]:
    """
    Collect the tensors inside tensors, tuples, lists, dicts, and objects with features.

    Args:
        obj: Any value a module receives or returns.

    Returns:
        The tensors found, in traversal order. Other values contribute nothing.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        return [t for item in obj for t in flatten_tensors(item)]
    if isinstance(obj, dict):
        return [t for item in obj.values() for t in flatten_tensors(item)]
    features = getattr(obj, "features", None)
    return [features] if isinstance(features, torch.Tensor) else []


def tags_of(tensors: Iterable[torch.Tensor]) -> frozenset[str]:
    """
    Return the union of the producer tags carried by the given tensors.

    Args:
        tensors: Tensors that may carry a tag attribute.

    Returns:
        The union of their tags, empty when none is tagged.
    """
    found: frozenset[str] = frozenset()
    for tensor in tensors:
        found |= getattr(tensor, TAG_ATTR, frozenset())
    return found


def set_tags(tensors: Iterable[torch.Tensor], tags: frozenset[str]) -> None:
    """
    Replace the producer tags on each tensor.

    Args:
        tensors: Tensors to tag.
        tags: The producer names to store.
    """
    for tensor in tensors:
        setattr(tensor, TAG_ATTR, tags)


def clear_tags(tensors: Iterable[torch.Tensor]) -> None:
    """
    Remove the producer tags from each tensor that carries them.

    Args:
        tensors: Tensors that may carry a tag attribute.
    """
    for tensor in tensors:
        if hasattr(tensor, TAG_ATTR):
            delattr(tensor, TAG_ATTR)


def tensors_in_call(plain_tensors: PlainTensors, args: tuple, kwargs: dict) -> list:
    """
    Collect the tensors in each positional and keyword argument of one call.

    Args:
        plain_tensors: Callable returning the tensors inside one argument.
        args: Positional arguments of the call.
        kwargs: Keyword arguments of the call.

    Returns:
        The tensors found, positional arguments first.
    """
    return [t for arg in (*args, *kwargs.values()) for t in plain_tensors(arg)]


class _TagPropagationMode(TorchFunctionMode):
    """
    Give every tensor result of a torch function the union of its tensor arguments' tags.

    """

    def __init__(self, plain_tensors: PlainTensors):
        super().__init__()
        self.plain_tensors = plain_tensors

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Run func, then tag its tensor results with the tags of its tensor arguments."""
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        tags = tags_of(tensors_in_call(self.plain_tensors, args, kwargs))
        if tags:
            set_tags(self.plain_tensors(result), tags)
        return result


def _shape_of(output: Any, plain_tensors: PlainTensors) -> list[int] | None:
    """Return the shape of the single tensor in output, or None when there is not one."""
    tensors = plain_tensors(output)
    return list(tensors[0].shape) if len(tensors) == 1 else None


def _parent_name(name: str) -> str | None:
    """Return the dotted parent path of a module name, or None for the root."""
    return None if name == "" else name.rpartition(".")[0]


class _Recorder:
    """
    Collect nodes and edges from module hooks during one forward pass.

    """

    def __init__(self, plain_tensors: PlainTensors):
        self.plain_tensors = plain_tensors
        self.nodes: dict[str, TraceNode] = {}
        self.edges: dict[tuple[str, str], None] = {}
        self.last_child: dict[str, str] = {}

    def record_input(self, batch: Any) -> None:
        """Add the input node and tag the batch tensors as coming from it."""
        self.nodes[INPUT] = TraceNode(
            name=INPUT,
            type=type(batch).__name__,
            parent=None,
            index=0,
            output=batch,
            output_shape=_shape_of(batch, self.plain_tensors),
        )
        set_tags(self.plain_tensors(batch), frozenset({INPUT}))

    def before(self, name: str, module: nn.Module, args: tuple, kwargs: dict) -> None:
        """Add the module node on its first call and an edge from each input producer."""
        if name not in self.nodes:
            self.nodes[name] = TraceNode(
                name=name,
                type=type(module).__name__,
                parent=_parent_name(name),
                index=len(self.nodes),
                leaf=next(module.children(), None) is None,
            )
        for producer in tags_of(tensors_in_call(self.plain_tensors, args, kwargs)):
            self.edges[(producer, name)] = None

    def after(self, name: str, output: Any) -> None:
        """Store the output, note a shared child output, and retag the output tensors."""
        node = self.nodes[name]
        node.output = output
        node.output_shape = _shape_of(output, self.plain_tensors)
        node.shares = self._shared_child(name, output)
        set_tags(self.plain_tensors(output), frozenset({name}))
        if node.parent is not None:
            self.last_child[node.parent] = name

    def _shared_child(self, name: str, output: Any) -> str | None:
        """Return the last executed child when output is that child's output object."""
        child = self.last_child.pop(name, None)
        if child is None or output is None or self.nodes[child].output is not output:
            return None
        return child

    def to_trace(self) -> Trace:
        """Return the recorded nodes, edges, and executed module tree."""
        modules = [n for n in self.nodes.values() if n.name != INPUT]
        children: dict[str, list[str]] = {n.name: [] for n in modules}
        for node in modules:
            if node.parent is not None:
                children[node.parent].append(node.name)
        return Trace(list(self.nodes.values()), list(self.edges), children)

    def clear_all_tags(self) -> None:
        """Remove tags from the batch and every node output, so no tag outlives the trace."""
        for node in self.nodes.values():
            clear_tags(self.plain_tensors(node.output))


def _register_hooks(model: nn.Module, recorder: _Recorder) -> list:
    """Attach a forward pre hook and a forward hook to every module and return the handles."""
    handles = []
    for name, module in model.named_modules():
        pre, post = _pre_hook(recorder, name), _post_hook(recorder, name)
        handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(module.register_forward_hook(post, with_kwargs=True))
    return handles


def _pre_hook(recorder: _Recorder, name: str) -> Callable:
    """Return a forward pre hook that reports the call of module name to the recorder."""

    def hook(module, args, kwargs):
        recorder.before(name, module, args, kwargs)

    return hook


def _post_hook(recorder: _Recorder, name: str) -> Callable:
    """Return a forward hook that reports the output of module name to the recorder."""

    def hook(module, args, kwargs, output):
        recorder.after(name, output)

    return hook


def _call_model(model: nn.Module, batch: Any) -> Any:
    """Run the model on the batch as a single positional argument."""
    return model(batch)


def trace_forward(
    model: nn.Module,
    batch: Any,
    forward: Forward | None = None,
    plain_tensors: PlainTensors | None = None,
) -> Trace:
    """
    Run one forward pass and record execution order, provenance, the tree, and outputs.

    Args:
        model: The module to trace. Every submodule gets temporary hooks.
        batch: The input. Its tensors are tagged as the input node.
        forward: Callable taking (model, batch) that runs the pass. Defaults to model(batch).
        plain_tensors: Callable returning the tensors inside one argument or output.
            Defaults to flatten_tensors.

    Returns:
        A Trace with the input node first and module nodes in call order.

    Notes:
        - Hooks, the function mode, and all tags are removed even when the forward raises.
        - A container retags its output, so downstream edges come from the container,
          not from its last child.
        - A module called more than once is one node that keeps its last output. Its edges
          can form a cycle, such as l2 to act and then act to l2.
    """
    forward = forward or _call_model
    plain_tensors = plain_tensors or flatten_tensors
    recorder = _Recorder(plain_tensors)
    recorder.record_input(batch)
    handles = _register_hooks(model, recorder)
    try:
        with _TagPropagationMode(plain_tensors):
            forward(model, batch)
    finally:
        for handle in handles:
            handle.remove()
        recorder.clear_all_tags()
    return recorder.to_trace()
