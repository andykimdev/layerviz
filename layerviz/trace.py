"""
Trace one forward pass into execution order, provenance edges, a module tree, and outputs.

Purpose:

- Record every module call with forward hooks and every tensor op with a TorchFunctionMode.
- Build provenance edges from producer tags that ride on tensors as a Python attribute.

Special Notes:

- The tag helpers and the function mode live in layerviz.tags.
- Nodes store detached clones of outputs, so a later in place op cannot change them.
- The recorder remembers every tensor it tags and clears exactly those when the trace ends.
"""

import copy
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from layerviz.tags import (
    _TagPropagationMode,
    clear_tags,
    flatten_tensors,
    set_tags,
    tags_of,
    tensors_in_call,
)

INPUT = "input"

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
    shares: str | None = None


@dataclass
class Trace:
    """
    The nodes in execution order, the provenance edges, and the executed module tree.

    """

    nodes: list[TraceNode]
    edges: list[tuple[str, str]]
    children: dict[str, list[str]] = field(default_factory=dict)


def snapshot(obj: Any) -> Any:
    """
    Copy the tensors inside obj, following the same cases as flatten_tensors.

    Args:
        obj: Any value a module receives or returns.

    Returns:
        A copy whose tensors are detached clones. Tuples, lists, and dicts are rebuilt as
        their plain type. An object with a features tensor gets a copy with cloned
        features. Any other value is returned as is.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(snapshot(item) for item in obj)
    if isinstance(obj, list):
        return [snapshot(item) for item in obj]
    if isinstance(obj, dict):
        return {key: snapshot(item) for key, item in obj.items()}
    features = getattr(obj, "features", None)
    if not isinstance(features, torch.Tensor):
        return obj
    return _with_features(obj, features.detach().clone())


def _with_features(obj: Any, features: torch.Tensor) -> Any:
    """
    Return a shallow copy of obj that holds other features.

    Args:
        obj: An object with a features tensor, such as a sparse tensor.
        features: The tensor the copy holds instead.

    Returns:
        The result of obj.replace_feature when obj has it, since spconv refuses a direct
        assignment, else a shallow copy with features set.
    """
    replace = getattr(obj, "replace_feature", None)
    if callable(replace):
        return replace(features)
    copied = copy.copy(obj)
    copied.features = features
    return copied


def _shape_of(output: Any) -> list[int] | None:
    """
    Return the shape of the single tensor in output.

    Args:
        output: Any value a module returns.

    Returns:
        The shape as a list, or None when output holds zero or several tensors.
    """
    tensors = flatten_tensors(output)
    return list(tensors[0].shape) if len(tensors) == 1 else None


def _parent_name(name: str) -> str | None:
    """
    Return the dotted parent path of a module name.

    Args:
        name: A module name from named_modules, empty for the root.

    Returns:
        The parent path, empty for a top level child, or None for the root.
    """
    return None if name == "" else name.rpartition(".")[0]


class _Recorder:
    """
    Collect nodes and edges from module hooks during one forward pass.

    """

    def __init__(self) -> None:
        self.nodes: dict[str, TraceNode] = {}
        self.edges: dict[tuple[str, str], None] = {}
        self.last_child: dict[str, str] = {}
        self.live_outputs: dict[str, Any] = {}
        # Weak references, so remembering a tensor never keeps it alive past its last use.
        self.tagged: list[weakref.ref[torch.Tensor]] = []

    def tag(self, tensors: list[torch.Tensor], tags: frozenset[str]) -> None:
        """
        Set tags on the tensors and remember them, so the trace can clear them at the end.

        Args:
            tensors: Tensors to tag.
            tags: The producer names to store.
        """
        set_tags(tensors, tags)
        self.tagged.extend(weakref.ref(tensor) for tensor in tensors)

    def record_input(self, batch: Any) -> None:
        """
        Add the input node and tag the batch tensors as coming from it.

        Args:
            batch: The model input.
        """
        self.nodes[INPUT] = TraceNode(
            name=INPUT,
            type=type(batch).__name__,
            parent=None,
            index=0,
            output=snapshot(batch),
            output_shape=_shape_of(batch),
        )
        self.tag(flatten_tensors(batch), frozenset({INPUT}))

    def before(self, name: str, module: nn.Module, args: tuple, kwargs: dict) -> None:
        """
        Add the module node on its first call and an edge from each input producer.

        Args:
            name: The module name from named_modules.
            module: The module being called.
            args: Positional arguments of the call.
            kwargs: Keyword arguments of the call.
        """
        if name not in self.nodes:
            self.nodes[name] = TraceNode(
                name=name,
                type=type(module).__name__,
                parent=_parent_name(name),
                index=len(self.nodes),
            )
        for producer in tags_of(tensors_in_call(args, kwargs)):
            self.edges[(producer, name)] = None

    def after(self, name: str, output: Any) -> None:
        """
        Store a snapshot of the output, note a shared child output, and retag the output.

        Args:
            name: The module name from named_modules.
            output: The live value the module returned.
        """
        node = self.nodes[name]
        # The clone runs under the tag mode, so it is tagged during the trace and cleared
        # with the other tagged tensors.
        node.output = snapshot(output)
        node.output_shape = _shape_of(output)
        node.shares = self._shared_child(name, output)
        self.live_outputs[name] = output
        self.tag(flatten_tensors(output), frozenset({name}))
        if node.parent is not None:
            self.last_child[node.parent] = name

    def _shared_child(self, name: str, output: Any) -> str | None:
        """
        Return the last executed child when output is that child's live output object.

        Args:
            name: The module name whose output is checked.
            output: The live value that module returned.

        Returns:
            The child name, or None when no child ran or the output is a new object.
        """
        child = self.last_child.pop(name, None)
        if child is None or output is None or self.live_outputs[child] is not output:
            return None
        return child

    def to_trace(self) -> Trace:
        """
        Build the trace from the recorded nodes and edges.

        Returns:
            The nodes in execution order, the edges, and the executed module tree.
        """
        modules = [n for n in self.nodes.values() if n.name != INPUT]
        children: dict[str, list[str]] = {n.name: [] for n in modules}
        for node in modules:
            if node.parent is not None:
                children[node.parent].append(node.name)
        return Trace(list(self.nodes.values()), list(self.edges), children)

    def clear_all_tags(self) -> None:
        """Remove the tags from every tagged tensor still alive, so none outlives the trace."""
        alive = [tensor for tensor in (ref() for ref in self.tagged) if tensor is not None]
        clear_tags(alive)
        self.tagged.clear()
        self.live_outputs.clear()


def _register_hooks(model: nn.Module, recorder: _Recorder) -> list[RemovableHandle]:
    """
    Attach a forward pre hook and a forward hook to every module.

    Args:
        model: The module whose submodules get hooks.
        recorder: The recorder the hooks report to.

    Returns:
        The handles that remove the hooks.
    """
    handles = []
    for name, module in model.named_modules():
        pre, post = _pre_hook(recorder, name), _post_hook(recorder, name)
        handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(module.register_forward_hook(post, with_kwargs=True))
    return handles


def _pre_hook(recorder: _Recorder, name: str) -> Callable:
    """
    Return a forward pre hook that reports the call of module name to the recorder.

    Args:
        recorder: The recorder the hook reports to.
        name: The module name from named_modules.

    Returns:
        A hook for register_forward_pre_hook with with_kwargs set.
    """

    def hook(module: nn.Module, args: tuple, kwargs: dict) -> None:
        """
        Report one module call.

        Args:
            module: The module being called.
            args: Positional arguments of the call.
            kwargs: Keyword arguments of the call.
        """
        recorder.before(name, module, args, kwargs)

    return hook


def _post_hook(recorder: _Recorder, name: str) -> Callable:
    """
    Return a forward hook that reports the output of module name to the recorder.

    Args:
        recorder: The recorder the hook reports to.
        name: The module name from named_modules.

    Returns:
        A hook for register_forward_hook with with_kwargs set.
    """

    def hook(module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        """
        Report one module output.

        Args:
            module: The module that ran, unused here.
            args: Positional arguments of the call, unused here.
            kwargs: Keyword arguments of the call, unused here.
            output: The value the module returned.
        """
        recorder.after(name, output)

    return hook


def _call_model(model: nn.Module, batch: Any) -> Any:
    """
    Run the model on the batch as a single positional argument.

    Args:
        model: The module to run.
        batch: The input.

    Returns:
        The model output.
    """
    return model(batch)


def trace_forward(model: nn.Module, batch: Any, forward: Forward | None = None) -> Trace:
    """
    Run one forward pass and record execution order, provenance, the tree, and outputs.

    Args:
        model: The module to trace. Every submodule gets temporary hooks.
        batch: The input. Its tensors are tagged as the input node.
        forward: Callable taking (model, batch) that runs the pass. Defaults to model(batch).

    Returns:
        A Trace with the input node first and module nodes in call order. Each node output
        is a detached clone taken when the module returned.

    Notes:
        - Hooks, the function mode, and every tag the trace set are removed even when the
          forward raises.
        - A container retags its output, so downstream edges come from the container,
          not from its last child.
        - A module called more than once is one node that keeps its last output. Its edges
          can form a cycle, such as l2 to act and then act to l2.
    """
    forward = forward or _call_model
    recorder = _Recorder()
    recorder.record_input(batch)
    handles = _register_hooks(model, recorder)
    try:
        with _TagPropagationMode(recorder.tag):
            forward(model, batch)
    finally:
        for handle in handles:
            handle.remove()
        recorder.clear_all_tags()
    return recorder.to_trace()
