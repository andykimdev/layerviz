"""
Read, set, clear, and propagate the producer tags that ride on tensors during a trace.

Purpose:

- Decide which tensors a value holds, so tags land on the same tensors a trace snapshots.
- Give each torch function result the union of its tensor arguments' tags.

Special Notes:

- Tags live in the _layerviz_tags attribute, which works under no_grad and inference_mode.
- Objects with a features tensor join provenance, so sparse outputs need no spconv import.
- flatten_tensors lives here, not in trace.py, because the function mode needs it and
  trace.py imports this module.
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.overrides import TorchFunctionMode

TAG_ATTR = "_layerviz_tags"

Tagger = Callable[[list[torch.Tensor], frozenset[str]], None]


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


def tensors_in_call(args: tuple, kwargs: dict) -> list[torch.Tensor]:
    """
    Collect the tensors in each positional and keyword argument of one call.

    Args:
        args: Positional arguments of the call.
        kwargs: Keyword arguments of the call.

    Returns:
        The tensors found, positional arguments first.
    """
    return [t for arg in (*args, *kwargs.values()) for t in flatten_tensors(arg)]


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


class _TagPropagationMode(TorchFunctionMode):
    """
    Give every tensor result of a torch function the union of its tensor arguments' tags.

    Args:
        tag: Callable taking the result tensors and the tags to set on them.
    """

    def __init__(self, tag: Tagger):
        super().__init__()
        self.tag = tag

    def __torch_function__(
        self,
        func: Callable,
        types: Iterable[type],
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        """
        Run func, then tag its tensor results with the tags of its tensor arguments.

        Args:
            func: The torch function being called.
            types: The tensor types that take part in the call, unused here.
            args: Positional arguments of the call.
            kwargs: Keyword arguments of the call.

        Returns:
            The result of func, unchanged apart from its tags.
        """
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        tags = tags_of(tensors_in_call(args, kwargs))
        if tags:
            self.tag(flatten_tensors(result), tags)
        return result
