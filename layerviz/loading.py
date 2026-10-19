"""
Build models and samples from module:function specs, checkpoints, and sample files.

Purpose:

- Resolve a module:function spec to a callable, with the current directory importable.
- Load a checkpoint state dict into a freshly built model and name any key mismatch.
- Load a sample batch from a spec or a .pt, .npy, or .npz file.

Special Notes:

- Checkpoints and .pt samples load with weights_only=True first. A file that needs full
  unpickling loads with weights_only=False after a warning, so only load files you trust.
"""

import importlib
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
import torch
from torch import nn

STATE_KEYS = ("state_dict", "model")
WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def parse_spec(spec: str) -> Any:
    """
    Resolve a pkg.mod:attr spec to the object it names.

    Args:
        spec: A module path and an attribute path joined by a colon, such as pkg.mod:build.

    Returns:
        The named object, usually a factory function.

    Raises:
        ValueError: When the spec has no colon or the attribute does not exist.
    """
    module_name, sep, attr_path = spec.partition(":")
    if not sep or not module_name or not attr_path:
        raise ValueError(f"spec {spec!r} must look like pkg.module:function")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    target = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        if not hasattr(target, attr):
            raise ValueError(f"spec {spec!r}: {module_name} has no attribute {attr_path}")
        target = getattr(target, attr)
    return target


def _is_tensor_dict(value: Any) -> TypeGuard[dict]:
    """Return True when value is a non empty dict whose values are all tensors."""
    return isinstance(value, dict) and bool(value) and all(
        isinstance(item, torch.Tensor) for item in value.values()
    )


def _shared_prefix(state: dict) -> str | None:
    """Return the wrapper prefix that every key of state starts with, or None."""
    for prefix in WRAPPER_PREFIXES:
        if state and all(key.startswith(prefix) for key in state):
            return prefix
    return None


def _strip_prefix(state: dict) -> dict:
    """
    Drop the wrapper prefixes that DistributedDataParallel or torch.compile add to every key.

    Notes:
        A prefix is stripped only when every key starts with it, so a real submodule named
        module keeps its keys. Stripping repeats while some prefix is shared by every key.
    """
    prefix = _shared_prefix(state)
    while prefix is not None:
        state = {key[len(prefix):]: value for key, value in state.items()}
        prefix = _shared_prefix(state)
    return state


def find_state_dict(checkpoint: Any, state_key: str | None = None) -> dict | None:
    """
    Find the dict of tensors in a loaded checkpoint and strip its shared wrapper prefixes.

    Args:
        checkpoint: The object torch.load returned.
        state_key: Key searched first, before state_dict, model, and the top level.

    Returns:
        The state dict, or None when no dict of tensors is found.

    Notes:
        A module. or _orig_mod. prefix is stripped only when every key starts with it.
    """
    if not isinstance(checkpoint, dict):
        return None
    keys = [key for key in (state_key, *STATE_KEYS) if key is not None]
    for candidate in [checkpoint.get(key) for key in keys] + [checkpoint]:
        if _is_tensor_dict(candidate):
            return _strip_prefix(candidate)
    return None


def safe_torch_load(path: Path | str) -> Any:
    """
    Load a torch.save file with weights_only=True, falling back to full unpickling.

    Args:
        path: The file written by torch.save.

    Returns:
        The loaded object, on the CPU.

    Notes:
        The fallback emits one UserWarning that names the file, since full unpickling can run
        arbitrary code from it.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        warnings.warn(
            f"{path} holds objects beyond tensors, loading it with weights_only=False. "
            "Only load files you trust.", UserWarning, stacklevel=2,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def load_state(path: Path | str, state_key: str | None = None) -> dict | nn.Module:
    """
    Load a checkpoint and find its state dict.

    Args:
        path: The checkpoint file written by torch.save.
        state_key: Key searched first for the state dict, before state_dict and model.

    Returns:
        The state dict with its shared wrapper prefixes stripped, or the saved nn.Module as is.

    Raises:
        ValueError: When no searched key or the top level holds a dict of tensors.
    """
    checkpoint = safe_torch_load(path)
    if isinstance(checkpoint, nn.Module):
        return checkpoint
    state = find_state_dict(checkpoint, state_key)
    if state is None:
        keys = [key for key in (state_key, *STATE_KEYS) if key is not None]
        found = list(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise ValueError(f"checkpoint {path} has no dict of tensors under {keys}, found {found}")
    return state


def load_model(
    spec: str, checkpoint: Path | str | None = None, state_key: str | None = None,
    seed: int = 0,
) -> nn.Module:
    """
    Build a model from a factory spec and load a checkpoint into it.

    Args:
        spec: The module:function factory, called with no arguments.
        checkpoint: Checkpoint file to load, or None to keep the seeded initial weights.
        state_key: Checkpoint key searched first for the state dict.
        seed: Torch seed set before the factory runs.

    Returns:
        The model in eval mode.

    Raises:
        ValueError: When the checkpoint keys miss or exceed the model's keys.

    Notes:
        The seed applies inside a forked random state, so the caller's torch RNG is unchanged.
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = parse_spec(spec)()
    if checkpoint is not None:
        state = load_state(checkpoint, state_key)
        if isinstance(state, nn.Module):
            state = state.state_dict()
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                f"checkpoint {checkpoint} does not fit {spec}: missing keys "
                f"{result.missing_keys}, unexpected keys {result.unexpected_keys}"
            )
    return model.eval()


def _load_npz(path: Path) -> torch.Tensor | dict:
    """Load an .npz file as one tensor when it holds one array, else as a name to tensor dict."""
    with np.load(path) as arrays:
        tensors = {name: torch.from_numpy(arrays[name]) for name in arrays.files}
    return next(iter(tensors.values())) if len(tensors) == 1 else tensors


def load_sample(spec_or_path: str) -> Any:
    """
    Load a model input from a module:function spec or a sample file.

    Args:
        spec_or_path: A spec called with no arguments, or a .pt, .npy, or .npz file path.

    Returns:
        The batch: what the spec or .pt file holds, or tensors made from the NumPy arrays.

    Raises:
        ValueError: When a path has another suffix and is not a spec either.
    """
    path = Path(spec_or_path)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return safe_torch_load(path)
    if suffix == ".npy":
        return torch.from_numpy(np.load(path))
    if suffix == ".npz":
        return _load_npz(path)
    if ":" in spec_or_path and not path.exists():
        return parse_spec(spec_or_path)()
    raise ValueError(
        f"sample {spec_or_path} is not a module:function spec or a .pt, .npy, .npz file"
    )
