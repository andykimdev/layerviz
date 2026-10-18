"""
Run trace, shapes, budget, and reduction over one model and batch.

Purpose:

- Trace one forward pass with the model in eval and no_grad, then restore its modes.
- Turn each executed module output into a subsampled, reduced layer ready for the bundle.

Special Notes:

- Writes no files. extract.py turns outputs into layers and names the skipped ones.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from layerviz.budget import (
    BYTES_PER_POSITION, MB, plan_channel_caps, plan_keep, positions_in, subsample,
)
from layerviz.bundle import channel_overhead, layer_overhead
from layerviz.extract import extract_layers, input_context, input_layer
from layerviz.reduce import Reduced, channel_bytes, reduce_layer
from layerviz.shapes import dense, sparse  # noqa: F401  Importing registers the defaults.
from layerviz.shapes.base import LayerData
from layerviz.trace import Trace, TraceNode, trace_forward


@dataclass
class CaptureOptions:
    """
    Choose the sample, the layers, the plugin overrides, and the size caps of one capture.

    """

    sample_index: int = 0
    include: Optional[list[str]] = None
    exclude: Optional[list[str]] = None
    shapes: Optional[dict[str, str]] = None
    forward: Optional[Callable[[nn.Module, Any], Any]] = None
    layer_mb: float = 16
    total_mb: float = 1024
    seed: int = 0
    # Compare mode passes COMPARE_BYTES_PER_POSITION to budget the joint rgb as well.
    bytes_per_position: int = BYTES_PER_POSITION


@dataclass
class CapturedLayer:
    """
    Hold one subsampled layer with its reduction and its quantized channels.

    """

    node: TraceNode
    data: LayerData
    reduced: Reduced
    total: int
    kept: int
    channels: Optional[tuple[np.ndarray, np.ndarray]]


@dataclass
class PipelineResult:
    """
    Hold the trace, the captured layers in execution order, and the skipped layers.

    """

    trace: Trace
    layers: list[CapturedLayer]
    skipped: dict[str, str]
    model_name: str
    input: Optional[LayerData]


def unwrap(model: nn.Module) -> nn.Module:
    """
    Return the module inside DistributedDataParallel and torch.compile wrappers.

    Args:
        model: A module, possibly wrapped one or more times.

    Returns:
        The innermost module.
    """
    while True:
        if isinstance(model, DistributedDataParallel):
            model = model.module
        elif isinstance(getattr(model, "_orig_mod", None), nn.Module):
            model = model._orig_mod
        else:
            return model


@contextmanager
def prepared(model: nn.Module) -> Iterator[nn.Module]:
    """
    Yield the unwrapped model in eval mode under no_grad, restoring both afterwards.

    Args:
        model: The module to capture, possibly wrapped.

    Returns:
        A context manager that yields the unwrapped module.

    Notes:
        The training flag of every submodule is restored one by one, so mixed modes
        survive, and restoring happens even when the body raises.
    """
    modes = [(module, module.training) for module in model.modules()]
    inner = unwrap(model)
    inner.eval()
    try:
        with torch.no_grad():
            yield inner
    finally:
        for module, mode in modes:
            module.training = mode


def captured_layer(node: TraceNode, data: LayerData, keep: int, seed: int) -> CapturedLayer:
    """
    Subsample and reduce one layer, leaving its channels for with_channels.

    Args:
        node: The traced module.
        data: Its full layer.
        keep: Number of positions to keep.
        seed: Seed for subsampling and the PCA fit.

    Returns:
        The CapturedLayer with channels None.
    """
    kept, _ = subsample(data, keep, seed)
    return CapturedLayer(
        node=node,
        data=kept,
        reduced=reduce_layer(kept.features, seed=seed),
        total=len(data.positions),
        kept=len(kept.positions),
        channels=None,
    )


def channel_count(data: LayerData) -> int:
    """
    Return the number of channels of one layer, the product of its feature dimensions.

    Args:
        data: The layer.

    Returns:
        The channel count.
    """
    return int(np.prod(data.features.shape[1:], dtype=np.int64))


def channel_need(layer: CapturedLayer) -> int:
    """
    Return the bytes a channel file holding every channel of a captured layer takes.

    Args:
        layer: The subsampled layer.

    Returns:
        The channel file overhead plus one byte per kept position and channel.
    """
    n_channels = channel_count(layer.data)
    return channel_overhead(layer.kept, n_channels) + layer.kept * n_channels


def layer_file_bytes(layer: CapturedLayer, overhead: int, joint_bytes: int) -> int:
    """
    Return the most bytes the layer file of a captured layer takes.

    Args:
        layer: The subsampled, reduced layer.
        overhead: Bytes its file adds beyond its positions, from layer_overhead.
        joint_bytes: Extra bytes per position, 3 for the joint rgb in compare mode, else 0.

    Returns:
        The overhead plus the float32 coordinates, rgb, float16 scores, and joint rgb bytes.
    """
    dim = layer.data.positions.shape[1]
    k = layer.reduced.scores.shape[1]
    return overhead + layer.kept * (dim * 4 + 3 + k * 2 + joint_bytes)


def with_channels(layer: CapturedLayer, cap_bytes: int) -> CapturedLayer:
    """
    Quantize the channels of a captured layer, keeping the highest variance ones that fit.

    Args:
        layer: The subsampled layer.
        cap_bytes: Largest size of the channel file, header included, in bytes.

    Returns:
        A copy with channels set, or None when not even one channel fits under cap_bytes.
    """
    values_cap = cap_bytes - channel_overhead(layer.kept, channel_count(layer.data))
    if values_cap < max(layer.kept, 1):
        return replace(layer, channels=None)
    return replace(layer, channels=channel_bytes(layer.data.features, values_cap))


def budget_layers(
    accepted: list[tuple[TraceNode, LayerData]], opts: CaptureOptions
) -> list[CapturedLayer]:
    """
    Plan the kept positions of all accepted layers and capture each one.

    Args:
        accepted: The (node, layer) pairs in execution order.
        opts: The capture options with the size caps and seed.

    Returns:
        One CapturedLayer per accepted layer, in the same order.

    Notes:
        Positions keep priority. The channel files share only the bytes the layer files
        leave under total_mb, so layer and channel files together stay within it.
    """
    counts = {node.name: len(data.positions) for node, data in accepted}
    overhead = {node.name: layer_overhead(data) for node, data in accepted}
    keep = plan_keep(counts, opts.layer_mb, opts.total_mb, opts.bytes_per_position, overhead)
    layers = [captured_layer(node, data, keep[node.name], opts.seed) for node, data in accepted]
    joint_bytes = max(0, opts.bytes_per_position - BYTES_PER_POSITION)
    used = sum(layer_file_bytes(x, overhead[x.node.name], joint_bytes) for x in layers)
    need = {layer.node.name: channel_need(layer) for layer in layers}
    caps = plan_channel_caps(need, opts.layer_mb, opts.total_mb, used)
    return [with_channels(layer, caps[layer.node.name]) for layer in layers]


def run_pipeline(model: nn.Module, batch: Any, opts: CaptureOptions) -> PipelineResult:
    """
    Capture every selected layer of one forward pass as a reduced, subsampled layer.

    Args:
        model: The module to capture, possibly wrapped by DDP or torch.compile.
        batch: The model input.
        opts: The capture options.

    Returns:
        PipelineResult with the trace, the layers, the skip reasons, and the input layer.

    Raises:
        ValueError: When layer_mb is too small to hold one position.

    Notes:
        The model keeps its training and grad modes even when the forward raises.
    """
    if positions_in(opts.layer_mb, opts.bytes_per_position) < 1:
        raise ValueError(
            f"layer_mb={opts.layer_mb} holds no position, use at least "
            f"{opts.bytes_per_position / MB:.2e} MB"
        )
    with prepared(model) as inner:
        trace = trace_forward(inner, batch, opts.forward)
        ctx = input_context(batch, opts.sample_index)
        input_data = input_layer(batch, ctx)
        accepted, skipped = extract_layers(trace, ctx, opts, input_data)
    layers = budget_layers(accepted, opts)
    return PipelineResult(trace, layers, skipped, type(inner).__name__, input_data)
