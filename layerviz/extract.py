"""
Turn the model input and each traced module output into layers for the pipeline.

Purpose:

- Describe the model input as a shape context and as a layer of its own.
- Extract one layer per selected module, inheriting sparse positions and naming skips.

Special Notes:

- Reads tensors only and writes no files. A layer no plugin accepts, or one holding NaN
  or inf, gets a skip reason instead of failing the run.
"""

from collections import deque
from dataclasses import replace
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from layerviz.reduce import bad_values
from layerviz.shapes import sparse
from layerviz.shapes.base import LayerData, ShapeContext, plugin_for
from layerviz.trace import INPUT, Trace, TraceNode

if TYPE_CHECKING:
    from layerviz.options import CaptureOptions

INHERITABLE_KINDS = ("sparse3d", "points")
SPARSE_FIELDS = ("features", "indices", "spatial_shape", "batch_size")


def _is_point_pair(batch: Any) -> bool:
    """Return whether the batch is a (coords [B, N, D], feats [B, N, C]) pair."""
    return (
        isinstance(batch, (tuple, list))
        and len(batch) == 2
        and all(isinstance(t, torch.Tensor) and t.dim() == 3 for t in batch)
    )


def batch_size_of(batch: Any) -> int | None:
    """
    Return the sample count of a dense batch, a sparse batch, or a point pair.

    Args:
        batch: The model input.

    Returns:
        The number of samples, or None for an input whose batch axis is unknown.
    """
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0]) if batch.dim() >= 1 else None
    if all(hasattr(batch, name) for name in SPARSE_FIELDS):
        return int(batch.batch_size)
    if _is_point_pair(batch):
        return int(batch[0].shape[0])
    return None


def check_sample_index(batch: Any, sample_index: int) -> None:
    """
    Reject a sample index that names no sample of the batch.

    Args:
        batch: The model input.
        sample_index: The batch sample the layers show.

    Raises:
        ValueError: When sample_index is negative or not below the batch size.
    """
    size = batch_size_of(batch)
    if size is not None and not 0 <= sample_index < size:
        raise ValueError(
            f"sample_index {sample_index} is outside the batch of {size} samples, "
            f"use 0 to {size - 1}"
        )


def input_context(batch: Any, sample_index: int) -> ShapeContext:
    """
    Build the shape context that describes the model input.

    Args:
        batch: A dense tensor, a sparse tensor object, or a (coords, feats) pair.
        sample_index: The batch sample the layers show.

    Returns:
        ShapeContext with the input grid of a dense or sparse batch and the input
        positions and row count of a sparse batch or a point pair.

    Raises:
        ValueError: When sample_index is outside the batch.
    """
    check_sample_index(batch, sample_index)
    if isinstance(batch, torch.Tensor):
        grid = tuple(batch.shape[2:]) if batch.dim() >= 4 else None
        return ShapeContext(sample_index, grid, None, None)
    if all(hasattr(batch, name) for name in SPARSE_FIELDS):
        rows = batch.indices[:, 0] == sample_index
        positions = batch.indices[rows, 1:].cpu().numpy().astype(np.float32)
        grid = tuple(int(size) for size in batch.spatial_shape)
        return ShapeContext(sample_index, grid, positions, len(positions))
    if _is_point_pair(batch):
        positions = batch[0][sample_index].detach().cpu().numpy().astype(np.float32)
        return ShapeContext(sample_index, None, positions, len(positions))
    return ShapeContext(sample_index, None, None, None)


def input_layer(batch: Any, ctx: ShapeContext) -> LayerData | None:
    """
    Return the model input as a layer, using the feats of a point pair.

    Args:
        batch: The model input.
        ctx: The shape context built from the batch.

    Returns:
        LayerData from the first plugin that accepts the input, or None.
    """
    target = batch[1] if _is_point_pair(batch) else batch
    plugin = plugin_for(target, ctx)
    return None if plugin is None else plugin.extract(target, ctx)


def selected(name: str, opts: "CaptureOptions") -> bool:
    """
    Return whether a module name passes the include and exclude patterns.

    Args:
        name: The dotted module name.
        opts: The capture options holding the include and exclude patterns.

    Returns:
        True when the name matches an include pattern, or none are set, and no exclude.
    """
    if opts.include is not None and not any(fnmatch(name, p) for p in opts.include):
        return False
    return not any(fnmatch(name, p) for p in opts.exclude or [])


def shape_override(name: str, shapes: dict[str, str] | None) -> str | None:
    """
    Return the plugin name of the first shapes pattern matching the module name.

    Args:
        name: The dotted module name.
        shapes: Plugin name by module name pattern, or None.

    Returns:
        The plugin name, or None when no pattern matches.
    """
    matches = (plugin for pattern, plugin in (shapes or {}).items() if fnmatch(name, pattern))
    return next(matches, None)


def sample_rows(output: Any, sample_index: int) -> torch.Tensor | None:
    """
    Return the boolean mask of the sample's rows in a sparse output, or None.

    Args:
        output: A module output, sparse when it has the sparse tensor fields.
        sample_index: The batch index of the captured sample.

    Returns:
        A boolean mask over the sparse rows, or None for a non sparse output.
    """
    is_sparse = all(hasattr(output, field) for field in SPARSE_FIELDS)
    return output.indices[:, 0] == sample_index if is_sparse else None


def nearest_inheritable(
    name: str,
    parents: dict[str, list[str]],
    found: dict[str, LayerData],
    shares: dict[str, str | None],
) -> str | None:
    """
    Return the name of the nearest sparse or points ancestor along the edges.

    Args:
        name: The module whose ancestors are searched.
        parents: Producer names of each node, from the trace edges.
        found: Layers extracted so far, by node name, including the input.
        shares: For each node, the child whose output it returns, or None.

    Returns:
        The name in found of the nearest layer of kind sparse3d or points, or None.
    """
    queue, seen = deque(parents.get(name, [])), {name}
    while queue:
        source = queue.popleft()
        if source in seen:
            continue
        seen.add(source)
        owner = source
        while (child := shares.get(owner)) is not None:
            owner = child
        layer = found.get(owner)
        if layer is not None and layer.kind in INHERITABLE_KINDS:
            return owner
        queue.extend(parents.get(source, []))
    return None


def _as_matrix(data: LayerData) -> LayerData:
    """Return the layer with 1D features turned into one channel per position."""
    if data.features.ndim != 1:
        return data
    return replace(data, features=data.features.reshape(len(data.positions), -1))


def layer_data(
    node: TraceNode,
    ctx: ShapeContext,
    override: str | None,
    ancestor: LayerData | None,
    rows: torch.Tensor | None,
) -> tuple[LayerData | None, str | None]:
    """
    Turn one module output into a layer, or give the reason no plugin accepts it.

    Args:
        node: The traced module with its output.
        ctx: The shape context of the input.
        override: Plugin name forced by the shapes option, or None.
        ancestor: Layer of the nearest sparse or points ancestor, or None.
        rows: Mask of the chosen sample's rows, applied to an output with all ancestor rows.

    Returns:
        Tuple of the LayerData and None, or None and the skip reason.
    """
    data, output = None, node.output
    if override is None and ancestor is not None:
        if rows is not None and output.shape[0] == len(rows):
            output = output[rows]
        data = sparse.inherit_positions(output, ancestor)
    if data is None:
        plugin = plugin_for(node.output, ctx, override)
        if plugin is None:
            return None, f"no shape plugin accepts {type(node.output).__name__}"
        try:
            extracted = plugin.extract(node.output, ctx)
        except ValueError as error:
            # A grid whose rank differs from the input grid cannot be placed, so the
            # run skips that layer with the reason instead of failing.
            return None, str(error)
        data = _as_matrix(extracted)
    return data, None


def skip_reason(data: LayerData) -> str | None:
    """
    Return why an extracted layer is skipped, when it is empty or holds NaN or inf.

    Args:
        data: The extracted layer.

    Returns:
        The skip reason, or None when the layer is kept.
    """
    if len(data.positions) == 0:
        return "no positions in this sample"
    bad = bad_values(data.features)
    return f"features hold {bad}" if bad else None


def extract_layers(
    trace: Trace, ctx: ShapeContext, opts: "CaptureOptions", input_data: LayerData | None
) -> tuple[list[tuple[TraceNode, LayerData]], dict[str, str]]:
    """
    Extract a layer from every selected module in execution order.

    Args:
        trace: The traced forward pass.
        ctx: The shape context of the input.
        opts: The capture options.
        input_data: The input as a layer, a position source for plain [N, C] outputs.

    Returns:
        Tuple of the accepted (node, layer) pairs and the skip reason by node name.
        A node that returns its child's output gets neither. A skipped layer still gives
        positions to its descendants.
    """
    parents: dict[str, list[str]] = {}
    for source, target in trace.edges:
        parents.setdefault(target, []).append(source)
    shares: dict[str, str | None] = {node.name: node.shares for node in trace.nodes}
    rows = {node.name: sample_rows(node.output, opts.sample_index) for node in trace.nodes}
    found = {INPUT: input_data} if input_data is not None else {}
    accepted, skipped = [], {}
    for node in trace.nodes:
        if node.name == INPUT or node.shares is not None or not selected(node.name, opts):
            continue
        override = shape_override(node.name, opts.shapes)
        is_matrix = isinstance(node.output, torch.Tensor) and node.output.dim() == 2
        source = nearest_inheritable(node.name, parents, found, shares) if is_matrix else None
        ancestor = found.get(source) if source is not None else None
        ancestor_rows = rows.get(source) if source is not None else None
        data, reason = layer_data(node, ctx, override, ancestor, ancestor_rows)
        if data is not None:
            found[node.name] = data
            reason = skip_reason(data)
        if reason is None:
            accepted.append((node, data))
        else:
            skipped[node.name] = reason
    return accepted, skipped
