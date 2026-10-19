"""
Capture one model and batch as a viewable bundle and build its graph dict.

Purpose:

- Turn a pipeline result and its layout into the graph.json dict that app.js reads.
- Run the pipeline and write the bundle, into a step folder when capturing during training.

Special Notes:

- A step that is not due returns before the model is touched, and only rank 0 captures.
- Node ids, leaf edges, and coordinates come from graph.py, shared with compare bundles.
- capture takes its options as keywords and splits them into CaptureOptions and OutputOptions.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from torch import nn

from layerviz.bundle import WEB_DIR, BundleOptions, LayerPayload, write_bundle
from layerviz.graph import annotate_nodes, graph_body, graph_meta, node_id
from layerviz.pipeline import CaptureOptions, PipelineResult, run_pipeline
from layerviz.snapshots import due, is_main_process, step_dir, write_index


def build_graph(
    result: PipelineResult, labels: list[str] | None = None, *, step: int | None = None,
    label: str | None = None, sample_index: int = 0,
) -> dict:
    """
    Build the graph.json dict of nodes, boxes, edges, and meta from a pipeline result.

    Args:
        result: Pipeline result whose trace and layers become nodes, boxes, and edges.
        labels: Run labels that switch meta to compare mode, or None for single mode.
        step: Training step recorded in meta, or None outside training.
        label: Capture label recorded in meta, or None.
        sample_index: Batch index of the captured sample, recorded in meta.

    Returns:
        The graph dict with file fields null, which write_bundle fills.
    """
    body = graph_body(result.trace, {layer.node.name: layer for layer in result.layers})
    annotate_nodes(body["nodes"], result)
    return {"meta": graph_meta(result, labels, step, label, sample_index), **body}


def layer_payloads(result: PipelineResult) -> dict[str, list[LayerPayload]]:
    """
    Map each captured layer's node id to its single pane payload.

    Args:
        result: The pipeline result.

    Returns:
        Node id to a one item list holding the layer payload.
    """
    return {
        node_id(layer.node.index): [
            LayerPayload(layer.data, layer.reduced, channels=layer.channels)
        ]
        for layer in result.layers
    }


@dataclass(frozen=True)
class OutputOptions:
    """
    Choose where one capture writes its bundle and which training steps it captures.
    """

    step: int | None = None
    every: int | None = None
    label: str | None = None
    inline_mb: float = 16
    web_dir: Path = WEB_DIR

    def __post_init__(self) -> None:
        """
        Reject an interval given without the step it schedules.

        Raises:
            ValueError: When every is set and step is None.
        """
        if self.every is not None and self.step is None:
            raise ValueError(f"every needs step to schedule snapshots, got every={self.every}")


CAPTURE_FIELDS = frozenset(f.name for f in fields(CaptureOptions))
OUTPUT_FIELDS = frozenset(f.name for f in fields(OutputOptions))


def split_options(options: dict[str, Any]) -> tuple[CaptureOptions, OutputOptions]:
    """
    Split the keyword options of capture into its capture and output options.

    Args:
        options: Keyword names of CaptureOptions and OutputOptions fields.

    Returns:
        The CaptureOptions and the OutputOptions the names describe.

    Raises:
        TypeError: When a name is a field of neither, with the unknown names listed.
    """
    unknown = sorted(set(options) - CAPTURE_FIELDS - OUTPUT_FIELDS)
    if unknown:
        raise TypeError(f"capture got unknown options {unknown}, expected fields of "
                        "CaptureOptions or OutputOptions")
    capture_opts = CaptureOptions(**{k: v for k, v in options.items() if k in CAPTURE_FIELDS})
    output_opts = OutputOptions(**{k: v for k, v in options.items() if k in OUTPUT_FIELDS})
    return capture_opts, output_opts


def capture(model: nn.Module, batch: Any, out: Path | str, **options: Any) -> Path | None:
    """
    Capture one forward pass of model on batch as a bundle under out.

    Args:
        model: The module to capture, possibly wrapped by DDP or torch.compile.
        batch: The model input.
        out: The bundle folder, or the snapshot root when step is given.
        **options: Keyword options, each a CaptureOptions or OutputOptions field.

            - step: The training step, 0 or more, written into out/step_<step>.
            - every: The capture interval, which needs step. Steps off the interval are skipped.
            - label: A label shown with the capture.
            - inline_mb: Largest index.html written as one self contained page, 16 by default.
            - web_dir: Folder of the page files copied into the bundle.
            - sample_index, include, exclude, shapes, forward: The sample and layers captured.
            - layer_mb, total_mb, seed, bytes_per_position: The size caps and the sampling.

    Returns:
        Path of the written index.html, or None when the step is not due or the rank is not 0.

    Raises:
        TypeError: When an option name is a field of neither options class.
        ValueError: When every is given without step, or step is negative.

    Notes:
        The model keeps its training and grad modes, since run_pipeline restores them.
    """
    opts, output = split_options(options)
    step = output.step
    target = step_dir(Path(out), step) if step is not None else Path(out)
    if step is not None and output.every is not None and not due(step, output.every):
        return None
    if not is_main_process():
        return None
    result = run_pipeline(model, batch, opts)
    graph = build_graph(result, step=step, label=output.label, sample_index=opts.sample_index)
    limits = BundleOptions(opts.layer_mb, opts.total_mb, Path(output.web_dir))
    index = write_bundle(target, graph, layer_payloads(result), output.inline_mb, limits)
    if step is not None:
        write_index(Path(out))
    return index
