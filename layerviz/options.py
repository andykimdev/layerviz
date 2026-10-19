"""
Hold the options of one capture in the lowest layer, so every module can import them.

Purpose:

- Choose the sample, the layers, the plugin overrides, and the size caps of one capture.

Special Notes:

- bytes_per_position defaults to None, the single mode size, because this module cannot
  import budget.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass
class CaptureOptions:
    """
    Choose the sample, the layers, the plugin overrides, and the size caps of one capture.

    """

    sample_index: int = 0
    include: list[str] | None = None
    exclude: list[str] | None = None
    shapes: dict[str, str] | None = None
    forward: Callable[[nn.Module, Any], Any] | None = None
    layer_mb: float = 16
    total_mb: float = 1024
    seed: int = 0
    # None means the single mode size. Compare mode passes COMPARE_BYTES_PER_POSITION.
    bytes_per_position: int | None = None
