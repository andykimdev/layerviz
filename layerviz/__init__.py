"""
Expose the public layerviz API through lazy imports.

Purpose:

- Define the package version and the public names capture, compare, load_state,
  register_shape.

Special Notes:

- Names load on first access, so importing layerviz never imports torch submodules
  eagerly and never imports spconv.
- The TYPE_CHECKING imports let type checkers see every public name without a runtime import.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layerviz.api import capture
    from layerviz.comparison import compare
    from layerviz.loading import load_state
    from layerviz.shapes.base import register_shape

__version__ = "0.1.0"
__all__ = ["capture", "compare", "load_state", "register_shape"]

_LAZY_SOURCES = {
    "capture": "layerviz.api",
    "compare": "layerviz.comparison",
    "load_state": "layerviz.loading",
    "register_shape": "layerviz.shapes.base",
}


def __getattr__(name: str) -> object:
    """
    Import a public name from its home module on first access.

    Args:
        name: Attribute requested on the layerviz package.

    Returns:
        The attribute from its source module.

    Raises:
        AttributeError: When name is not a public layerviz name.
    """
    if name not in _LAZY_SOURCES:
        raise AttributeError(f"module 'layerviz' has no attribute '{name}'")
    return getattr(import_module(_LAZY_SOURCES[name]), name)
