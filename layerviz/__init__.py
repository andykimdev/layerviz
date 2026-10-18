"""
Expose the public layerviz API through lazy imports.

Purpose:

- Define the package version and the public names capture, compare, register_shape.

Special Notes:

- Names load on first access, so importing layerviz never imports torch submodules
  eagerly and never imports spconv.
"""

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["capture", "compare", "register_shape"]

_LAZY_SOURCES = {
    "capture": "layerviz.api",
    "compare": "layerviz.comparison",
    "register_shape": "layerviz.shapes.base",
}


def __getattr__(name):
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
