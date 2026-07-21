"""High-level one-call API: build the console, fuse, and budget-select a pool."""
from .select import SelectionResult, build_console, select_pool  # noqa: F401

__all__ = ["select_pool", "build_console", "SelectionResult"]
