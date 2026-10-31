"""Compact, research-oriented I-JEPA implementation."""

try:
    # On Windows, importing pyarrow (via datasets) after torch has initialised segfaults
    # with a DLL conflict, so load it first. Optional: only the imagenet100 data path needs it.
    import datasets  # noqa: F401
except ImportError:
    pass

from .model import IJEPAModel
from .masks import MultiBlockMaskSampler

__all__ = ["IJEPAModel", "MultiBlockMaskSampler"]
