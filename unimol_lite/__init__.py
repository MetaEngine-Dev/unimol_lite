"""unimol_lite: lightweight Uni-Mol featurization and model toolkit."""

# ``data`` is imported first on purpose: ``unimol_lite.batch`` imports
# ``.data.prepare_unimol_item``, and ``data/__init__`` re-exports
# ``featurize_batch`` — initializing ``data`` first keeps the cycle one-way.
from . import data  # noqa: F401
from .batch import featurize_batch

__all__ = ["featurize_batch"]
