"""openmed-studio: PII / PHI de-identification for clinical text, built on OpenMed."""

from __future__ import annotations

from .engine import (
    DEFAULT_NER_MODEL,
    DEFAULT_PII_MODEL,
    DEFAULT_ZERO_SHOT_MODEL,
    NER_MODELS,
    ZERO_SHOT_MODELS,
    PIIEngine,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_NER_MODEL",
    "DEFAULT_PII_MODEL",
    "DEFAULT_ZERO_SHOT_MODEL",
    "NER_MODELS",
    "ZERO_SHOT_MODELS",
    "PIIEngine",
    "__version__",
]
