"""openmed-studio: PII / PHI de-identification for clinical text, built on OpenMed."""

from __future__ import annotations

from .engine import DEFAULT_PII_MODEL, PIIEngine

__version__ = "0.1.0"

__all__ = ["DEFAULT_PII_MODEL", "PIIEngine", "__version__"]
