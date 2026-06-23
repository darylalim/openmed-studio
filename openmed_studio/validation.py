"""Pydantic request models for in-process de-identification.

These import only ``pydantic``/``os``/``re`` (no web framework), so the Streamlit
app reuses them as the in-process validation seam (``openmed_studio.service``):
``model_validate`` enforces the text/batch/mapping caps and value checks before any
engine call. They were the FastAPI request bodies; the HTTP-only response/error/
health/compat models were removed with the service.
"""

from __future__ import annotations

import os
import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from .engine import DeidMethod


def _max_text_chars() -> int:
    """Per-request character cap, from ``OPENMED_STUDIO_MAX_TEXT_LENGTH`` (default 50k).

    Read once at import (set the env var before launching the app, mirroring
    OpenMed's own ``OPENMED_SERVICE_MAX_TEXT_LENGTH`` knob). A missing, non-integer,
    or non-positive value falls back to the default so a typo can't disable the
    guard. The value is baked into ``ClinicalText``.
    """
    raw = os.environ.get("OPENMED_STUDIO_MAX_TEXT_LENGTH")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return 50_000


# Bounds that keep a single request from pinning the shared model worker.
MAX_TEXT_CHARS = _max_text_chars()
MAX_BATCH_ITEMS = 100
MAX_MAPPING_ENTRIES = 5_000

# Languages OpenMed ships PII models for (openmed.core.pii_i18n.SUPPORTED_LANGUAGES).
# A non-"en" value makes openmed auto-select a larger language-specific model.
# test_validation_lang_subset_of_openmed keeps this from drifting past what openmed supports.
Lang = Literal["en", "fr", "de", "it", "es", "nl", "hi", "te", "pt", "ar", "ja", "tr"]

# Strip surrounding whitespace, then require 1..MAX_TEXT_CHARS chars — this also
# rejects whitespace-only input (it strips to empty and fails min_length).
ClinicalText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS),
]

_MODEL_NAME_RE = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)?")


def _check_model_name(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not _MODEL_NAME_RE.fullmatch(value):
        raise ValueError("model_name must look like 'org/model' or 'model'")
    return value


# An optional HF/registry model id, format-validated (no path traversal / spaces).
ModelName = Annotated[str | None, AfterValidator(_check_model_name)]


class _Strict(BaseModel):
    """Reject unknown fields so request typos fail loudly with a validation error."""

    model_config = ConfigDict(extra="forbid")


class ExtractRequest(_Strict):
    text: ClinicalText
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    use_smart_merging: bool = True
    lang: Lang = "en"
    model_name: ModelName = None


class _DeidentifyOptions(_Strict):
    """Shared de-identification options (single + batch requests)."""

    method: DeidMethod = "mask"
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    lang: Lang = "en"
    model_name: ModelName = None
    keep_mapping: bool = False
    consistent: bool = False
    seed: int | None = None
    date_shift_days: int | None = Field(
        default=None, description="Only used with method='shift_dates'."
    )
    keep_year: bool = True
    # openmed 1.6.0 runs a deterministic structured-identifier sweep after model
    # detection (default on); pinned here so the behavior is explicit, not silently
    # inherited from openmed's default.
    use_safety_sweep: bool = True


class DeidentifyRequest(_DeidentifyOptions):
    text: ClinicalText


class DeidentifyBatchRequest(_DeidentifyOptions):
    items: list[ClinicalText] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)


class ReidentifyRequest(_Strict):
    deidentified_text: ClinicalText
    mapping: dict[str, str] = Field(max_length=MAX_MAPPING_ENTRIES)


__all__ = [
    "DeidMethod",
    "DeidentifyBatchRequest",
    "DeidentifyRequest",
    "ExtractRequest",
    "Lang",
    "ReidentifyRequest",
]
