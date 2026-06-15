"""Pydantic request/response models for the de-identification API."""

from __future__ import annotations

import os
import re
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from .engine import DeidMethod


def _max_text_chars() -> int:
    """Per-request character cap, from ``OPENMED_STUDIO_MAX_TEXT_LENGTH`` (default 50k).

    Read once at import (set the env var before launching the service, mirroring
    OpenMed's own ``OPENMED_SERVICE_MAX_TEXT_LENGTH`` knob). A missing, non-integer,
    or non-positive value falls back to the default so a typo can't disable the
    guard. The value is baked into ``ClinicalText``, so ``/docs`` reports the limit.
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

# Languages OpenMed ships PII models for (matches OpenMed's own service surface).
# A non-"en" value makes openmed auto-select a larger language-specific model.
Lang = Literal["en", "fr", "de", "it", "es", "nl", "hi", "te", "pt"]

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
    """Reject unknown fields so request typos fail loudly with a 422."""

    model_config = ConfigDict(extra="forbid")


class Entity(_Strict):
    label: str
    text: str
    start: int
    end: int
    confidence: float | None = None


class ExtractRequest(_Strict):
    text: ClinicalText
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    use_smart_merging: bool = True
    lang: Lang = "en"
    model_name: ModelName = None


class ExtractResponse(_Strict):
    entities: list[Entity]


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


class DeidentifyRequest(_DeidentifyOptions):
    text: ClinicalText


class DeidentifyResponse(_Strict):
    deidentified_text: str
    method: DeidMethod
    entities: list[Entity]
    mapping: dict[str, str] | None = None


class DeidentifyBatchRequest(_DeidentifyOptions):
    items: list[ClinicalText] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)


class DeidentifyBatchResponse(_Strict):
    results: list[DeidentifyResponse]


class ReidentifyRequest(_Strict):
    deidentified_text: ClinicalText
    mapping: dict[str, str] = Field(max_length=MAX_MAPPING_ENTRIES)


class ReidentifyResponse(_Strict):
    text: str


# --- OpenMed-REST compatibility surface (opt-in, off by default) -------------
# These deliberately do NOT use `_Strict`: Pydantic's default `extra="ignore"`
# lets a request carry upstream-only fields (notably `keep_alive`) without a 422,
# so an OpenMed-REST client can post unchanged.


class CompatExtractRequest(BaseModel):
    """OpenMed-REST-shaped ``/pii/extract`` body; unknown fields are ignored."""

    text: ClinicalText
    lang: str = "en"
    use_smart_merging: bool = True
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    model_name: ModelName = None
    keep_alive: str | int | None = Field(
        default=None,
        description="Accepted for OpenMed-REST parity; ignored (no model lifecycle).",
    )


class CompatDeidentifyRequest(BaseModel):
    """OpenMed-REST-shaped ``/pii/deidentify`` body; unknown fields are ignored."""

    text: ClinicalText
    method: DeidMethod = "mask"
    lang: str = "en"
    keep_mapping: bool = False
    date_shift_days: int | None = None
    keep_year: bool = True
    consistent: bool = False
    seed: int | None = None
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    model_name: ModelName = None
    keep_alive: str | int | None = Field(
        default=None,
        description="Accepted for OpenMed-REST parity; ignored (no model lifecycle).",
    )


class HealthResponse(_Strict):
    status: str
    service: str
    version: str = Field(description="openmed-studio application version.")
    model: str
    backend: str = Field(
        description="Configured inference backend: 'auto' (openmed detects — MLX on Apple "
        "Silicon when the mlx extra is installed, else HuggingFace), 'hf', or 'mlx'. Reflects "
        "the OPENMED_STUDIO_BACKEND setting, not the backend actually resolved at model load."
    )
    max_text_chars: int = Field(
        description="Per-request text length cap from OPENMED_STUDIO_MAX_TEXT_LENGTH "
        "(default 50,000)."
    )
    model_loaded: bool = Field(
        description="True once the engine has initialized its ModelLoader (on the served "
        "path, after the first /pii/* request); not a guarantee the model is resident."
    )
    auth_required: bool


class ErrorDetail(_Strict):
    code: str = Field(
        description="Machine-readable error class: 'validation_error', 'bad_request', "
        "'unauthorized', 'not_found', 'service_unavailable', or 'internal_error'."
    )
    message: str = Field(description="Human-readable explanation.")
    details: Any = Field(
        default=None,
        description="Optional structured context (e.g. the field errors for a "
        "'validation_error'); null when absent.",
    )


class ErrorResponse(_Strict):
    """Uniform error envelope returned for every non-2xx response."""

    error: ErrorDetail


__all__ = [
    "CompatDeidentifyRequest",
    "CompatExtractRequest",
    "DeidMethod",
    "DeidentifyBatchRequest",
    "DeidentifyBatchResponse",
    "DeidentifyRequest",
    "DeidentifyResponse",
    "Entity",
    "ErrorDetail",
    "ErrorResponse",
    "ExtractRequest",
    "ExtractResponse",
    "HealthResponse",
    "Lang",
    "ReidentifyRequest",
    "ReidentifyResponse",
]
