"""Pydantic request/response models for the de-identification API."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from .engine import DeidMethod

# Bounds that keep a single request from pinning the shared model worker.
MAX_TEXT_CHARS = 50_000
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


class HealthResponse(_Strict):
    status: str
    service: str
    model: str
    backend: str = Field(
        description="Configured inference backend: 'auto' (openmed detects — MLX on Apple "
        "Silicon when the mlx extra is installed, else HuggingFace), 'hf', or 'mlx'. Reflects "
        "the OPENMED_STUDIO_BACKEND setting, not the backend actually resolved at model load."
    )
    model_loaded: bool = Field(
        description="True once the engine has initialized its ModelLoader (on the served "
        "path, after the first /pii/* request); not a guarantee the model is resident."
    )
    auth_required: bool


__all__ = [
    "DeidMethod",
    "DeidentifyBatchRequest",
    "DeidentifyBatchResponse",
    "DeidentifyRequest",
    "DeidentifyResponse",
    "Entity",
    "ExtractRequest",
    "ExtractResponse",
    "HealthResponse",
    "Lang",
    "ReidentifyRequest",
    "ReidentifyResponse",
]
