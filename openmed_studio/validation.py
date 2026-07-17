"""Pydantic request models for de-identification (and the other capabilities).

These import only ``pydantic``/``os``/``re`` (no web framework), so they serve double duty:
the in-process seam (``openmed_studio.service``) uses ``model_validate`` to enforce the
text/batch/mapping caps and value checks before any engine call, and the FastAPI service
(``openmed_studio.main``) declares them directly as request bodies (free OpenAPI schemas +
automatic 422s). The HTTP-only *response*/error/health/compat models live in ``main.py``,
not here, so this module stays framework-free.
"""

from __future__ import annotations

import os
import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from .engine import DeidMethod, Policy


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
# Zero-shot extraction runs one forward pass per label and GLiNER's accuracy degrades with
# very large label sets, so cap the count (and each label's length) the way MAX_BATCH_ITEMS
# caps notes.
MAX_ZERO_SHOT_LABELS = 30
MAX_ZERO_SHOT_LABEL_CHARS = 80

# Languages OpenMed ships PII models for (openmed.core.pii_i18n.SUPPORTED_LANGUAGES).
# A non-"en" value makes openmed auto-select a larger language-specific model.
# test_validation_lang_subset_of_openmed keeps this from drifting past what openmed supports.
Lang = Literal["en", "fr", "de", "it", "es", "nl", "hi", "te", "pt", "ar", "ja", "tr"]

# Strip surrounding whitespace, then require 1..MAX_TEXT_CHARS chars — this also
# rejects whitespace-only input (it strips to empty and fails min_length). Note the
# de-identified/re-identified output is therefore stripped at the edges too; internal
# whitespace and newlines are preserved.
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

# A *required* model id (same format check, but not optional): clinical NER is one model
# per domain, so an absent model_name would silently fall back to openmed's disease-only
# default — make callers pick one explicitly.
RequiredModelName = Annotated[str, AfterValidator(_check_model_name)]


_LOCALE_RE = re.compile(r"[A-Za-z]{2,3}(?:_[A-Za-z0-9]{2,8})?")


def _check_locale(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not _LOCALE_RE.fullmatch(value):
        raise ValueError("locale must look like 'en_US' or 'pt_BR'")
    return value


# An optional Faker locale for `replace` surrogates, format-validated. openmed/Faker
# validate that the locale actually exists at call time; this only guards the shape.
LocaleName = Annotated[str | None, AfterValidator(_check_locale)]


def _check_zero_shot_labels(values: list[str]) -> list[str]:
    """Normalize the user's zero-shot labels: strip, drop empties, dedup, cap.

    Runs after each item is coerced to ``str``. Strips surrounding whitespace, drops blanks,
    bounds each label's length, and dedups case-insensitively (a repeated label is harmless
    — unlike an unknown *field*, which ``extra="forbid"`` still rejects — so we quietly
    collapse duplicates rather than error). Requires at least one surviving label and caps
    the total at :data:`MAX_ZERO_SHOT_LABELS`. Errors name the cap, never a label value.
    """
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        if len(value) > MAX_ZERO_SHOT_LABEL_CHARS:
            raise ValueError(
                f"each label must be at most {MAX_ZERO_SHOT_LABEL_CHARS} characters"
            )
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    if not out:
        raise ValueError("provide at least one entity label")
    if len(out) > MAX_ZERO_SHOT_LABELS:
        raise ValueError(f"at most {MAX_ZERO_SHOT_LABELS} labels are allowed")
    return out


# The user's arbitrary zero-shot entity labels, normalized/deduped/capped. A plain
# list[str] with one AfterValidator (matching _check_model_name/_check_locale) so the
# whole set is validated together after per-item string coercion.
ZeroShotLabels = Annotated[list[str], AfterValidator(_check_zero_shot_labels)]


class _Strict(BaseModel):
    """Reject unknown fields so request typos fail loudly with a validation error."""

    model_config = ConfigDict(extra="forbid")


class ExtractRequest(_Strict):
    text: ClinicalText
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    use_smart_merging: bool = True
    lang: Lang = "en"
    model_name: ModelName = None


class NerRequest(_Strict):
    """Clinical NER (token-classification) detection request.

    Reuses ``ClinicalText`` and the ``model_name`` format guard, but its field set
    differs from de-identification: ``model_name`` is required (NER is one model per
    domain), the confidence default is ``0.0`` (openmed's NER default keeps all), and
    there is no ``lang``/``use_smart_merging`` (``analyze_text`` has neither).
    """

    text: ClinicalText
    model_name: RequiredModelName
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    aggregation_strategy: Literal["simple", "first", "average", "max"] = "simple"
    group_entities: bool = False


class ZeroShotRequest(_Strict):
    """Zero-shot (GLiNER) extraction request: arbitrary labels + a domain-tuned model.

    Like :class:`NerRequest`, ``model_name`` is required (each GLiNER checkpoint is
    domain-tuned; the UI passes a :data:`~openmed_studio.engine.ZERO_SHOT_MODELS` alias) and
    there is no ``lang``. Unlike it, the user supplies ``labels`` — normalized, deduped, and
    capped by ``ZeroShotLabels`` — and the confidence default is ``0.6`` (GLiNER's own
    recommendation), not NER's ``0.0``.
    """

    text: ClinicalText
    model_name: RequiredModelName
    labels: ZeroShotLabels
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class AnonymizePolicyRequest(_Strict):
    """Policy-driven anonymization: a named regulatory profile picks the per-label action.

    Distinct from :class:`DeidentifyRequest` in two deliberate ways. It has **no ``method``**:
    a ``policy`` overrides the flat method (openmed assigns a per-label action from the profile),
    so exposing a method here would be a control the policy silently ignores. And it has **no
    ``keep_mapping``**: reversibility is the policy's decision (surrogate profiles keep a mapping,
    masking ones don't), so the seam always requests it and surfaces whatever the policy yields.
    ``policy`` is a required closed :data:`~openmed_studio.engine.Policy` Literal (mirroring
    ``RequiredModelName``'s "make the caller choose" rationale), so an unknown/typo'd policy is
    rejected here — with a PHI-safe message — before the engine. The surrogate knobs
    ``consistent``/``seed``/``locale`` apply to the ``replace``-based (reversible) policies.
    """

    text: ClinicalText
    policy: Policy
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    use_smart_merging: bool = True
    lang: Lang = "en"
    model_name: ModelName = None
    consistent: bool = False
    seed: int | None = None
    locale: LocaleName = None
    use_safety_sweep: bool = True


class _DeidentifyOptions(_Strict):
    """Shared de-identification options (single + batch requests)."""

    method: DeidMethod = "mask"
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # Recombine token-fragmented PII (dates, SSNs) into whole spans — matches the
    # Detect tab / extract_pii, which exposes this too. Defaults on, as openmed does.
    use_smart_merging: bool = True
    lang: Lang = "en"
    model_name: ModelName = None
    keep_mapping: bool = False
    consistent: bool = False
    seed: int | None = None
    # Faker locale for the surrogate methods (e.g. 'pt_BR' for Brazilian CPF/CNPJ),
    # overriding the default openmed derives from `lang`. Used by method='replace'
    # and method='format_preserve'.
    locale: LocaleName = None
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
    "AnonymizePolicyRequest",
    "DeidMethod",
    "DeidentifyBatchRequest",
    "DeidentifyRequest",
    "ExtractRequest",
    "Lang",
    "NerRequest",
    "Policy",
    "ReidentifyRequest",
    "ZeroShotRequest",
]
