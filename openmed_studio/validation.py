"""Pydantic request models for de-identification (and the other capabilities).

These import only ``pydantic`` and the standard library, plus the registry data baked into
``openmed_studio.engine`` (no web framework, no ``openmed``), so they serve double duty:
the in-process seam (``openmed_studio.service``) uses ``model_validate`` to enforce the
text/batch/mapping caps, value checks and per-capability ``model_name`` allowlists before
any engine call, and the FastAPI service (``openmed_studio.main``) declares them directly
as request bodies (free OpenAPI schemas + automatic 422s). The HTTP-only
*response*/error/health/compat models live in ``main.py``, not here, so this module stays
framework-free.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints
from pydantic.fields import FieldInfo

from .engine import (
    DEFAULT_PII_MLX_MODEL,
    DEFAULT_PII_MODEL,
    NER_MODELS,
    ZERO_SHOT_MODELS,
    DeidMethod,
    Policy,
)

EXTRA_MODELS_ENV = "OPENMED_STUDIO_EXTRA_MODELS"


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
#
# This is a deliberately CURATED SUBSET, not a mirror: openmed 2.x took SUPPORTED_LANGUAGES from
# 12 to 35 (36 as of 2.5, whose new `fa` also defaults to `OpenMed/privacy-filter-multilingual`),
# but most of the additions route through openmed's privacy-filter family, and that path is
# incompatible with how this app runs models. `extract_pii`'s `uses_privacy_filter` branch
# (openmed/core/pii.py) builds its pipeline via `create_privacy_filter_pipeline` and never
# touches `loader=`, `openmed/torch/privacy_filter.py` memoizes only the tokenizer (via
# `tokenizer_cache`), never the model (so the model is rebuilt from scratch on EVERY call), and
# it loads with `trust_remote_code=True`. Adding those languages here would silently bypass the
# shared ModelLoader, the engine's `OpenMedConfig` pins, and `engine.is_loaded`. Widening the list
# means fixing that first (thread `config=` through and cache the pipeline), not editing the
# Literal. The guard is `<=` precisely so a curated subset stays legal. Every `lang` field —
# the /compat bodies in main.py included — uses this Literal, so none of those languages
# (whose defaults openmed swaps in for the default model) is reachable through the app.
Lang = Literal["en", "fr", "de", "it", "es", "nl", "hi", "te", "pt", "ar", "ja", "tr"]

# Strip surrounding whitespace, then require 1..MAX_TEXT_CHARS chars — this also
# rejects whitespace-only input (it strips to empty and fails min_length). Note the
# de-identified/re-identified output is therefore stripped at the edges too; internal
# whitespace and newlines are preserved.
ClinicalText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS),
]

# The shape of an HF repo id ("org/model") or an openmed registry alias ("model"): one or
# two "/"-separated segments of [A-Za-z0-9._-]. _model_name_format adds one rule on top:
# no segment may start with ".". The charset has no "~", "\", ":" or space and a name
# can't open with "/", so no absolute or home-relative path passes, and the one-"/" cap
# bounds the depth. The dot rule then rejects "." and ".." segments ("..", "../x", "x/..")
# and hidden entries (".venv", ".streamlit/config.toml"). It costs no real model: the Hub
# forbids a leading "." in a repo id, and no openmed registry alias or model id has one
# (the tests pin it). Both messages name the rule, never the value.
_MODEL_NAME_RE = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)?")


def _model_name_format(value: str) -> str:
    value = value.strip()
    if not _MODEL_NAME_RE.fullmatch(value):
        raise ValueError("model_name must look like 'org/model' or 'model'")
    if any(segment.startswith(".") for segment in value.split("/")):
        raise ValueError("model_name segments must not start with '.'")
    return value


def _check_model_name(value: str | None) -> str | None:
    """The format check every ``model_name`` field (and every extra model) runs first."""
    if value is None:
        return None
    return _model_name_format(value)


def _extra_models() -> frozenset[str]:
    """Operator-added model ids, from ``OPENMED_STUDIO_EXTRA_MODELS`` (comma-separated).

    Read once at import, like :func:`_max_text_chars` (set the env var before launching
    the app). Entries are whitespace-trimmed and empty ones dropped, so a trailing comma
    is harmless; each must then pass :func:`_check_model_name`. Unlike the text cap, where
    a typo falls back to a safe default, a malformed entry raises and so stops the app at
    startup: an allowlist has no safe guess to fall back to — dropping the entry would
    silently refuse a model the operator meant to allow. The error names the entry (it is
    the operator's own config, printed to the operator's console, never to a caller).

    Every id listed here is accepted on EVERY ``model_name`` field, matched exactly (case
    included), and the operator owns what it loads: openmed downloads an unregistered Hub
    id without an integrity check, and adding a first-party privacy-filter repo
    (``openai/privacy-filter``, ``OpenMed/privacy-filter-*``) re-opens openmed's
    ``trust_remote_code=True`` path (``core/pii.py`` routes those names to
    ``create_privacy_filter_pipeline``).
    """
    names: set[str] = set()
    for entry in os.environ.get(EXTRA_MODELS_ENV, "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            names.add(_model_name_format(entry))
        except ValueError as exc:
            raise ValueError(
                f"{EXTRA_MODELS_ENV} entry {entry!r} is not a valid model id: {exc}"
            ) from None
    return frozenset(names)


# The operator's extra model ids and, from them plus engine.py's baked registry data, the
# per-capability allowlists every model_name field is checked against (after the format
# check). Exact string match, case included: openmed's registry lookup
# (core/model_registry.py::get_model_info) is exact, so a case-variant of an allowed id
# isn't recognized as that registry model — it would skip openmed's registry-backed
# integrity check and get its own cache entry — while openmed's privacy-filter trust check
# matches case-INsensitively. No variant slips through to a different resolution.
EXTRA_MODELS = _extra_models()
# PII (every de-identification route and both /compat bodies): the default model plus its
# pre-converted MLX build. model_name=None still means "the default" — including openmed's
# per-language default, which it swaps in for the default model when lang != "en".
_CURATED_PII_MODELS = (DEFAULT_PII_MODEL, DEFAULT_PII_MLX_MODEL)
PII_MODEL_NAMES = frozenset(_CURATED_PII_MODELS) | EXTRA_MODELS
# Clinical NER: the curated NER_MODELS aliases only — not also each alias's HF repo id.
# The UI sends aliases, and the baked NerModel (pinned by the drift guard) carries only the
# alias, so accepting repo ids would mean baking a second id per model (or importing
# openmed here) for a spelling no caller uses; one accepted name per model also keeps the
# allowlist auditable at a glance. An API caller who wants a repo id gets it through
# OPENMED_STUDIO_EXTRA_MODELS.
_CURATED_NER_MODELS = tuple(m.alias for m in NER_MODELS.values())
NER_MODEL_NAMES = frozenset(_CURATED_NER_MODELS) | EXTRA_MODELS
# Zero-shot: the curated ZERO_SHOT_MODELS aliases. PIIEngine.extract_zero_shot re-checks
# this same set, since it would otherwise resolve any of openmed's registry aliases.
_CURATED_ZERO_SHOT_MODELS = tuple(m.alias for m in ZERO_SHOT_MODELS.values())
ZERO_SHOT_MODEL_NAMES = frozenset(_CURATED_ZERO_SHOT_MODELS) | EXTRA_MODELS


def _allowed_in(
    allowed: frozenset[str], capability: str
) -> Callable[[str | None], str | None]:
    """Build one capability's allowlist check (it runs after :func:`_check_model_name`)."""

    def check(value: str | None) -> str | None:
        if value is None or value in allowed:
            return value
        # Name the rule and the knob, never the value: a model_name field is one stray
        # paste away from holding note text. The allowed ids aren't listed either — the
        # operator's extras are their own business, not an anonymous caller's.
        raise ValueError(
            f"model_name is not an allowed {capability} model (an operator can allow "
            f"more with {EXTRA_MODELS_ENV})"
        )

    return check


def _model_name_schema(curated: tuple[str, ...], what: str) -> FieldInfo:
    """OpenAPI metadata listing a capability's CURATED names (``description`` + ``examples``).

    A 422 deliberately names no allowed id, so the schema is where an API caller discovers
    them (``GET /openapi.json``, ``/docs``). Built from the curated tuples only: the
    operator's ``OPENMED_STUDIO_EXTRA_MODELS`` stay out of the published schema, as they stay
    out of the error message. No ``enum``: an extra is accepted too, and an enum would make
    a schema-validating client refuse it.
    """
    return Field(
        description=f"{what} One of: {', '.join(curated)}. The operator may allow more "
        f"with {EXTRA_MODELS_ENV}; those are not listed here.",
        examples=list(curated),
    )


# The per-capability model_name types. Each runs the format check FIRST (charset, one "/",
# no "."-leading segment — its messages name the rule, never the value) and the allowlist
# SECOND. The format check alone left any format-valid Hub id loadable — a download openmed
# doesn't verify and ModelLoader._pipelines never evicts — and, on the PII routes, the
# first-party privacy-filter names (openai/privacy-filter, OpenMed/privacy-filter-*, in any
# casing) that openmed loads with trust_remote_code=True.
#
# RESIDUAL — the allowlist fixes the NAME, not what it resolves to. openmed resolves a
# name against the filesystem BEFORE its registry or the Hub
# (core/models.py::_resolve_model_name), relative to the process's working directory, so
# a directory named like an allowed model would shadow it — and on the PII routes a local
# dir whose config.json names the privacy-filter family is loaded with
# trust_remote_code=True (core/backends.py::create_privacy_filter_pipeline). Nothing in a
# request can tell the two apart, so the ENGINE closes it instead: every model call first
# refuses (engine.LocalModelPathError, a 503) when a name it would hand openmed — or an
# "OpenMed"/"openai" entry — exists in the working directory. What's left is a race (a
# directory created between that check and openmed's own) and the operator's extras;
# both are why the app must run from a directory nobody else can write to.
#
# An optional PII model id; None means openmed's default.
PiiModelName = Annotated[
    str | None,
    _model_name_schema(
        _CURATED_PII_MODELS,
        "PII model id; omit it for openmed's default (the per-language default when "
        "lang isn't 'en').",
    ),
    AfterValidator(_check_model_name),
    AfterValidator(_allowed_in(PII_MODEL_NAMES, "PII")),
]

# A *required* clinical-NER model id (not optional): NER is one model per domain, so an
# absent model_name would silently fall back to openmed's disease-only default — make
# callers pick one explicitly.
NerModelName = Annotated[
    str,
    _model_name_schema(
        _CURATED_NER_MODELS, "Clinical NER model: the alias of one domain's model."
    ),
    AfterValidator(_check_model_name),
    AfterValidator(_allowed_in(NER_MODEL_NAMES, "clinical NER")),
]

# A *required* zero-shot model id, for the same reason (each GLiNER checkpoint is
# domain-tuned).
ZeroShotModelName = Annotated[
    str,
    _model_name_schema(
        _CURATED_ZERO_SHOT_MODELS,
        "Zero-shot (GLiNER) model: the alias of one domain's checkpoint.",
    ),
    AfterValidator(_check_model_name),
    AfterValidator(_allowed_in(ZERO_SHOT_MODEL_NAMES, "zero-shot")),
]


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
    model_name: PiiModelName = None


class NerRequest(_Strict):
    """Clinical NER (token-classification) detection request.

    Reuses ``ClinicalText``, but its field set differs from de-identification:
    ``model_name`` is required and must be a curated
    :data:`~openmed_studio.engine.NER_MODELS` alias (NER is one model per domain), the
    confidence default is ``0.0`` (openmed's NER default keeps all), and there is no
    ``lang``/``use_smart_merging`` (``analyze_text`` has neither).
    """

    text: ClinicalText
    model_name: NerModelName
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    aggregation_strategy: Literal["simple", "first", "average", "max"] = "simple"
    group_entities: bool = False


class ZeroShotRequest(_Strict):
    """Zero-shot (GLiNER) extraction request: arbitrary labels + a domain-tuned model.

    Like :class:`NerRequest`, ``model_name`` is required (each GLiNER checkpoint is
    domain-tuned) and must be a :data:`~openmed_studio.engine.ZERO_SHOT_MODELS` alias, and
    there is no ``lang``. Unlike it, the user supplies ``labels`` — normalized, deduped, and
    capped by ``ZeroShotLabels`` — and the confidence default is ``0.6`` (GLiNER's own
    recommendation), not NER's ``0.0``.
    """

    text: ClinicalText
    model_name: ZeroShotModelName
    labels: ZeroShotLabels
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class AnonymizePolicyRequest(_Strict):
    """Policy-driven anonymization: a named regulatory profile picks the per-label action.

    Distinct from :class:`DeidentifyRequest` in two deliberate ways. It has **no ``method``**:
    a ``policy`` overrides the flat method (openmed assigns a per-label action from the profile),
    so exposing a method here would be a control the policy silently ignores. And it has **no
    ``keep_mapping``**: reversibility is the policy's decision (by the profile's own flag — some
    surrogate profiles keep no key), so the seam surfaces whatever mapping the policy yields.
    ``policy`` is a required closed :data:`~openmed_studio.engine.Policy` Literal (mirroring
    ``NerModelName``'s "make the caller choose" rationale), so an unknown/typo'd policy —
    or one of openmed's :data:`~openmed_studio.engine.HIDDEN_POLICIES` — is rejected here, with a
    PHI-safe message, before the engine. The surrogate knobs ``consistent``/``seed``/``locale``
    apply to the ``replace``-based policies.
    """

    text: ClinicalText
    policy: Policy
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    use_smart_merging: bool = True
    lang: Lang = "en"
    model_name: PiiModelName = None
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
    model_name: PiiModelName = None
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
    "EXTRA_MODELS",
    "NER_MODEL_NAMES",
    "PII_MODEL_NAMES",
    "ZERO_SHOT_MODEL_NAMES",
    "AnonymizePolicyRequest",
    "DeidMethod",
    "DeidentifyBatchRequest",
    "DeidentifyRequest",
    "ExtractRequest",
    "Lang",
    "NerModelName",
    "NerRequest",
    "PiiModelName",
    "Policy",
    "ReidentifyRequest",
    "ZeroShotModelName",
    "ZeroShotRequest",
]
