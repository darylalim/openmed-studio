"""In-process service seam over :class:`PIIEngine`: validate, call, adapt.

This is the single chokepoint the Streamlit app funnels every engine call through.
It is framework-free — no Streamlit, no HTTP — so it unit-tests without a browser
or a server. It reuses the Pydantic request models in :mod:`openmed_studio.validation`
as the in-process validation layer, so the text/batch/mapping caps and value checks
the old FastAPI service enforced survive the move off HTTP. It then adapts openmed's
result objects into the plain dicts the UI helpers consume.

Errors are normalized to a single :class:`ServiceError` carrying a user-facing,
PHI-safe message (validation messages never echo the offending input); the caller
renders it. ``ValueError`` from openmed (bad options) and ``RuntimeError``/``OSError``
(model download/load failure) map to distinct messages, mirroring the old service's
400-vs-503 split without the HTTP status codes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from . import validation
from .engine import Backend, PIIEngine

logger = logging.getLogger("openmed_studio")

BACKEND_ENV = "OPENMED_STUDIO_BACKEND"


class ServiceError(Exception):
    """A user-facing failure: invalid input, bad options, or backend unavailable."""


def resolve_backend() -> Backend | None:
    """Read ``OPENMED_STUDIO_BACKEND`` -> ``'hf'``/``'mlx'``, or ``None`` when unset.

    An invalid value degrades to auto-detection (``None``) with a warning rather
    than crashing the app on a typo, matching the old service's behavior.
    """
    raw = os.environ.get(BACKEND_ENV)
    if not raw:
        return None
    value = raw.strip().lower()
    if value == "hf":
        return "hf"
    if value == "mlx":
        return "mlx"
    logger.warning(
        "%s=%r is not a valid backend ('hf' or 'mlx'); using auto-detection instead.",
        BACKEND_ENV,
        raw,
    )
    return None


def build_engine() -> PIIEngine:
    """Construct the shared engine, pinning the backend from the environment.

    The Streamlit app wraps this in ``st.cache_resource`` so the ~44M-parameter
    model loads at most once per process; this factory stays cache-free (and
    Streamlit-free) so tests can substitute a stub engine.
    """
    return PIIEngine(backend=resolve_backend())


def _validate(model: type[BaseModel], data: dict[str, Any]) -> Any:
    """Validate ``data`` against ``model``, raising a PHI-safe ``ServiceError``.

    Only each error's field location and message are surfaced — never Pydantic's
    ``input``, which would echo the offending clinical text (possible PHI).
    """
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        parts: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()) if p != "__root__")
            msg = err.get("msg", "invalid")
            parts.append(f"{loc}: {msg}" if loc else msg)
        raise ServiceError("Invalid input — " + "; ".join(parts)) from exc


def _run(call: Callable[[], Any]) -> Any:
    """Run a model call, translating failures into ``ServiceError``."""
    try:
        return call()
    except ValueError as exc:  # invalid options, e.g. date_shift_days w/o shift_dates
        raise ServiceError(str(exc)) from exc
    except (RuntimeError, OSError) as exc:  # model download/load failure on first call
        logger.exception("model backend failure")
        raise ServiceError(
            "Model backend unavailable (the model failed to load)."
        ) from exc
    except ImportError as exc:
        # An optional backend isn't installed — e.g. openmed's MissingDependencyError when
        # the Zero-shot tab is used without the `gliner` extra. The message is a safe,
        # actionable install hint (no PHI), so pass it straight through. We log the traceback
        # too (like the branches above): a *different* ImportError — an installed-but-broken
        # optional dep — would otherwise reach the UI as a bare message with no server-side
        # trail to diagnose the real import regression.
        logger.exception("optional dependency import failure")
        raise ServiceError(str(exc)) from exc
    except Exception as exc:  # any other engine/pipeline error — never surface raw
        # A raw exception would escape to Streamlit, whose default showErrorDetails
        # renders the message in the browser (possible PHI). Normalize to a generic
        # ServiceError; the detail goes to the server log, not the UI.
        logger.exception("unexpected model failure")
        raise ServiceError("The request failed unexpectedly.") from exc


def _entity_dict(entity: Any) -> dict[str, Any]:
    """Map an openmed entity (extract or deidentify shape) to a plain UI dict."""
    label = getattr(entity, "label", None) or getattr(entity, "entity_type", None) or ""
    text = getattr(entity, "text", None)
    if text is None:
        text = getattr(entity, "original_text", "")
    confidence = getattr(entity, "confidence", None)
    if confidence is None:
        # openmed.ner.Entity (GLiNER / zero-shot) names it .score, not .confidence.
        confidence = getattr(entity, "score", None)
    return {
        "label": str(label),
        "text": str(text),
        "start": int(getattr(entity, "start", 0) or 0),
        "end": int(getattr(entity, "end", 0) or 0),
        "confidence": None if confidence is None else float(confidence),
    }


def _deidentify_dict(result: Any, *, method: str, keep_mapping: bool) -> dict[str, Any]:
    """Shape an openmed ``DeidentificationResult`` into the UI's dict."""
    entities = getattr(result, "pii_entities", None) or []
    mapping = getattr(result, "mapping", None) if keep_mapping else None
    return {
        "deidentified_text": result.deidentified_text,
        "method": method,
        "entities": [_entity_dict(e) for e in entities],
        "mapping": mapping,
    }


def _deidentify_call(engine: PIIEngine, text: str, req: Any) -> Any:
    """Forward one validated request to ``engine.deidentify`` (single + batch)."""
    return engine.deidentify(
        text,
        method=req.method,
        confidence_threshold=req.confidence_threshold,
        use_smart_merging=req.use_smart_merging,
        keep_mapping=req.keep_mapping,
        consistent=req.consistent,
        seed=req.seed,
        locale=req.locale,
        lang=req.lang,
        model_name=req.model_name,
        date_shift_days=req.date_shift_days,
        keep_year=req.keep_year,
        use_safety_sweep=req.use_safety_sweep,
    )


def extract(engine: PIIEngine, text: str, **opts: Any) -> dict[str, Any]:
    """Detect PII entities; returns ``{"entities": [...]}``."""
    req = _validate(validation.ExtractRequest, {"text": text, **opts})
    entities = _run(
        lambda: engine.extract(
            req.text,
            confidence_threshold=req.confidence_threshold,
            use_smart_merging=req.use_smart_merging,
            lang=req.lang,
            model_name=req.model_name,
        )
    )
    return {"entities": [_entity_dict(e) for e in entities]}


def analyze(engine: PIIEngine, text: str, **opts: Any) -> dict[str, Any]:
    """Detect clinical entities with an NER model; returns ``{"entities": [...]}``.

    Mirrors :func:`extract` but validates against ``NerRequest`` and calls
    ``engine.analyze`` (openmed ``analyze_text``). Reuses ``_entity_dict`` unchanged —
    NER entities expose the same ``.label``/``.text``/``.start``/``.end``/``.confidence``
    (labels just come back UPPERCASE).
    """
    req = _validate(validation.NerRequest, {"text": text, **opts})
    entities = _run(
        lambda: engine.analyze(
            req.text,
            model_name=req.model_name,
            confidence_threshold=req.confidence_threshold,
            aggregation_strategy=req.aggregation_strategy,
            group_entities=req.group_entities,
        )
    )
    return {"entities": [_entity_dict(e) for e in entities]}


def extract_zero_shot(engine: PIIEngine, text: str, **opts: Any) -> dict[str, Any]:
    """Extract user-named entity labels with a GLiNER model; returns ``{"entities": [...]}``.

    Mirrors :func:`analyze` but validates against ``ZeroShotRequest`` (which requires the
    ``labels`` list) and calls ``engine.extract_zero_shot``. Reuses ``_entity_dict``, whose
    ``.score`` fallback handles openmed's zero-shot ``Entity`` (it exposes ``.score`` rather
    than ``.confidence``). A missing ``gliner`` extra surfaces via ``_run``'s ImportError
    branch as an actionable ``ServiceError``.
    """
    req = _validate(validation.ZeroShotRequest, {"text": text, **opts})
    entities = _run(
        lambda: engine.extract_zero_shot(
            req.text,
            model_name=req.model_name,
            labels=req.labels,
            confidence_threshold=req.confidence_threshold,
        )
    )
    return {"entities": [_entity_dict(e) for e in entities]}


def deidentify(engine: PIIEngine, text: str, **opts: Any) -> dict[str, Any]:
    """De-identify one note; returns ``{deidentified_text, method, entities, mapping}``."""
    req = _validate(validation.DeidentifyRequest, {"text": text, **opts})
    result = _run(lambda: _deidentify_call(engine, req.text, req))
    return _deidentify_dict(result, method=req.method, keep_mapping=req.keep_mapping)


def anonymize_policy(engine: PIIEngine, text: str, **opts: Any) -> dict[str, Any]:
    """Anonymize one note under a named regulatory policy; returns the de-identify dict shape.

    Wraps ``engine.deidentify(policy=...)`` (the policy overrides the flat method, assigning a
    per-label action from that compliance profile — so no ``method`` is sent). ``keep_mapping`` is
    not a request field: **reversibility is the policy's decision.** The seam passes
    ``keep_mapping=False`` and lets openmed OR in the profile's own flag — so the reversible
    surrogate policies (GDPR/PIPEDA/UK ICO) keep a re-identification key while the masking policies
    (HIPAA Safe Harbor, strict-no-leak) stay irreversible. Forcing ``True`` here would wrongly make
    a masking policy reversible, contradicting its posture (and the tab's "irreversible" preview).
    The dict adapter is then asked to **surface** whatever mapping the policy produced (``None`` for
    a masking one). Reuses ``_validate``/``_run``/``_deidentify_dict`` verbatim; an unknown policy is
    rejected by validation (the closed :data:`~openmed_studio.engine.Policy` Literal) before the
    engine, with a PHI-safe message. The returned ``method`` field carries the policy name.
    """
    req = _validate(validation.AnonymizePolicyRequest, {"text": text, **opts})
    result = _run(
        lambda: engine.deidentify(
            req.text,
            policy=req.policy,
            confidence_threshold=req.confidence_threshold,
            use_smart_merging=req.use_smart_merging,
            lang=req.lang,
            model_name=req.model_name,
            # The policy's own reversibility decides (openmed ORs it); don't force it on.
            keep_mapping=False,
            consistent=req.consistent,
            seed=req.seed,
            locale=req.locale,
            use_safety_sweep=req.use_safety_sweep,
        )
    )
    # Surface whatever mapping the policy produced (present for reversible policies, None for
    # masking ones) — this ``keep_mapping`` is "include the mapping in the dict", not a request.
    return _deidentify_dict(result, method=req.policy, keep_mapping=True)


def deidentify_batch(
    engine: PIIEngine, items: list[str], **opts: Any
) -> dict[str, Any]:
    """De-identify many notes in order; returns ``{"results": [...]}`` (one per item).

    Each result is tagged ``ok``: a success is ``{"ok": True, **deidentify dict}``; a note
    that trips a ``ValueError`` (bad options/content for *that* note) is isolated as
    ``{"ok": False, "error": <message>}`` so one bad note doesn't abort the whole batch.
    A backend-load failure (``RuntimeError``/``OSError``) is *not* note-specific — it would
    fail every note identically — so it propagates through ``_run`` and aborts the batch,
    surfacing one ``ServiceError`` rather than N identical failed rows.
    """
    req = _validate(validation.DeidentifyBatchRequest, {"items": items, **opts})

    def _process_all() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for text in req.items:
            try:
                result = _deidentify_call(engine, text, req)
            except ValueError as exc:  # bad options/content for THIS note — isolate it
                out.append({"ok": False, "error": str(exc)})
                continue
            out.append(
                {
                    "ok": True,
                    **_deidentify_dict(
                        result, method=req.method, keep_mapping=req.keep_mapping
                    ),
                }
            )
        return out

    return {"results": _run(_process_all)}


def reidentify(
    engine: PIIEngine, deidentified_text: str, mapping: dict[str, str]
) -> dict[str, Any]:
    """Restore originals from a kept mapping; returns ``{"text": ...}``."""
    req = _validate(
        validation.ReidentifyRequest,
        {"deidentified_text": deidentified_text, "mapping": mapping},
    )
    return {"text": _run(lambda: engine.reidentify(req.deidentified_text, req.mapping))}
