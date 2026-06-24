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
        logger.exception("de-identification backend failure")
        raise ServiceError(
            "De-identification backend unavailable (model failed to load)."
        ) from exc
    except Exception as exc:  # any other engine/pipeline error — never surface raw
        # A raw exception would escape to Streamlit, whose default showErrorDetails
        # renders the message in the browser (possible PHI). Normalize to a generic
        # ServiceError; the detail goes to the server log, not the UI.
        logger.exception("unexpected de-identification failure")
        raise ServiceError("De-identification failed unexpectedly.") from exc


def _entity_dict(entity: Any) -> dict[str, Any]:
    """Map an openmed entity (extract or deidentify shape) to a plain UI dict."""
    label = getattr(entity, "label", None) or getattr(entity, "entity_type", None) or ""
    text = getattr(entity, "text", None)
    if text is None:
        text = getattr(entity, "original_text", "")
    confidence = getattr(entity, "confidence", None)
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


def deidentify(engine: PIIEngine, text: str, **opts: Any) -> dict[str, Any]:
    """De-identify one note; returns ``{deidentified_text, method, entities, mapping}``."""
    req = _validate(validation.DeidentifyRequest, {"text": text, **opts})
    result = _run(lambda: _deidentify_call(engine, req.text, req))
    return _deidentify_dict(result, method=req.method, keep_mapping=req.keep_mapping)


def deidentify_batch(
    engine: PIIEngine, items: list[str], **opts: Any
) -> dict[str, Any]:
    """De-identify many notes in order; returns ``{"results": [...]}`` (one per item)."""
    req = _validate(validation.DeidentifyBatchRequest, {"items": items, **opts})
    results = _run(
        lambda: [
            _deidentify_dict(
                _deidentify_call(engine, text, req),
                method=req.method,
                keep_mapping=req.keep_mapping,
            )
            for text in req.items
        ]
    )
    return {"results": results}


def reidentify(
    engine: PIIEngine, deidentified_text: str, mapping: dict[str, str]
) -> dict[str, Any]:
    """Restore originals from a kept mapping; returns ``{"text": ...}``."""
    req = _validate(
        validation.ReidentifyRequest,
        {"deidentified_text": deidentified_text, "mapping": mapping},
    )
    return {"text": _run(lambda: engine.reidentify(req.deidentified_text, req.mapping))}
