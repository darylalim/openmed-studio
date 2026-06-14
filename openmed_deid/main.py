"""FastAPI service exposing OpenMed PII/PHI de-identification over HTTP.

Launch with::

    uv run uvicorn openmed_deid.main:app --port 8080
    # or: uv run python -m openmed_deid

Then open http://127.0.0.1:8080/docs for interactive API docs.

Authentication: if ``OPENMED_DEID_API_KEY`` is set, every ``/pii/*`` request must
send a matching ``X-API-Key`` header (otherwise 401). If it is unset the API runs
UNAUTHENTICATED for local use and logs a startup warning — set the key (and use
TLS) before exposing the service on a network or processing real PHI.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader

from . import schemas
from .engine import DEFAULT_PII_MODEL, PIIEngine

logger = logging.getLogger("openmed_deid")

API_KEY_ENV = "OPENMED_DEID_API_KEY"
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_engine: PIIEngine | None = None


def get_engine() -> PIIEngine:
    """Return the process-wide engine (overridden in tests via dependency_overrides)."""
    global _engine
    if _engine is None:
        _engine = PIIEngine()
    return _engine


def require_api_key(provided: str | None = Depends(_api_key_header)) -> None:
    """Enforce ``X-API-Key`` when ``OPENMED_DEID_API_KEY`` is set; a no-op otherwise."""
    expected = os.environ.get(API_KEY_ENV)
    if not expected:
        return  # auth disabled for local use; create_app() warns at startup
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _to_entity(entity: Any) -> schemas.Entity:
    """Map an openmed entity (extract_pii or deidentify shape) to the API model."""
    label = getattr(entity, "label", None) or getattr(entity, "entity_type", None) or ""
    text = getattr(entity, "text", None)
    if text is None:
        text = getattr(entity, "original_text", "")
    return schemas.Entity(
        label=str(label),
        text=str(text),
        start=int(getattr(entity, "start", 0) or 0),
        end=int(getattr(entity, "end", 0) or 0),
        confidence=getattr(entity, "confidence", None),
    )


def _run(call: Callable[[], Any]) -> Any:
    """Run a model call, translating backend failures into clean HTTP errors."""
    try:
        return call()
    except (
        ValueError
    ) as exc:  # invalid options, e.g. date_shift_days without shift_dates
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, OSError) as exc:  # model download/load failure on first call
        logger.exception("de-identification backend failure")
        raise HTTPException(
            status_code=503,
            detail="De-identification backend unavailable (model failed to load).",
        ) from exc


def _deidentify_one(
    engine: PIIEngine, text: str, opts: schemas._DeidentifyOptions
) -> schemas.DeidentifyResponse:
    result = engine.deidentify(
        text,
        method=opts.method,
        confidence_threshold=opts.confidence_threshold,
        keep_mapping=opts.keep_mapping,
        consistent=opts.consistent,
        seed=opts.seed,
        lang=opts.lang,
        model_name=opts.model_name,
        date_shift_days=opts.date_shift_days,
        keep_year=opts.keep_year,
    )
    entities = getattr(result, "pii_entities", None) or []
    mapping = getattr(result, "mapping", None) if opts.keep_mapping else None
    return schemas.DeidentifyResponse(
        deidentified_text=result.deidentified_text,
        method=opts.method,
        entities=[_to_entity(e) for e in entities],
        mapping=mapping,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="openmed-deid",
        description="PII / PHI de-identification for clinical text, built on OpenMed.",
        version="0.1.0",
    )

    if not os.environ.get(API_KEY_ENV):
        logger.warning(
            "%s is not set: the de-identification API is running WITHOUT authentication. "
            "Set %s (and use TLS) before exposing it on a network or processing real PHI.",
            API_KEY_ENV,
            API_KEY_ENV,
        )

    protected = [Depends(require_api_key)]

    @app.get("/health", response_model=schemas.HealthResponse)
    def health(engine: PIIEngine = Depends(get_engine)) -> schemas.HealthResponse:
        return schemas.HealthResponse(
            status="ok",
            service="openmed-deid",
            model=engine.model_name or DEFAULT_PII_MODEL,
            model_loaded=engine.is_loaded,
            auth_required=bool(os.environ.get(API_KEY_ENV)),
        )

    @app.post(
        "/pii/extract", response_model=schemas.ExtractResponse, dependencies=protected
    )
    def extract(
        req: schemas.ExtractRequest, engine: PIIEngine = Depends(get_engine)
    ) -> schemas.ExtractResponse:
        entities = _run(
            lambda: engine.extract(
                req.text,
                confidence_threshold=req.confidence_threshold,
                use_smart_merging=req.use_smart_merging,
                lang=req.lang,
                model_name=req.model_name,
            )
        )
        return schemas.ExtractResponse(entities=[_to_entity(e) for e in entities])

    @app.post(
        "/pii/deidentify",
        response_model=schemas.DeidentifyResponse,
        dependencies=protected,
    )
    def deidentify(
        req: schemas.DeidentifyRequest, engine: PIIEngine = Depends(get_engine)
    ) -> schemas.DeidentifyResponse:
        return _run(lambda: _deidentify_one(engine, req.text, req))

    @app.post(
        "/pii/deidentify/batch",
        response_model=schemas.DeidentifyBatchResponse,
        dependencies=protected,
    )
    def deidentify_batch(
        req: schemas.DeidentifyBatchRequest, engine: PIIEngine = Depends(get_engine)
    ) -> schemas.DeidentifyBatchResponse:
        results = _run(
            lambda: [_deidentify_one(engine, text, req) for text in req.items]
        )
        return schemas.DeidentifyBatchResponse(results=results)

    @app.post(
        "/pii/reidentify",
        response_model=schemas.ReidentifyResponse,
        dependencies=protected,
    )
    def reidentify(
        req: schemas.ReidentifyRequest, engine: PIIEngine = Depends(get_engine)
    ) -> schemas.ReidentifyResponse:
        return schemas.ReidentifyResponse(
            text=engine.reidentify(req.deidentified_text, req.mapping)
        )

    return app


app = create_app()
