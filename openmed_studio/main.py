"""FastAPI service exposing OpenMed PII/PHI de-identification over HTTP.

Launch with::

    uv run uvicorn openmed_studio.main:app --port 8080
    # or: uv run python -m openmed_studio

Then open http://127.0.0.1:8080/docs for interactive API docs.

Authentication: if ``OPENMED_STUDIO_API_KEY`` is set, every ``/pii/*`` request must
send a matching ``X-API-Key`` header (otherwise 401). If it is unset the API runs
UNAUTHENTICATED for local use and logs a startup warning — set the key (and use
TLS) before exposing the service on a network or processing real PHI.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import schemas
from .engine import DEFAULT_PII_MODEL, Backend, PIIEngine

logger = logging.getLogger("openmed_studio")

APP_VERSION = "0.1.0"
API_KEY_ENV = "OPENMED_STUDIO_API_KEY"
BACKEND_ENV = "OPENMED_STUDIO_BACKEND"
PRELOAD_ENV = "OPENMED_STUDIO_PRELOAD"
COMPAT_ENV = "OPENMED_STUDIO_COMPAT"
_TRUTHY = {"1", "true", "yes", "on"}
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _truthy_env(name: str) -> bool:
    """True when env var ``name`` is set to a truthy value (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


_engine: PIIEngine | None = None

# --- Uniform error envelope -------------------------------------------------
# Every non-2xx response is shaped {"error": {"code", "message", "details"}} to
# match OpenMed's own REST service, so a client can target either interchangeably.
_STATUS_ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    422: "validation_error",
    503: "service_unavailable",
}


def _error_code(status_code: int) -> str:
    code = _STATUS_ERROR_CODES.get(status_code)
    if code is not None:
        return code
    return "internal_error" if status_code >= 500 else "http_error"


def _error_response(
    status_code: int, message: str, *, details: Any = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": _error_code(status_code),
                "message": message,
                "details": details,
            }
        },
    )


# OpenAPI: document the envelope for the error statuses the /pii routes emit.
_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "model": schemas.ErrorResponse,
        "description": "Invalid de-identification options.",
    },
    401: {
        "model": schemas.ErrorResponse,
        "description": "Missing or invalid API key.",
    },
    422: {
        "model": schemas.ErrorResponse,
        "description": "Request failed schema validation.",
    },
    503: {
        "model": schemas.ErrorResponse,
        "description": "Inference backend unavailable.",
    },
}


def _resolve_backend() -> Backend | None:
    """Read OPENMED_STUDIO_BACKEND -> 'hf'/'mlx', or None (auto-detect) when unset.

    An invalid value degrades to auto-detection with a warning rather than
    crashing the service on a typo.
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


def get_engine() -> PIIEngine:
    """Return the process-wide engine (overridden in tests via dependency_overrides)."""
    global _engine
    if _engine is None:
        _engine = PIIEngine(backend=_resolve_backend())
    return _engine


def require_api_key(provided: str | None = Depends(_api_key_header)) -> None:
    """Enforce ``X-API-Key`` when ``OPENMED_STUDIO_API_KEY`` is set; a no-op otherwise."""
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


def _preload_enabled() -> bool:
    """True when OPENMED_STUDIO_PRELOAD is set to a truthy value (1/true/yes/on)."""
    return _truthy_env(PRELOAD_ENV)


def _compat_enabled() -> bool:
    """True when OPENMED_STUDIO_COMPAT is set, mounting the /compat OpenMed-REST surface."""
    return _truthy_env(COMPAT_ENV)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Optionally warm the model at startup so the first request isn't slow.

    Enabled by a truthy ``OPENMED_STUDIO_PRELOAD``. The warm-up runs in a worker
    thread (so it never blocks the event loop) and a failure degrades to lazy
    loading on the first request rather than crashing startup.
    """
    if _preload_enabled():
        engine = get_engine()
        model = engine.model_name or DEFAULT_PII_MODEL
        try:
            await run_in_threadpool(engine.extract, "warm-up")
            logger.info("Preloaded PII model %s at startup.", model)
        except Exception:
            logger.warning(
                "Preload of %s failed; falling back to lazy load on first request.",
                model,
                exc_info=True,
            )
    yield


def _compat_entity(entity: Any, *, redacted: bool = False) -> dict[str, Any]:
    """openmed-shaped entity dict: carries label *and* entity_type, plus the extra
    field the upstream stage emits — ``redacted_text`` for deidentify, ``metadata``
    for extract. Confidence is coerced to a built-in float for safe JSON encoding.
    """
    label = getattr(entity, "label", "") or getattr(entity, "entity_type", "") or ""
    entity_type = getattr(entity, "entity_type", "") or label
    text = getattr(entity, "text", None)
    if text is None:
        text = getattr(entity, "original_text", "")
    confidence = getattr(entity, "confidence", None)
    out: dict[str, Any] = {
        "text": str(text),
        "label": str(label),
        "entity_type": str(entity_type),
        "start": int(getattr(entity, "start", 0) or 0),
        "end": int(getattr(entity, "end", 0) or 0),
        "confidence": None if confidence is None else float(confidence),
    }
    if redacted:
        out["redacted_text"] = getattr(entity, "redacted_text", None)
    else:
        out["metadata"] = getattr(entity, "metadata", None) or {}
    return out


def _build_compat_router() -> APIRouter:
    """OpenMed-REST-compatible ``/pii/*`` surface under ``/compat`` (opt-in).

    Mirrors OpenMed's own REST service so a client can point its base URL at
    ``<host>/compat`` unchanged: requests accept (and ignore) ``keep_alive``, and
    responses use openmed's shape — ``pii_entities``, ``num_entities_redacted``,
    ``timestamp``, and the echoed ``original_text``. Echoing ``original_text``
    returns the request input (possibly PHI) in the response, matching upstream;
    that is why this surface is off by default (enable with OPENMED_STUDIO_COMPAT).
    """
    router = APIRouter(
        prefix="/compat",
        tags=["openmed-rest-compat"],
        dependencies=[Depends(require_api_key)],
    )

    @router.post("/pii/extract")
    def compat_extract(
        req: schemas.CompatExtractRequest, engine: PIIEngine = Depends(get_engine)
    ) -> dict[str, Any]:
        entities = _run(
            lambda: engine.extract(
                req.text,
                confidence_threshold=req.confidence_threshold,
                use_smart_merging=req.use_smart_merging,
                lang=req.lang,
                model_name=req.model_name,
            )
        )
        return {"entities": [_compat_entity(e) for e in entities]}

    @router.post("/pii/deidentify")
    def compat_deidentify(
        req: schemas.CompatDeidentifyRequest, engine: PIIEngine = Depends(get_engine)
    ) -> dict[str, Any]:
        result = _run(
            lambda: engine.deidentify(
                req.text,
                method=req.method,
                confidence_threshold=req.confidence_threshold,
                keep_mapping=req.keep_mapping,
                consistent=req.consistent,
                seed=req.seed,
                lang=req.lang,
                model_name=req.model_name,
                date_shift_days=req.date_shift_days,
                keep_year=req.keep_year,
            )
        )
        entities = [
            _compat_entity(e, redacted=True)
            for e in (getattr(result, "pii_entities", None) or [])
        ]
        payload: dict[str, Any] = {
            "original_text": req.text,
            "deidentified_text": result.deidentified_text,
            "pii_entities": entities,
            "method": req.method,
            "num_entities_redacted": len(entities),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if req.keep_mapping:
            payload["mapping"] = getattr(result, "mapping", None)
        return payload

    return router


def create_app() -> FastAPI:
    app = FastAPI(
        title="openmed-studio",
        description="PII / PHI de-identification for clinical text, built on OpenMed.",
        version=APP_VERSION,
        lifespan=_lifespan,
    )

    if not os.environ.get(API_KEY_ENV):
        logger.warning(
            "%s is not set: the openmed-studio API is running WITHOUT authentication. "
            "Set %s (and use TLS) before exposing it on a network or processing real PHI.",
            API_KEY_ENV,
            API_KEY_ENV,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _on_http_exception(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        response = _error_response(exc.status_code, str(exc.detail))
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Keep type/loc/msg but drop `input`, so request content (possible PHI) is
        # never echoed back in the error body.
        details = jsonable_encoder(
            [
                {k: err[k] for k in ("type", "loc", "msg") if k in err}
                for err in exc.errors()
            ]
        )
        return _error_response(
            422, "Request failed schema validation.", details=details
        )

    protected = [Depends(require_api_key)]

    @app.get("/health", response_model=schemas.HealthResponse)
    def health(engine: PIIEngine = Depends(get_engine)) -> schemas.HealthResponse:
        return schemas.HealthResponse(
            status="ok",
            service="openmed-studio",
            version=APP_VERSION,
            model=engine.model_name or DEFAULT_PII_MODEL,
            backend=engine.backend or "auto",
            max_text_chars=schemas.MAX_TEXT_CHARS,
            model_loaded=engine.is_loaded,
            auth_required=bool(os.environ.get(API_KEY_ENV)),
        )

    @app.post(
        "/pii/extract",
        response_model=schemas.ExtractResponse,
        dependencies=protected,
        responses=_ERROR_RESPONSES,
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
        responses=_ERROR_RESPONSES,
    )
    def deidentify(
        req: schemas.DeidentifyRequest, engine: PIIEngine = Depends(get_engine)
    ) -> schemas.DeidentifyResponse:
        return _run(lambda: _deidentify_one(engine, req.text, req))

    @app.post(
        "/pii/deidentify/batch",
        response_model=schemas.DeidentifyBatchResponse,
        dependencies=protected,
        responses=_ERROR_RESPONSES,
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
        responses=_ERROR_RESPONSES,
    )
    def reidentify(
        req: schemas.ReidentifyRequest, engine: PIIEngine = Depends(get_engine)
    ) -> schemas.ReidentifyResponse:
        return schemas.ReidentifyResponse(
            text=engine.reidentify(req.deidentified_text, req.mapping)
        )

    if _compat_enabled():
        app.include_router(_build_compat_router())

    return app


app = create_app()
