"""FastAPI service exposing OpenMed clinical NLP over HTTP.

Launch with::

    uv run uvicorn openmed_studio.main:app --port 8080
    # or: uv run python -m openmed_studio

Then open http://127.0.0.1:8080/docs for interactive API docs.

This is a **second surface over the same in-process seam** the Streamlit app uses: every
route is a thin wrapper over :mod:`openmed_studio.service`, so validation, the PHI-safe error
handling, and the object->dict adapters are shared, not reimplemented. The service in turn
drives one shared :class:`~openmed_studio.engine.PIIEngine` (the model loads once per process
and is reused; an engine-internal lock serializes concurrent inference).

Authentication: if ``OPENMED_STUDIO_API_KEY`` is set, every model route must send a matching
``X-API-Key`` header (otherwise 401). If it is unset the API runs UNAUTHENTICATED for local use
and logs a startup warning — set the key (and use TLS, or put it behind your own proxy) before
exposing the service on a network or processing real PHI. ``GET /health`` is always open.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, service, validation
from .engine import DEFAULT_PII_MODEL, DeidMethod, PIIEngine
from .service import ServiceError, ServiceErrorKind
from .validation import (
    AnonymizePolicyRequest,
    DeidentifyBatchRequest,
    DeidentifyRequest,
    ExtractRequest,
    ModelName,
    NerRequest,
    ReidentifyRequest,
    ZeroShotRequest,
)

logger = logging.getLogger("openmed_studio")

API_KEY_ENV = "OPENMED_STUDIO_API_KEY"
PRELOAD_ENV = "OPENMED_STUDIO_PRELOAD"
COMPAT_ENV = "OPENMED_STUDIO_COMPAT"
_TRUTHY = {"1", "true", "yes", "on"}
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _truthy_env(name: str) -> bool:
    """True when env var ``name`` is set to a truthy value (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


# --- Engine (process-wide, shared with the service seam) ---------------------
@lru_cache(maxsize=1)
def get_engine() -> PIIEngine:
    """Return the process-wide engine (overridden in tests via ``dependency_overrides``).

    Built through :func:`openmed_studio.service.build_engine`, so backend resolution
    (``OPENMED_STUDIO_BACKEND``) is the same one the Streamlit app uses.

    ``lru_cache`` memoizes the single instance **thread-safely**: the model routes are sync
    ``def``s that FastAPI runs in a threadpool, so a plain check-then-set singleton could race
    two cold-start requests into two ``PIIEngine`` instances — each with its own ``ModelLoader``
    and its own inference lock, which would silently break the "one engine, one serializing lock
    per process" invariant (and load the model twice). FastAPI's ``dependency_overrides`` replaces
    this callable wholesale, so the cache is transparent to the test stubs.
    """
    return service.build_engine()


# --- Uniform error envelope --------------------------------------------------
# Every non-2xx response is shaped {"error": {"code", "message", "details"}} so a client
# can rely on one error contract (and the /compat surface matches OpenMed's own REST shape).
# A ServiceError's transport-neutral `kind` maps to the HTTP status here — the seam never
# names a status code itself.
_KIND_STATUS: dict[ServiceErrorKind, int] = {
    "validation": 422,
    "bad_options": 400,
    "unavailable": 503,
    "dependency": 503,
    "internal": 500,
}

_STATUS_ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    422: "validation_error",
    500: "internal_error",
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
    # Build the body from the same ErrorResponse/ErrorDetail models declared for the OpenAPI
    # docs (below), so the documented schema and the runtime body can't drift apart.
    body = ErrorResponse(
        error=ErrorDetail(
            code=_error_code(status_code), message=message, details=details
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


# --- Response models (typed contract + OpenAPI schemas) ----------------------
class _Strict(BaseModel):
    """Reject unknown fields so a response never silently gains an undocumented key."""

    model_config = ConfigDict(extra="forbid")


class Entity(_Strict):
    label: str
    text: str
    start: int
    end: int
    confidence: float | None = None


class EntitiesResponse(_Strict):
    """Shared by /pii/extract, /ner, and /zero-shot — all return matched entities."""

    entities: list[Entity]


class DeidentifyResponse(_Strict):
    deidentified_text: str
    # A DeidMethod for /pii/deidentify[/batch]; a policy name for /pii/anonymize-policy.
    method: str
    entities: list[Entity]
    mapping: dict[str, str] | None = None


class BatchItemResult(_Strict):
    """One note's result. ``ok`` success carries the de-identify fields; a per-note
    failure (a bad option/content for *that* note) carries ``error`` instead — mirroring
    the seam's per-note isolation, so one bad note doesn't fail the whole batch."""

    ok: bool
    deidentified_text: str | None = None
    method: str | None = None
    entities: list[Entity] | None = None
    mapping: dict[str, str] | None = None
    error: str | None = None


class DeidentifyBatchResponse(_Strict):
    results: list[BatchItemResult]


class ReidentifyResponse(_Strict):
    text: str


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
        description="True once the engine has initialized its ModelLoader (after the first "
        "model request); not a guarantee the model is resident."
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


# OpenAPI: document the envelope for the error statuses the model routes emit.
_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "Invalid options for the request."},
    401: {"model": ErrorResponse, "description": "Missing or invalid API key."},
    422: {"model": ErrorResponse, "description": "Request failed schema validation."},
    500: {"model": ErrorResponse, "description": "Unexpected server error."},
    503: {"model": ErrorResponse, "description": "Inference backend unavailable."},
}


# --- Auth --------------------------------------------------------------------
def require_api_key(provided: str | None = Depends(_api_key_header)) -> None:
    """Enforce ``X-API-Key`` when ``OPENMED_STUDIO_API_KEY`` is set; a no-op otherwise."""
    expected = os.environ.get(API_KEY_ENV)
    if not expected:
        return  # auth disabled for local use; create_app() warns at startup
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# --- Lifespan (optional model warm-up) ---------------------------------------
@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Optionally warm the model at startup so the first request isn't slow.

    Enabled by a truthy ``OPENMED_STUDIO_PRELOAD``. The warm-up runs in a worker thread
    (so it never blocks the event loop) and a failure degrades to lazy loading on the
    first request rather than crashing startup.
    """
    if _truthy_env(PRELOAD_ENV):
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


# --- OpenMed-REST compatibility surface (opt-in, off by default) -------------
# These deliberately do NOT use `_Strict`: Pydantic's default `extra="ignore"` lets a
# request carry upstream-only fields (notably `keep_alive`) without a 422, so an
# OpenMed-REST client can post unchanged. `lang` is a plain str (not the Lang Literal)
# for the same parity reason; an unsupported value still fails in the seam's engine call.


class CompatExtractRequest(BaseModel):
    """OpenMed-REST-shaped ``/pii/extract`` body; unknown fields are ignored."""

    text: validation.ClinicalText
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

    text: validation.ClinicalText
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


def _compat_entity(entity: Any, *, redacted: bool = False) -> dict[str, Any]:
    """openmed-shaped entity dict: the seam's `label`/`text`/`start`/`end`/`confidence` fields
    plus `entity_type`, plus the extra field the upstream stage emits — ``redacted_text`` for
    deidentify, ``metadata`` for extract. Reuses ``service._entity_dict`` for the base fields
    (its `.score` fallback is inert here — compat only wraps `extract`/`deidentify` results,
    which carry `.confidence`).
    """
    out: dict[str, Any] = service._entity_dict(entity)
    out["entity_type"] = str(getattr(entity, "entity_type", None) or out["label"])
    if redacted:
        out["redacted_text"] = getattr(entity, "redacted_text", None)
    else:
        out["metadata"] = getattr(entity, "metadata", None) or {}
    return out


def _build_compat_router() -> APIRouter:
    """OpenMed-REST-compatible ``/pii/*`` surface under ``/compat`` (opt-in).

    Mirrors OpenMed's own REST service so a client can point its base URL at ``<host>/compat``
    unchanged: requests accept (and ignore) ``keep_alive``, and responses use openmed's shape —
    ``pii_entities``, ``num_entities_redacted``, ``timestamp``, and the echoed ``original_text``.
    Echoing ``original_text`` returns the request input (possibly PHI) in the response, matching
    upstream; that is why this surface is off by default (enable with OPENMED_STUDIO_COMPAT).

    It calls the engine directly for the raw entity objects upstream's shape needs (the seam's
    dict adapter drops ``redacted_text``/``metadata``), but reuses the seam's error translation
    (:func:`service._run`) so its failures map to the same envelope as the primary routes. The
    request caps/format guards are still enforced by the Compat* request models above.
    """
    router = APIRouter(
        prefix="/compat",
        tags=["openmed-rest-compat"],
        dependencies=[Depends(require_api_key)],
    )

    @router.post("/pii/extract")
    def compat_extract(
        req: CompatExtractRequest, engine: PIIEngine = Depends(get_engine)
    ) -> dict[str, Any]:
        entities = service._run(
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
        req: CompatDeidentifyRequest, engine: PIIEngine = Depends(get_engine)
    ) -> dict[str, Any]:
        result = service._run(
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


# --- App factory -------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="openmed-studio",
        description=(
            "Clinical NLP for clinical text, built on OpenMed: PII/PHI de-identification "
            "(mask/replace/surrogate + regulatory policies), clinical NER, and zero-shot "
            "extraction."
        ),
        version=__version__,
        lifespan=_lifespan,
    )

    if not os.environ.get(API_KEY_ENV):
        logger.warning(
            "%s is not set: the openmed-studio API is running WITHOUT authentication. "
            "Set %s (and use TLS) before exposing it on a network or processing real PHI.",
            API_KEY_ENV,
            API_KEY_ENV,
        )

    @app.exception_handler(ServiceError)
    async def _on_service_error(_request: Request, exc: ServiceError) -> JSONResponse:
        status_code = _KIND_STATUS.get(exc.kind, 500)
        return _error_response(status_code, str(exc))

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
        # Keep type/loc/msg but drop `input`, so request content (possible PHI) is never
        # echoed back in the error body.
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

    @app.get("/health", response_model=HealthResponse)
    def health(engine: PIIEngine = Depends(get_engine)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="openmed-studio",
            version=__version__,
            model=engine.model_name or DEFAULT_PII_MODEL,
            backend=engine.backend or "auto",
            max_text_chars=validation.MAX_TEXT_CHARS,
            model_loaded=engine.is_loaded,
            auth_required=bool(os.environ.get(API_KEY_ENV)),
        )

    @app.post(
        "/pii/extract",
        response_model=EntitiesResponse,
        dependencies=protected,
        responses=_ERROR_RESPONSES,
        tags=["pii"],
    )
    def extract(
        req: ExtractRequest, engine: PIIEngine = Depends(get_engine)
    ) -> dict[str, Any]:
        return service.extract(engine, **req.model_dump())

    @app.post(
        "/ner",
        response_model=EntitiesResponse,
        dependencies=protected,
        responses=_ERROR_RESPONSES,
        tags=["ner"],
    )
    def ner(req: NerRequest, engine: PIIEngine = Depends(get_engine)) -> dict[str, Any]:
        return service.analyze(engine, **req.model_dump())

    @app.post(
        "/zero-shot",
        response_model=EntitiesResponse,
        dependencies=protected,
        responses=_ERROR_RESPONSES,
        tags=["ner"],
    )
    def zero_shot(
        req: ZeroShotRequest, engine: PIIEngine = Depends(get_engine)
    ) -> dict[str, Any]:
        return service.extract_zero_shot(engine, **req.model_dump())

    @app.post(
        "/pii/deidentify",
        response_model=DeidentifyResponse,
        dependencies=protected,
        responses=_ERROR_RESPONSES,
        tags=["pii"],
    )
    def deidentify(
        req: DeidentifyRequest, engine: PIIEngine = Depends(get_engine)
    ) -> dict[str, Any]:
        return service.deidentify(engine, **req.model_dump())

    @app.post(
        "/pii/deidentify/batch",
        response_model=DeidentifyBatchResponse,
        dependencies=protected,
        responses=_ERROR_RESPONSES,
        tags=["pii"],
    )
    def deidentify_batch(
        req: DeidentifyBatchRequest, engine: PIIEngine = Depends(get_engine)
    ) -> dict[str, Any]:
        data = req.model_dump()
        items = data.pop("items")
        return service.deidentify_batch(engine, items, **data)

    @app.post(
        "/pii/anonymize-policy",
        response_model=DeidentifyResponse,
        dependencies=protected,
        responses=_ERROR_RESPONSES,
        tags=["pii"],
    )
    def anonymize_policy(
        req: AnonymizePolicyRequest, engine: PIIEngine = Depends(get_engine)
    ) -> dict[str, Any]:
        return service.anonymize_policy(engine, **req.model_dump())

    @app.post(
        "/pii/reidentify",
        response_model=ReidentifyResponse,
        dependencies=protected,
        responses=_ERROR_RESPONSES,
        tags=["pii"],
    )
    def reidentify(
        req: ReidentifyRequest, engine: PIIEngine = Depends(get_engine)
    ) -> dict[str, Any]:
        return service.reidentify(engine, req.deidentified_text, req.mapping)

    if _truthy_env(COMPAT_ENV):
        app.include_router(_build_compat_router())

    return app


app = create_app()
