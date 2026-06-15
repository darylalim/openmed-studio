"""Tests for the FastAPI de-identification service.

The fast tests inject a model-free stub engine via FastAPI dependency overrides,
so they validate request/response wiring and schema validation without
``--run-model``. The tests marked ``@pytest.mark.model`` exercise the real
OpenMed engine (reusing the session-scoped ``loader`` fixture) and are skipped
unless ``--run-model`` is passed.
"""

from __future__ import annotations

import contextlib
import logging
import typing
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from openmed_studio.engine import PIIEngine
from openmed_studio.main import (
    API_KEY_ENV,
    APP_VERSION,
    BACKEND_ENV,
    _resolve_backend,
    app,
    create_app,
    get_engine,
)


class _StubEngine:
    """A model-free stand-in for PIIEngine with the same call surface."""

    model_name = "stub-model"
    backend: str | None = None
    is_loaded = True

    def extract(self, text, **_):
        return [
            SimpleNamespace(
                label="first_name", text="John", start=0, end=4, confidence=0.99
            )
        ]

    def deidentify(self, text, *, keep_mapping=False, **_):
        mapping = {"[first_name]": "John"} if keep_mapping else None
        return SimpleNamespace(
            deidentified_text="[first_name] A. Doe",
            pii_entities=[
                SimpleNamespace(
                    label="first_name", text="John", start=0, end=4, confidence=0.99
                )
            ],
            mapping=mapping,
        )

    def reidentify(self, deidentified_text, mapping):
        for key, value in mapping.items():
            deidentified_text = deidentified_text.replace(key, value)
        return deidentified_text


class _RaisingEngine(_StubEngine):
    """Stub whose model calls raise, to exercise the API error paths."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def extract(self, text, **_):
        raise self._exc

    def deidentify(self, text, **_):
        raise self._exc


@pytest.fixture
def client():
    app.dependency_overrides[get_engine] = lambda: _StubEngine()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@contextlib.contextmanager
def _override_engine(engine):
    app.dependency_overrides[get_engine] = lambda: engine
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_health_ok(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "openmed-studio"
    assert body["version"] == APP_VERSION
    assert body["model"] == "stub-model"
    assert body["backend"] == "auto"  # stub engine has backend=None -> "auto"
    assert body["max_text_chars"] == 50_000  # OPENMED_STUDIO_MAX_TEXT_LENGTH default
    assert body["model_loaded"] is True
    assert body["auth_required"] is False  # no OPENMED_STUDIO_API_KEY set in tests


def test_health_reports_configured_backend() -> None:
    engine = SimpleNamespace(model_name=None, backend="mlx", is_loaded=False)
    with _override_engine(engine) as override:
        assert override.get("/health").json()["backend"] == "mlx"


def test_resolve_backend_unset_is_none(monkeypatch) -> None:
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    assert _resolve_backend() is None


def test_resolve_backend_empty_is_none(monkeypatch) -> None:
    # A set-but-empty value is treated like unset (auto-detect), not an error.
    monkeypatch.setenv(BACKEND_ENV, "")
    assert _resolve_backend() is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("mlx", "mlx"), ("hf", "hf"), ("  MLX ", "mlx"), ("HF", "hf")],
)
def test_resolve_backend_normalizes_valid_values(monkeypatch, value, expected) -> None:
    monkeypatch.setenv(BACKEND_ENV, value)
    assert _resolve_backend() == expected


def test_resolve_backend_invalid_falls_back_to_auto(monkeypatch, caplog) -> None:
    # A typo must degrade to auto-detect (None) AND warn, naming the bad value, so the
    # silent fallback is observable rather than crashing the service.
    monkeypatch.setenv(BACKEND_ENV, "cuda")
    with caplog.at_level(logging.WARNING, logger="openmed_studio"):
        assert _resolve_backend() is None
    assert "cuda" in caplog.text


def test_get_engine_wires_resolved_backend(monkeypatch) -> None:
    # get_engine() must build a real PIIEngine carrying the resolved backend while
    # staying lazy (no model load). Reset the cached module singleton first so a
    # fresh engine is constructed; monkeypatch restores it after the test.
    import openmed_studio.main as main

    monkeypatch.setattr(main, "_engine", None)
    monkeypatch.setenv(BACKEND_ENV, "mlx")
    engine = get_engine()
    assert isinstance(engine, PIIEngine)
    assert engine.backend == "mlx"
    assert engine.is_loaded is False  # constructed but no model loaded


# --- Startup model preload (OPENMED_STUDIO_PRELOAD) --------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("", False),
        ("nope", False),
    ],
)
def test_preload_enabled_parses_truthy(monkeypatch, value, expected) -> None:
    import openmed_studio.main as main

    monkeypatch.setenv(main.PRELOAD_ENV, value)
    assert main._preload_enabled() is expected


class _RecordingEngine(_StubEngine):
    """Stub that records the warm-up extract() the startup lifespan makes."""

    def __init__(self) -> None:
        self.warmed: list[str] = []

    def extract(self, text, **_):
        self.warmed.append(text)
        return []


def test_preload_warms_model_when_enabled(monkeypatch) -> None:
    import openmed_studio.main as main

    engine = _RecordingEngine()
    monkeypatch.setattr(main, "_engine", engine)  # get_engine() returns this stub
    monkeypatch.setenv(main.PRELOAD_ENV, "1")
    with TestClient(main.app):  # entering runs the lifespan startup
        pass
    assert engine.warmed == ["warm-up"]


def test_preload_skipped_by_default(monkeypatch) -> None:
    import openmed_studio.main as main

    engine = _RecordingEngine()
    monkeypatch.setattr(main, "_engine", engine)
    monkeypatch.delenv(main.PRELOAD_ENV, raising=False)
    with TestClient(main.app):
        pass
    assert engine.warmed == []  # no warm-up without the flag


def test_preload_failure_degrades_gracefully(monkeypatch, caplog) -> None:
    import openmed_studio.main as main

    class _BoomEngine(_StubEngine):
        model_name = "boom-model"

        def extract(self, text, **_):
            raise RuntimeError("model download failed")

    monkeypatch.setattr(main, "_engine", _BoomEngine())
    monkeypatch.setenv(main.PRELOAD_ENV, "1")
    with caplog.at_level(logging.WARNING, logger="openmed_studio"):
        with TestClient(main.app):  # must not raise despite the failed warm-up
            pass
    assert "falling back to lazy load" in caplog.text


def test_extract_returns_entities(client) -> None:
    resp = client.post("/pii/extract", json={"text": "Patient John."})
    assert resp.status_code == 200
    assert {
        "label": "first_name",
        "text": "John",
        "start": 0,
        "end": 4,
        "confidence": 0.99,
    } in resp.json()["entities"]


def test_deidentify_omits_mapping_by_default(client) -> None:
    resp = client.post(
        "/pii/deidentify", json={"text": "Patient John.", "method": "mask"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["deidentified_text"] == "[first_name] A. Doe"
    assert body["mapping"] is None


def test_deidentify_includes_mapping_when_requested(client) -> None:
    resp = client.post(
        "/pii/deidentify",
        json={"text": "Patient John.", "method": "replace", "keep_mapping": True},
    )
    assert resp.status_code == 200
    assert resp.json()["mapping"] == {"[first_name]": "John"}


def test_reidentify_restores(client) -> None:
    resp = client.post(
        "/pii/reidentify",
        json={
            "deidentified_text": "Hi [first_name].",
            "mapping": {"[first_name]": "John"},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "Hi John."


def test_rejects_unknown_field(client) -> None:
    assert (
        client.post("/pii/extract", json={"text": "x", "bogus": 1}).status_code == 422
    )


def test_rejects_empty_text(client) -> None:
    assert client.post("/pii/extract", json={"text": ""}).status_code == 422


def test_rejects_bad_method(client) -> None:
    resp = client.post("/pii/deidentify", json={"text": "x", "method": "encrypt"})
    assert resp.status_code == 422


def test_rejects_out_of_range_confidence(client) -> None:
    resp = client.post("/pii/extract", json={"text": "x", "confidence_threshold": 1.5})
    assert resp.status_code == 422


def test_schema_deidmethod_matches_openmed() -> None:
    # Keep the API's method enum in sync with openmed's canonical set (no model load).
    from openmed.core.pii import DeidentificationMethod

    from openmed_studio.schemas import DeidMethod

    assert set(typing.get_args(DeidMethod)) == set(
        typing.get_args(DeidentificationMethod)
    )


def test_to_entity_maps_deidentify_entity_shape() -> None:
    # deidentify() entities expose entity_type/original_text (not label/text) and may
    # carry no confidence; _to_entity must normalize that shape too. (The stub engine
    # above uses the extract_pii shape, so this covers the other branch.)
    from openmed_studio.main import _to_entity

    raw = SimpleNamespace(
        entity_type="ssn", original_text="123-45-6789", start=5, end=16
    )
    entity = _to_entity(raw)
    assert entity.label == "ssn"
    assert entity.text == "123-45-6789"
    assert (entity.start, entity.end) == (5, 16)
    assert entity.confidence is None


def test_rejects_whitespace_only_text(client) -> None:
    assert client.post("/pii/extract", json={"text": "   "}).status_code == 422


def test_rejects_oversize_text(client) -> None:
    assert client.post("/pii/extract", json={"text": "x" * 50_001}).status_code == 422


def test_max_text_chars_env_override(monkeypatch) -> None:
    # The cap is read from OPENMED_STUDIO_MAX_TEXT_LENGTH; invalid/non-positive/unset
    # values fall back to the 50k default so a typo can't silently disable the guard.
    from openmed_studio import schemas

    monkeypatch.setenv("OPENMED_STUDIO_MAX_TEXT_LENGTH", "1234")
    assert schemas._max_text_chars() == 1234
    monkeypatch.setenv("OPENMED_STUDIO_MAX_TEXT_LENGTH", "not-a-number")
    assert schemas._max_text_chars() == 50_000
    monkeypatch.setenv("OPENMED_STUDIO_MAX_TEXT_LENGTH", "0")
    assert schemas._max_text_chars() == 50_000
    monkeypatch.delenv("OPENMED_STUDIO_MAX_TEXT_LENGTH", raising=False)
    assert schemas._max_text_chars() == 50_000


# --- OpenMed-REST compat surface (/compat, OPENMED_STUDIO_COMPAT) ------------


@pytest.fixture
def compat_client(monkeypatch):
    # The compat surface mounts only when OPENMED_STUDIO_COMPAT is truthy, so build a
    # fresh app with it enabled (create_app re-reads the env on each call).
    monkeypatch.setenv("OPENMED_STUDIO_COMPAT", "1")
    compat_app = create_app()
    compat_app.dependency_overrides[get_engine] = lambda: _StubEngine()
    with TestClient(compat_app) as test_client:
        yield test_client
    compat_app.dependency_overrides.clear()


def test_compat_routes_absent_by_default(client) -> None:
    # The module app is built without OPENMED_STUDIO_COMPAT, so /compat 404s.
    assert client.post("/compat/pii/extract", json={"text": "x"}).status_code == 404


def test_compat_extract_returns_openmed_entity_shape(compat_client) -> None:
    # keep_alive is an upstream-only field; it must be accepted (not 422'd), and the
    # entities come back in openmed's shape (entity_type + metadata).
    resp = compat_client.post(
        "/compat/pii/extract", json={"text": "Patient John.", "keep_alive": "10m"}
    )
    assert resp.status_code == 200
    entity = resp.json()["entities"][0]
    assert entity["label"] == "first_name"
    assert entity["entity_type"] == "first_name"  # mirrored from label
    assert "metadata" in entity


def test_compat_deidentify_returns_openmed_result_shape(compat_client) -> None:
    resp = compat_client.post(
        "/compat/pii/deidentify",
        json={"text": "Patient John.", "method": "mask", "keep_alive": 600},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["deidentified_text"] == "[first_name] A. Doe"
    assert body["original_text"] == "Patient John."  # upstream echoes the input
    assert body["num_entities_redacted"] == 1
    assert isinstance(body["timestamp"], str)
    assert "pii_entities" in body and "entities" not in body  # upstream key name
    assert "redacted_text" in body["pii_entities"][0]


def test_compat_deidentify_includes_mapping_when_requested(compat_client) -> None:
    resp = compat_client.post(
        "/compat/pii/deidentify",
        json={"text": "Patient John.", "method": "replace", "keep_mapping": True},
    )
    assert resp.status_code == 200
    assert resp.json()["mapping"] == {"[first_name]": "John"}


def test_rejects_invalid_model_name(client) -> None:
    resp = client.post(
        "/pii/extract", json={"text": "x", "model_name": "../etc/passwd"}
    )
    assert resp.status_code == 422


def test_extract_accepts_lang_and_model_name(client) -> None:
    resp = client.post(
        "/pii/extract",
        json={"text": "x", "lang": "fr", "model_name": "OpenMed/Some-Model"},
    )
    assert resp.status_code == 200


def test_deidentify_accepts_date_controls(client) -> None:
    resp = client.post(
        "/pii/deidentify",
        json={"text": "x", "method": "shift_dates", "date_shift_days": 180},
    )
    assert resp.status_code == 200


def test_batch_returns_per_item_results(client) -> None:
    resp = client.post(
        "/pii/deidentify/batch",
        json={"items": ["Patient John.", "Patient Jane."], "method": "mask"},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert all(r["deidentified_text"] == "[first_name] A. Doe" for r in results)


def test_batch_rejects_empty_items(client) -> None:
    assert client.post("/pii/deidentify/batch", json={"items": []}).status_code == 422


def test_batch_rejects_too_many_items(client) -> None:
    resp = client.post("/pii/deidentify/batch", json={"items": ["x"] * 101})
    assert resp.status_code == 422


def test_value_error_from_engine_maps_to_400() -> None:
    with _override_engine(_RaisingEngine(ValueError("bad option"))) as override:
        resp = override.post("/pii/deidentify", json={"text": "x", "method": "mask"})
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "bad_request"
    assert error["message"] == "bad option"


def test_backend_failure_maps_to_503() -> None:
    with _override_engine(_RaisingEngine(RuntimeError("model load exploded"))) as ov:
        resp = ov.post("/pii/extract", json={"text": "x"})
    assert resp.status_code == 503
    error = resp.json()["error"]
    assert error["code"] == "service_unavailable"
    assert "exploded" not in error["message"]  # internal detail must not leak


def test_validation_error_uses_error_envelope(client) -> None:
    # Schema failures (422) are wrapped in the same {"error": {...}} envelope, with the
    # field errors carried in `details` — and `input` stripped so PHI isn't echoed back.
    resp = client.post("/pii/extract", json={"text": "x", "bogus": 1})
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert isinstance(error["message"], str)
    assert isinstance(error["details"], list) and error["details"]  # the field errors
    assert all("input" not in item for item in error["details"])  # no request echo


def test_pii_requires_api_key_when_configured(client, monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "s3cret")
    assert client.post("/pii/extract", json={"text": "x"}).status_code == 401
    wrong = client.post("/pii/extract", json={"text": "x"}, headers={"X-API-Key": "no"})
    assert wrong.status_code == 401
    right = client.post(
        "/pii/extract", json={"text": "x"}, headers={"X-API-Key": "s3cret"}
    )
    assert right.status_code == 200


def test_health_open_and_reports_auth_required(client, monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "s3cret")
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["auth_required"] is True


# --- Model-backed tests (real OpenMed engine; need --run-model) -------------


@pytest.fixture
def model_client(loader):
    engine = PIIEngine(loader=loader)
    app.dependency_overrides[get_engine] = lambda: engine
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.mark.model
def test_extract_detects_real_pii(model_client, note) -> None:
    resp = model_client.post("/pii/extract", json={"text": note})
    assert resp.status_code == 200
    found = {(e["label"], e["text"]) for e in resp.json()["entities"]}
    assert ("ssn", "123-45-6789") in found


@pytest.mark.model
def test_deidentify_masks_real_pii(model_client, note) -> None:
    resp = model_client.post("/pii/deidentify", json={"text": note, "method": "mask"})
    assert resp.status_code == 200
    text = resp.json()["deidentified_text"]
    assert "123-45-6789" not in text
    assert "john.doe@example.com" not in text


@pytest.mark.model
def test_date_shift_without_shift_method_returns_400(model_client) -> None:
    # openmed raises ValueError (date_shift_days requires method='shift_dates'),
    # which the API surfaces as 400 — exercising the now-reachable error path.
    resp = model_client.post(
        "/pii/deidentify",
        json={"text": "Seen on 03/22/2024.", "method": "mask", "date_shift_days": 30},
    )
    assert resp.status_code == 400
