"""Tests for ``streamlit_app.py`` — the AppTest render path and the HTTP client.

Skipped unless the optional ``ui`` extra (streamlit + requests) is installed
(run with ``uv run --extra ui pytest``). The network is mocked at
``requests.Session.request`` with a call-recording router, so these need no
running service and no model — and a mock that fails to intercept (e.g. after a
refactor) is caught because the recorded-call assertions and the sentinel
``HEALTH``/``DEID`` payloads can only come from the mock, never a live service.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("requests")

import requests  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import streamlit_app  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")

# Sentinel values a live service would never return — so any assertion on them
# proves the data came from the mock, not from a real :8080 service.
HEALTH = {
    "status": "ok",
    "service": "openmed-studio",
    "version": "9.9.9-stub",
    "model": "STUB/sentinel-model",
    "backend": "hf",
    "max_text_chars": 50000,
    "model_loaded": True,
    "auth_required": False,
}
HEALTH_AUTH = {**HEALTH, "auth_required": True}

DEID = {
    "deidentified_text": "[[STUB-DEID-OUTPUT]]",
    "method": "mask",
    "entities": [
        {
            "label": "first_name",
            "text": "John",
            "start": 8,
            "end": 12,
            "confidence": 0.99,
        },
        {
            "label": "last_name",
            "text": "Doe",
            "start": 13,
            "end": 16,
            "confidence": 0.98,
        },
    ],
    "mapping": None,
}
DEID_MAP = {
    "deidentified_text": "Patient [PERSON_1].",
    "method": "replace",
    "entities": [
        {
            "label": "first_name",
            "text": "John",
            "start": 8,
            "end": 12,
            "confidence": 0.9,
        }
    ],
    "mapping": {"PERSON_1": "John"},
}


class _FakeResp:
    def __init__(self, status_code=200, payload=None, reason="OK", text=""):
        self.status_code = status_code
        self._payload = payload
        self.reason = reason
        self.text = text

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def _patch(monkeypatch, routes=None, exc=None):
    """Patch Session.request with a call-recording router; return the calls list.

    Each recorded call is ``(METHOD, url, json_body)`` so tests can assert not just
    *that* an endpoint was hit but *what* payload the UI sent.
    """
    routes = routes or {}
    calls: list[tuple[str, str, Any]] = []

    def _request(_self, method, url, **_kwargs):
        calls.append((method.upper(), url, _kwargs.get("json")))
        if exc is not None:
            raise exc
        for (verb, suffix), resp in routes.items():
            if method.upper() == verb and url.endswith(suffix):
                return resp
        return _FakeResp(404, {"error": {"code": "not_found", "message": "no route"}})

    monkeypatch.setattr(requests.sessions.Session, "request", _request)
    return calls


def _posted(calls, suffix):
    return any(verb == "POST" and url.endswith(suffix) for verb, url, _ in calls)


def _body_for(calls, suffix):
    """The JSON body of the first POST whose URL ends with ``suffix`` (or None)."""
    for verb, url, body in calls:
        if verb == "POST" and url.endswith(suffix):
            return body
    return None


def _html(at):
    return " ".join(getattr(el, "body", "") for el in at.get("html"))


def _set_area(at, label, value):
    next(t for t in at.text_area if t.label == label).set_value(value)


def _click(at, label):
    next(b for b in at.button if b.label == label).click().run(timeout=30)


HEALTH_ROUTE = {("GET", "/health"): _FakeResp(200, HEALTH)}


@pytest.fixture(autouse=True)
def _clear_caches():
    """Streamlit caches are process-global; reset them around each test."""
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


# --- AppTest: sidebar / connection status -------------------------------------
def test_app_renders_when_service_unreachable(monkeypatch):
    calls = _patch(monkeypatch, exc=requests.ConnectionError("down"))
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "PII / PHI de-identification"
    assert len(at.tabs) == 4
    assert any("Not connected" in m.value for m in at.sidebar.markdown)
    assert calls  # the mock was actually exercised (health was attempted)


def test_app_shows_connected_status(monkeypatch):
    calls = _patch(monkeypatch, HEALTH_ROUTE)
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception
    assert any("Connected" in m.value for m in at.sidebar.markdown)
    # The sentinel model can only appear if the mock (not a live service) served /health.
    assert any("STUB/sentinel-model" in c.value for c in at.sidebar.caption)
    assert any(verb == "GET" and url.endswith("/health") for verb, url, _ in calls)


def test_sidebar_warns_when_auth_required(monkeypatch):
    _patch(monkeypatch, {("GET", "/health"): _FakeResp(200, HEALTH_AUTH)})
    at = AppTest.from_file(APP).run(timeout=30)
    assert any("Service requires an API key" in w.value for w in at.sidebar.warning)


# --- AppTest: single-note de-identify -----------------------------------------
def test_single_note_renders_metrics_and_output(monkeypatch):
    calls = _patch(
        monkeypatch,
        {**HEALTH_ROUTE, ("POST", "/pii/deidentify"): _FakeResp(200, DEID)},
    )
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    metrics = {m.label: str(m.value) for m in at.metric}
    assert metrics.get("Entities found") == "2"
    assert metrics.get("Method") == "mask"
    body = _html(at)
    assert "[[STUB-DEID-OUTPUT]]" in body  # de-identified text rendered
    assert "<mark" in body  # original highlighted
    assert _posted(calls, "/pii/deidentify")


def test_single_note_sends_sidebar_options_in_body(monkeypatch):
    # Verifies the sidebar -> build_base_opts -> request-body wiring end to end.
    calls = _patch(
        monkeypatch,
        {**HEALTH_ROUTE, ("POST", "/pii/deidentify"): _FakeResp(200, DEID)},
    )
    at = AppTest.from_file(APP).run(timeout=30)
    next(s for s in at.segmented_control if s.label == "Method").set_value("replace")
    next(t for t in at.toggle if t.label == "Deterministic replace").set_value(True)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    body = _body_for(calls, "/pii/deidentify")
    assert body is not None
    assert body["method"] == "replace"
    assert body["consistent"] is True
    assert "seed" in body  # included because consistent is on
    assert body["keep_mapping"] is True
    assert "confidence_threshold" in body


def test_single_note_persists_mapping_to_session_state(monkeypatch):
    _patch(
        monkeypatch,
        {**HEALTH_ROUTE, ("POST", "/pii/deidentify"): _FakeResp(200, DEID_MAP)},
    )
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert at.session_state["last_mapping"] == {"PERSON_1": "John"}
    assert at.session_state["last_deidentified"] == "Patient [PERSON_1]."


def test_single_note_clears_stale_mapping_when_none(monkeypatch):
    # Regression guard: a run whose result has no mapping must clear a previous one,
    # so the Re-identify tab can't pair this run's text with a stale mapping.
    _patch(
        monkeypatch, {**HEALTH_ROUTE, ("POST", "/pii/deidentify"): _FakeResp(200, DEID)}
    )
    at = AppTest.from_file(APP)
    at.session_state["last_mapping"] = {"OLD_1": "secret"}
    at.run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert at.session_state["last_mapping"] is None


def test_single_note_connection_error_shows_guidance(monkeypatch):
    # End-to-end transport-failure branch: submit with the service unreachable.
    _patch(monkeypatch, exc=requests.ConnectionError("down"))
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert any("Could not reach the service" in e.value for e in at.error)
    assert not at.metric  # no result rendered


def test_empty_single_note_warns_and_skips_call(monkeypatch):
    calls = _patch(monkeypatch, HEALTH_ROUTE)
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "   ")  # whitespace only
    _click(at, "De-identify")

    assert not at.exception
    assert any("Enter some text" in w.value for w in at.warning)
    assert not _posted(calls, "/pii/deidentify")  # the call was actually skipped


def test_error_envelope_renders_message_and_details(monkeypatch):
    details = {"max": 50000}
    _patch(
        monkeypatch,
        {
            **HEALTH_ROUTE,
            ("POST", "/pii/deidentify"): _FakeResp(
                422,
                {
                    "error": {
                        "code": "validation_error",
                        "message": "text too long",
                        "details": details,
                    }
                },
            ),
        },
    )
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert any("422" in e.value and "text too long" in e.value for e in at.error)
    # st.json renders the details object as a JSON string in AppTest.
    assert any(json.loads(e.value) == details for e in at.json)


def test_401_shows_api_key_hint(monkeypatch):
    _patch(
        monkeypatch,
        {
            **HEALTH_ROUTE,
            ("POST", "/pii/deidentify"): _FakeResp(
                401, {"error": {"message": "no key"}}
            ),
        },
    )
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert any("OPENMED_STUDIO_API_KEY" in c.value for c in at.caption)


# --- AppTest: detect (extract) ------------------------------------------------
def test_detect_renders_entities_and_sends_extract_body(monkeypatch):
    extract = {
        "entities": [
            {
                "label": "first_name",
                "text": "John",
                "start": 8,
                "end": 12,
                "confidence": 0.9,
            },
            {
                "label": "last_name",
                "text": "Doe",
                "start": 13,
                "end": 16,
                "confidence": 0.8,
            },
        ]
    }
    calls = _patch(
        monkeypatch,
        {**HEALTH_ROUTE, ("POST", "/pii/extract"): _FakeResp(200, extract)},
    )
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note to scan", "Patient John Doe.")
    _click(at, "Detect")

    assert not at.exception
    assert any(m.label == "Entities found" and str(m.value) == "2" for m in at.metric)
    assert "<mark" in _html(at)  # highlighted, with a legend
    body = _body_for(calls, "/pii/extract")
    assert body is not None
    assert "use_smart_merging" in body
    assert "confidence_threshold" in body and "lang" in body
    # Extract takes no method/keep_mapping — sending them would 422 the strict schema.
    assert "method" not in body and "keep_mapping" not in body


# --- AppTest: batch -----------------------------------------------------------
def test_batch_deidentify_renders(monkeypatch):
    # The data_editor seeds one non-empty note by default, so clicking "De-identify
    # all" without editing exercises the request/zip/table path with a single item.
    calls = _patch(
        monkeypatch,
        {
            **HEALTH_ROUTE,
            ("POST", "/pii/deidentify/batch"): _FakeResp(200, {"results": [DEID]}),
        },
    )
    at = AppTest.from_file(APP).run(timeout=30)
    _click(at, "De-identify all")

    assert not at.exception
    assert _posted(calls, "/pii/deidentify/batch")
    assert any(
        m.label == "Notes de-identified" and str(m.value) == "1" for m in at.metric
    )


# --- AppTest: re-identify -----------------------------------------------------
def test_reidentify_renders_text(monkeypatch):
    calls = _patch(
        monkeypatch,
        {
            **HEALTH_ROUTE,
            ("POST", "/pii/reidentify"): _FakeResp(200, {"text": "[[STUB-RESTORED]]"}),
        },
    )
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "De-identified text", "Patient [PERSON_1].")
    _set_area(at, "Mapping (JSON)", '{"PERSON_1": "John"}')
    _click(at, "Re-identify")

    assert not at.exception
    assert "[[STUB-RESTORED]]" in _html(at)
    assert _posted(calls, "/pii/reidentify")


def test_reidentify_invalid_json_errors_without_call(monkeypatch):
    calls = _patch(monkeypatch, HEALTH_ROUTE)
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Mapping (JSON)", "{not valid json")
    _click(at, "Re-identify")

    assert not at.exception
    assert any("not valid JSON" in e.value for e in at.error)
    assert not _posted(calls, "/pii/reidentify")


def test_reidentify_empty_mapping_warns_without_call(monkeypatch):
    calls = _patch(monkeypatch, HEALTH_ROUTE)
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Mapping (JSON)", "{}")
    _click(at, "Re-identify")

    assert not at.exception
    assert any("non-empty mapping" in w.value for w in at.warning)
    assert not _posted(calls, "/pii/reidentify")


# --- HTTP client (api / fetch_health), directly ------------------------------
def test_api_returns_json_on_success(monkeypatch):
    _patch(monkeypatch, {("POST", "/pii/deidentify"): _FakeResp(200, DEID)})
    assert (
        streamlit_app.api("http://api-ok", "", "/pii/deidentify", {"text": "hi"})
        == DEID
    )


def test_api_returns_none_on_4xx(monkeypatch):
    _patch(
        monkeypatch,
        {("POST", "/pii/deidentify"): _FakeResp(422, {"error": {"message": "bad"}})},
    )
    assert (
        streamlit_app.api("http://api-4xx", "", "/pii/deidentify", {"text": ""}) is None
    )


def test_api_returns_none_on_connection_error(monkeypatch):
    _patch(monkeypatch, exc=requests.ConnectionError("down"))
    assert (
        streamlit_app.api("http://api-down", "", "/pii/deidentify", {"text": "hi"})
        is None
    )


def test_api_non_json_4xx_body_falls_back_to_text(monkeypatch):
    # No JSON body: resp.json() raises ValueError → message=resp.text fallback path.
    _patch(
        monkeypatch,
        {
            ("POST", "/pii/deidentify"): _FakeResp(
                500, payload=None, reason="Server Error", text="boom"
            )
        },
    )
    assert (
        streamlit_app.api("http://api-500", "", "/pii/deidentify", {"text": "hi"})
        is None
    )


def test_api_sends_api_key_header(monkeypatch):
    seen = {}

    def _request(self, _method, _url, **_kwargs):
        seen["key"] = self.headers.get("X-API-Key")
        return _FakeResp(200, DEID)

    monkeypatch.setattr(requests.sessions.Session, "request", _request)
    streamlit_app.api("http://api-key", "secret-key", "/pii/deidentify", {"text": "hi"})
    assert seen["key"] == "secret-key"


def test_fetch_health_returns_payload_when_ok(monkeypatch):
    _patch(monkeypatch, {("GET", "/health"): _FakeResp(200, HEALTH)})
    assert streamlit_app.fetch_health("http://h-ok", "") == HEALTH


def test_fetch_health_none_on_exception(monkeypatch):
    _patch(monkeypatch, exc=requests.Timeout("slow"))
    assert streamlit_app.fetch_health("http://h-exc", "") is None


def test_fetch_health_none_on_non_2xx(monkeypatch):
    _patch(monkeypatch, {("GET", "/health"): _FakeResp(503, {"error": {}})})
    assert streamlit_app.fetch_health("http://h-503", "") is None
