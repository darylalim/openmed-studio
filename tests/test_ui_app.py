"""Tests for ``streamlit_app.py`` — the in-process AppTest render path.

The engine is stubbed by patching ``service.build_engine`` (the shared module the
running app imports), so these need no model and no network. Sentinel values a real
model would never produce (``[[STUB-DEID-OUTPUT]]``, ``STUB/sentinel-model``) prove
the rendered data came from the stub, not a live model.

Skipped unless ``streamlit`` is importable (it is a core dependency, so the default
suite runs them).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from openmed_studio import service  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


class _StubEngine:
    """A model-free engine returning openmed-shaped objects for the service to adapt."""

    model_name = "STUB/sentinel-model"
    backend: str | None = "hf"
    is_loaded = True

    def extract(self, _text, **_):
        return [
            SimpleNamespace(
                label="first_name", text="John", start=8, end=12, confidence=0.9
            ),
            SimpleNamespace(
                label="last_name", text="Doe", start=13, end=16, confidence=0.8
            ),
        ]

    def deidentify(self, _text, *, keep_mapping=False, **_):
        mapping = {"PERSON_1": "John"} if keep_mapping else None
        return SimpleNamespace(
            deidentified_text="[[STUB-DEID-OUTPUT]]",
            pii_entities=[
                SimpleNamespace(
                    label="first_name", text="John", start=8, end=12, confidence=0.99
                ),
                SimpleNamespace(
                    label="last_name", text="Doe", start=13, end=16, confidence=0.98
                ),
            ],
            mapping=mapping,
        )

    def reidentify(self, _deidentified_text, _mapping):
        return "[[STUB-RESTORED]]"


class _RaisingEngine(_StubEngine):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def deidentify(self, _text, **_):
        raise self._exc


@pytest.fixture(autouse=True)
def _clear_caches():
    """Streamlit caches are process-global; reset them around each test."""
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def _use_engine(monkeypatch, engine) -> None:
    """Make the app's cached get_engine() build this stub (cache cleared per test)."""
    monkeypatch.setattr(service, "build_engine", lambda: engine)


def _html(at):
    return " ".join(getattr(el, "body", "") for el in at.get("html"))


def _set_area(at, label, value):
    next(t for t in at.text_area if t.label == label).set_value(value)


def _click(at, label):
    next(b for b in at.button if b.label == label).click().run(timeout=30)


# --- sidebar / render --------------------------------------------------------
def test_app_renders(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "PII / PHI de-identification"
    assert len(at.tabs) == 4
    # The sentinel model name can only appear if the stub (not a live model) was used.
    assert any("STUB/sentinel-model" in c.value for c in at.sidebar.caption)
    assert any("model loaded" in c.value for c in at.sidebar.caption)


def test_sidebar_reports_lazy_load(monkeypatch):
    engine = _StubEngine()
    engine.is_loaded = False  # type: ignore[misc]  # instance override of class attr
    _use_engine(monkeypatch, engine)
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception
    assert any("loads on first request" in c.value for c in at.sidebar.caption)


# --- single-note de-identify -------------------------------------------------
def test_single_note_renders_metrics_and_output(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
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


def test_single_note_persists_mapping_to_session_state(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    # "Keep mapping" defaults on, so the result carries a mapping.
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert at.session_state["last_mapping"] == {"PERSON_1": "John"}
    assert at.session_state["last_deidentified"] == "[[STUB-DEID-OUTPUT]]"


def test_single_note_clears_stale_mapping_when_keep_off(monkeypatch):
    # Regression guard: a run whose result has no mapping must clear a previous one.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP)
    at.session_state["last_mapping"] = {"OLD_1": "secret"}
    at.run(timeout=30)
    next(t for t in at.toggle if t.label == "Keep mapping").set_value(False)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert at.session_state["last_mapping"] is None


def test_empty_single_note_warns_and_skips(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "   ")  # whitespace only
    _click(at, "De-identify")

    assert not at.exception
    assert any("Enter some text" in w.value for w in at.warning)
    assert not at.metric  # no result rendered


def test_engine_error_renders_message(monkeypatch):
    _use_engine(monkeypatch, _RaisingEngine(ValueError("bad option here")))
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert any("bad option here" in e.value for e in at.error)
    assert not at.metric


def test_oversize_text_rejected_without_phi_echo(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    secret = "SECRET-PHI-"
    _set_area(at, "Clinical note", secret * 6000)  # > 50k chars → validation rejects
    _click(at, "De-identify")

    assert not at.exception
    assert at.error  # a ServiceError was surfaced
    assert all(secret not in e.value for e in at.error)  # the input is not echoed back
    assert not at.metric


# --- detect ------------------------------------------------------------------
def test_detect_renders_entities(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note to scan", "Patient John Doe.")
    _click(at, "Detect")

    assert not at.exception
    assert any(m.label == "Entities found" and str(m.value) == "2" for m in at.metric)
    assert "<mark" in _html(at)  # highlighted, with a legend


# --- batch -------------------------------------------------------------------
def test_batch_renders(monkeypatch):
    # The data_editor seeds one non-empty note, so clicking without editing exercises
    # the per-item loop / zip / table path with a single note.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _click(at, "De-identify all")

    assert not at.exception
    assert any(
        m.label == "Notes de-identified" and str(m.value) == "1" for m in at.metric
    )


# --- re-identify -------------------------------------------------------------
def test_reidentify_renders_text(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "De-identified text", "Patient [PERSON_1].")
    _set_area(at, "Mapping (JSON)", '{"PERSON_1": "John"}')
    _click(at, "Re-identify")

    assert not at.exception
    assert "[[STUB-RESTORED]]" in _html(at)


def test_reidentify_invalid_json_errors_without_call(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Mapping (JSON)", "{not valid json")
    _click(at, "Re-identify")

    assert not at.exception
    assert any("not valid JSON" in e.value for e in at.error)


def test_reidentify_empty_mapping_warns_without_call(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Mapping (JSON)", "{}")
    _click(at, "Re-identify")

    assert not at.exception
    assert any("non-empty mapping" in w.value for w in at.warning)
