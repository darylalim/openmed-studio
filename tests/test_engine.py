"""Tests for the framework-free PIIEngine (openmed_studio.engine).

The fast tests verify the lazy-loading contract without touching a model; the
``@pytest.mark.model`` tests drive the real OpenMed model and are skipped unless
``--run-model`` is passed (reusing the session-scoped ``loader`` fixture).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from openmed_studio import DEFAULT_PII_MODEL, PIIEngine

if TYPE_CHECKING:
    from openmed import ModelLoader


def test_engine_is_lazy_by_default() -> None:
    # Constructing the engine must not instantiate a ModelLoader or load a model.
    engine = PIIEngine()
    assert engine.is_loaded is False
    assert engine.lang == "en"
    assert engine.model_name is None


def test_default_pii_model_is_an_openmed_repo() -> None:
    assert DEFAULT_PII_MODEL.startswith("OpenMed/")


# --- Backend selection plumbing (no model) ----------------------------------


def test_engine_backend_defaults_to_none() -> None:
    assert PIIEngine().backend is None
    assert PIIEngine(backend="mlx").backend == "mlx"


def test_engine_default_backend_builds_bare_loader(monkeypatch) -> None:
    # backend=None must construct ModelLoader() with no config (openmed auto-detects).
    import openmed

    captured = {}

    class _FakeLoader:
        def __init__(self, config=None):
            captured["config"] = config

    def _no_config(**_kwargs):
        raise AssertionError("OpenMedConfig must not be built when backend is None")

    monkeypatch.setattr(openmed, "ModelLoader", _FakeLoader)
    monkeypatch.setattr(openmed, "OpenMedConfig", _no_config)

    engine = PIIEngine()
    assert isinstance(engine.loader, _FakeLoader)
    assert captured["config"] is None


def test_engine_backend_forwarded_via_openmedconfig(monkeypatch) -> None:
    # backend="mlx" must reach ModelLoader as OpenMedConfig(backend="mlx").
    import openmed

    captured = {}

    class _FakeConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs

    class _FakeLoader:
        def __init__(self, config=None):
            captured["loader_config"] = config

    monkeypatch.setattr(openmed, "OpenMedConfig", _FakeConfig)
    monkeypatch.setattr(openmed, "ModelLoader", _FakeLoader)

    engine = PIIEngine(backend="mlx")
    assert isinstance(engine.loader, _FakeLoader)
    assert captured["config_kwargs"] == {"backend": "mlx"}
    assert isinstance(captured["loader_config"], _FakeConfig)


# --- deidentify delegation (no model) ---------------------------------------


def test_deidentify_delegates_every_method_to_openmed(monkeypatch) -> None:
    # The engine special-cases nothing: shift_dates (and its date controls) must be
    # forwarded straight to openmed.deidentify, just like mask/replace/hash/remove.
    import openmed

    captured: dict[str, object] = {}

    def fake_deidentify(text, **kwargs):
        captured.update(kwargs)
        captured["text"] = text
        return SimpleNamespace(deidentified_text="ok", pii_entities=[], mapping=None)

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    # A non-None loader short-circuits the lazy loader, so no model is built; the
    # stub is never actually used, hence the cast for the type checker.
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    result = engine.deidentify(
        "Seen 03/22/2024.",
        method="shift_dates",
        date_shift_days=180,
        keep_year=False,
    )

    assert result.deidentified_text == "ok"
    assert captured["text"] == "Seen 03/22/2024."
    assert captured["method"] == "shift_dates"
    assert captured["date_shift_days"] == 180
    assert captured["keep_year"] is False
    assert captured["loader"] is engine.loader  # the shared loader is threaded through


# --- Model-backed tests (real OpenMed engine; need --run-model) -------------


@pytest.mark.model
def test_engine_marks_loaded_once_a_loader_is_present(loader) -> None:
    engine = PIIEngine(loader=loader)
    assert engine.is_loaded is True
    assert engine.loader is loader


@pytest.mark.model
def test_engine_extracts_and_deidentifies(loader, note) -> None:
    engine = PIIEngine(loader=loader)

    found = {(e.label, e.text) for e in engine.extract(note)}
    assert ("ssn", "123-45-6789") in found

    masked = engine.deidentify(note, method="mask").deidentified_text
    assert "123-45-6789" not in masked


@pytest.mark.model
def test_engine_round_trips_with_kept_mapping(loader, note) -> None:
    engine = PIIEngine(loader=loader)
    result = engine.deidentify(
        note, method="replace", consistent=True, seed=7, keep_mapping=True
    )
    assert result.deidentified_text != note  # PII was actually replaced
    assert result.mapping  # non-empty mapping
    assert PIIEngine.reidentify(result.deidentified_text, result.mapping) == note
