"""Tests for the in-process service seam (``openmed_studio.service``).

These exercise backend resolution, the dict adapters, the success paths, and the
error taxonomy with a model-free stub engine — no ``--run-model``, no network.
Validation rules are covered separately in ``test_validation.py``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import cast

import pytest

from openmed_studio import PIIEngine, service
from openmed_studio.service import ServiceError


class _StubEngine:
    """A model-free stand-in for PIIEngine with the same call surface."""

    model_name = "stub-model"
    backend: str | None = None
    is_loaded = True

    def extract(self, _text, **_):
        return [
            SimpleNamespace(
                label="first_name", text="John", start=0, end=4, confidence=0.99
            )
        ]

    def deidentify(self, _text, *, keep_mapping=False, **_):
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

    def analyze(self, _text, **_):
        return [
            SimpleNamespace(
                label="DISEASE", text="diabetes", start=0, end=8, confidence=0.97
            )
        ]

    def reidentify(self, deidentified_text, mapping):
        for key, value in mapping.items():
            deidentified_text = deidentified_text.replace(key, value)
        return deidentified_text


class _RaisingEngine(_StubEngine):
    """Stub whose model calls raise, to exercise the error taxonomy."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def extract(self, _text, **_):
        raise self._exc

    def deidentify(self, _text, **_):
        raise self._exc

    def analyze(self, _text, **_):
        raise self._exc

    def reidentify(self, deidentified_text, mapping):
        raise self._exc


def _stub() -> PIIEngine:
    # The stub is structural, not a real PIIEngine; cast to satisfy the typed seam.
    return cast("PIIEngine", _StubEngine())


def _raising(exc: Exception) -> PIIEngine:
    return cast("PIIEngine", _RaisingEngine(exc))


def _capturing(method: str = "deidentify") -> tuple[PIIEngine, dict[str, object]]:
    """A stub engine whose ``method`` records its kwargs; returns ``(engine, captured)``.

    Lets the forwarding tests assert what reaches the engine without each re-declaring an
    identical capturing stub. ``deidentify`` returns the canned ``DeidentificationResult``
    shape the dict adapter consumes; ``analyze`` returns an empty entity list.
    """
    captured: dict[str, object] = {}
    canned = (
        []
        if method == "analyze"
        else SimpleNamespace(deidentified_text="ok", pii_entities=[], mapping=None)
    )

    def _record(self, _text, **kwargs):
        captured.update(kwargs)
        return canned

    capturing = type("_Capturing", (_StubEngine,), {method: _record})
    return cast("PIIEngine", capturing()), captured


# --- backend resolution (no model) ------------------------------------------


def test_resolve_backend_unset_is_none(monkeypatch) -> None:
    monkeypatch.delenv(service.BACKEND_ENV, raising=False)
    assert service.resolve_backend() is None


def test_resolve_backend_empty_is_none(monkeypatch) -> None:
    # A set-but-empty value is treated like unset (auto-detect), not an error.
    monkeypatch.setenv(service.BACKEND_ENV, "")
    assert service.resolve_backend() is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("mlx", "mlx"), ("hf", "hf"), ("  MLX ", "mlx"), ("HF", "hf")],
)
def test_resolve_backend_normalizes_valid_values(monkeypatch, value, expected) -> None:
    monkeypatch.setenv(service.BACKEND_ENV, value)
    assert service.resolve_backend() == expected


def test_resolve_backend_invalid_falls_back_to_auto(monkeypatch, caplog) -> None:
    # A typo must degrade to auto-detect (None) AND warn, naming the bad value.
    monkeypatch.setenv(service.BACKEND_ENV, "cuda")
    with caplog.at_level(logging.WARNING, logger="openmed_studio"):
        assert service.resolve_backend() is None
    assert "cuda" in caplog.text


def test_build_engine_wires_resolved_backend(monkeypatch) -> None:
    # build_engine() constructs a real PIIEngine carrying the resolved backend while
    # staying lazy (no model load).
    monkeypatch.setenv(service.BACKEND_ENV, "mlx")
    engine = service.build_engine()
    assert isinstance(engine, PIIEngine)
    assert engine.backend == "mlx"
    assert engine.is_loaded is False  # constructed but no model loaded


# --- adapters + success paths (stub engine) ---------------------------------


def test_extract_returns_entity_dicts() -> None:
    result = service.extract(_stub(), "Patient John.")
    assert result["entities"] == [
        {
            "label": "first_name",
            "text": "John",
            "start": 0,
            "end": 4,
            "confidence": 0.99,
        }
    ]


def test_deidentify_omits_mapping_by_default() -> None:
    result = service.deidentify(_stub(), "Patient John.", method="mask")
    assert result["deidentified_text"] == "[first_name] A. Doe"
    assert result["method"] == "mask"
    assert result["mapping"] is None


def test_deidentify_includes_mapping_when_requested() -> None:
    result = service.deidentify(
        _stub(), "Patient John.", method="replace", keep_mapping=True
    )
    assert result["mapping"] == {"[first_name]": "John"}


def test_deidentify_forwards_use_safety_sweep_to_engine() -> None:
    # The 1.6.0 structured-identifier safety sweep is wired through the service layer:
    # on by default, and overridable per request (the engine->openmed hop is covered
    # in test_engine.py; this pins the service->engine hop).
    engine, captured = _capturing()
    service.deidentify(engine, "x", method="mask")
    assert captured["use_safety_sweep"] is True
    captured.clear()
    service.deidentify(engine, "x", method="mask", use_safety_sweep=False)
    assert captured["use_safety_sweep"] is False


def test_deidentify_forwards_locale_to_engine() -> None:
    # A valid `replace` locale passes validation and reaches the engine unchanged
    # (default None when unset). Pins the validation->service->engine hop; the
    # engine->openmed hop is covered in test_engine.py.
    engine, captured = _capturing()
    service.deidentify(engine, "x", method="mask")
    assert captured["locale"] is None
    captured.clear()
    service.deidentify(engine, "x", method="replace", locale="pt_BR")
    assert captured["locale"] == "pt_BR"


def test_deidentify_forwards_use_smart_merging_to_engine() -> None:
    # deidentify forwards use_smart_merging like extract does: on by default, and
    # overridable per request. Pins the service->engine hop (engine->openmed is in
    # test_engine.py).
    engine, captured = _capturing()
    service.deidentify(engine, "x", method="mask")
    assert captured["use_smart_merging"] is True
    captured.clear()
    service.deidentify(engine, "x", method="mask", use_smart_merging=False)
    assert captured["use_smart_merging"] is False


def test_deidentify_batch_returns_per_item_results() -> None:
    result = service.deidentify_batch(
        _stub(), ["Patient John.", "Patient Jane."], method="mask"
    )
    results = result["results"]
    assert len(results) == 2
    assert all(r["ok"] for r in results)
    assert all(r["deidentified_text"] == "[first_name] A. Doe" for r in results)


def test_batch_isolates_failing_note_keeps_others() -> None:
    # One pathological note fails (ValueError) while the others succeed — the per-item
    # isolation a single shared _run would not provide (it would abort the whole batch).
    class _Mixed(_StubEngine):
        def deidentify(self, _text, **kwargs):
            if "boom" in _text:
                raise ValueError("note-specific failure")
            return super().deidentify(_text, **kwargs)

    engine = cast("PIIEngine", _Mixed())
    result = service.deidentify_batch(
        engine, ["Patient John.", "boom note", "Patient Jane."], method="mask"
    )
    results = result["results"]
    assert [r["ok"] for r in results] == [True, False, True]
    assert "note-specific failure" in results[1]["error"]
    assert results[0]["deidentified_text"] == "[first_name] A. Doe"


def test_reidentify_restores() -> None:
    result = service.reidentify(_stub(), "Hi [first_name].", {"[first_name]": "John"})
    assert result["text"] == "Hi John."


def test_analyze_returns_entity_dicts() -> None:
    # NER flows through the same _entity_dict adapter; UPPERCASE labels are preserved.
    result = service.analyze(
        _stub(), "Has diabetes.", model_name="disease_detection_superclinical_141m"
    )
    assert result["entities"] == [
        {
            "label": "DISEASE",
            "text": "diabetes",
            "start": 0,
            "end": 8,
            "confidence": 0.97,
        }
    ]


def test_analyze_forwards_options_to_engine() -> None:
    # The validated model_name/confidence/aggregation/group_entities reach engine.analyze
    # (the engine->openmed hop is covered in test_engine.py).
    engine, captured = _capturing("analyze")
    service.analyze(
        engine,
        "x",
        model_name="anatomy_detection_superclinical_141m",
        confidence_threshold=0.4,
        aggregation_strategy="first",
        group_entities=True,
    )
    assert captured["model_name"] == "anatomy_detection_superclinical_141m"
    assert captured["confidence_threshold"] == 0.4
    assert captured["aggregation_strategy"] == "first"
    assert captured["group_entities"] is True


def test_analyze_uses_ner_defaults_when_omitted() -> None:
    # NerRequest's defaults reach the engine: confidence_threshold is 0.0 (openmed's NER
    # default — deliberately NOT the de-identify 0.5/0.7), aggregation 'simple', no grouping.
    engine, captured = _capturing("analyze")
    service.analyze(engine, "x", model_name="disease_detection_superclinical_141m")
    assert captured["confidence_threshold"] == 0.0
    assert captured["aggregation_strategy"] == "simple"
    assert captured["group_entities"] is False


def test_analyze_value_error_maps_to_service_error() -> None:
    with pytest.raises(ServiceError, match="bad option"):
        service.analyze(
            _raising(ValueError("bad option")),
            "x",
            model_name="disease_detection_superclinical_141m",
        )


def test_analyze_backend_failure_does_not_leak() -> None:
    with pytest.raises(ServiceError) as excinfo:
        service.analyze(
            _raising(RuntimeError("ner model exploded")),
            "x",
            model_name="disease_detection_superclinical_141m",
        )
    message = str(excinfo.value)
    assert "exploded" not in message
    assert "unavailable" in message.lower()


def test_entity_dict_maps_deidentify_entity_shape() -> None:
    # deidentify() entities expose entity_type/original_text (not label/text) and may
    # carry no confidence; _entity_dict must normalize that shape too.
    raw = SimpleNamespace(
        entity_type="ssn", original_text="123-45-6789", start=5, end=16
    )
    entity = service._entity_dict(raw)
    assert entity["label"] == "ssn"
    assert entity["text"] == "123-45-6789"
    assert (entity["start"], entity["end"]) == (5, 16)
    assert entity["confidence"] is None


# --- error taxonomy ---------------------------------------------------------


def test_value_error_from_engine_maps_to_service_error() -> None:
    with pytest.raises(ServiceError, match="bad option"):
        service.deidentify(_raising(ValueError("bad option")), "x", method="mask")


def test_backend_failure_does_not_leak_internal_message() -> None:
    with pytest.raises(ServiceError) as excinfo:
        service.extract(_raising(RuntimeError("model load exploded")), "x")
    message = str(excinfo.value)
    assert "exploded" not in message  # internal detail must not leak to the user
    assert "unavailable" in message.lower()


def test_batch_isolates_per_item_value_error() -> None:
    # A note that trips a ValueError is isolated as a failed item (ok=False) so the rest of
    # the batch still completes — it no longer aborts the whole batch.
    result = service.deidentify_batch(
        _raising(ValueError("bad option")), ["x", "y"], method="mask"
    )
    results = result["results"]
    assert [r["ok"] for r in results] == [False, False]
    assert all("bad option" in r["error"] for r in results)


def test_batch_backend_failure_aborts_and_does_not_leak() -> None:
    # A backend-load failure isn't note-specific (it fails every note identically), so it
    # aborts the whole batch via _run rather than spamming N failed rows — and never leaks.
    with pytest.raises(ServiceError) as excinfo:
        service.deidentify_batch(_raising(RuntimeError("kaboom")), ["x", "y"])
    assert "kaboom" not in str(excinfo.value)


def test_reidentify_error_maps_to_service_error() -> None:
    # reidentify is wrapped in _run like the other entrypoints, so it can't leak raw.
    with pytest.raises(ServiceError) as excinfo:
        service.reidentify(_raising(RuntimeError("boom")), "x", {"A": "B"})
    assert "boom" not in str(excinfo.value)


def test_unexpected_engine_error_maps_to_service_error() -> None:
    # An exception outside the ValueError/RuntimeError/OSError taxonomy must still be
    # caught and normalized, so a raw message (possible PHI) never reaches the UI.
    with pytest.raises(ServiceError) as excinfo:
        service.extract(_raising(KeyError("leak-me")), "x")
    message = str(excinfo.value)
    assert "leak-me" not in message
    assert "unexpectedly" in message.lower()


# --- Model-backed tests (real OpenMed engine; need --run-model) -------------


@pytest.mark.model
def test_service_extract_detects_real_pii(loader, note) -> None:
    result = service.extract(PIIEngine(loader=loader), note)
    found = {(e["label"], e["text"]) for e in result["entities"]}
    assert ("ssn", "123-45-6789") in found


@pytest.mark.model
def test_service_deidentify_masks_real_pii(loader, note) -> None:
    result = service.deidentify(PIIEngine(loader=loader), note, method="mask")
    text = result["deidentified_text"]
    assert "123-45-6789" not in text
    assert "john.doe@example.com" not in text
