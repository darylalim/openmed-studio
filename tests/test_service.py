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


def _stub() -> PIIEngine:
    # The stub is structural, not a real PIIEngine; cast to satisfy the typed seam.
    return cast("PIIEngine", _StubEngine())


def _raising(exc: Exception) -> PIIEngine:
    return cast("PIIEngine", _RaisingEngine(exc))


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


def test_deidentify_batch_returns_per_item_results() -> None:
    result = service.deidentify_batch(
        _stub(), ["Patient John.", "Patient Jane."], method="mask"
    )
    results = result["results"]
    assert len(results) == 2
    assert all(r["deidentified_text"] == "[first_name] A. Doe" for r in results)


def test_reidentify_restores() -> None:
    result = service.reidentify(_stub(), "Hi [first_name].", {"[first_name]": "John"})
    assert result["text"] == "Hi John."


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
