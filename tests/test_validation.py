"""Tests for the input guarantees that survive the move off HTTP.

The old FastAPI service validated requests with Pydantic before calling the
engine; ``openmed_studio.service`` now does the same in-process, raising
``ServiceError`` on rejection (validation runs before the stub engine is reached).
This pins the text/batch/mapping caps, the value/enum/format checks, the
``OPENMED_STUDIO_MAX_TEXT_LENGTH`` knob, the ``DeidMethod``↔openmed sync, and that
rejection messages never echo the offending input (possible PHI).
"""

from __future__ import annotations

import typing
from types import SimpleNamespace
from typing import cast

import pytest

from openmed_studio import PIIEngine, service, validation
from openmed_studio.service import ServiceError


class _StubEngine:
    """Returns canned results; only reached when validation passes."""

    def extract(self, _text, **_):
        return []

    def deidentify(self, _text, **_):
        return SimpleNamespace(deidentified_text="ok", pii_entities=[], mapping=None)

    def reidentify(self, deidentified_text, _mapping):
        return deidentified_text


ENGINE = cast("PIIEngine", _StubEngine())


# --- rejections -------------------------------------------------------------


def test_rejects_unknown_field() -> None:
    with pytest.raises(ServiceError):
        service.extract(ENGINE, "x", bogus=1)


def test_rejects_empty_text() -> None:
    with pytest.raises(ServiceError):
        service.extract(ENGINE, "")


def test_rejects_whitespace_only_text() -> None:
    with pytest.raises(ServiceError):
        service.extract(ENGINE, "   ")


def test_rejects_oversize_text() -> None:
    with pytest.raises(ServiceError):
        service.extract(ENGINE, "x" * 50_001)


def test_rejects_bad_method() -> None:
    with pytest.raises(ServiceError):
        service.deidentify(ENGINE, "x", method="encrypt")


def test_rejects_bad_lang() -> None:
    with pytest.raises(ServiceError):
        service.extract(ENGINE, "x", lang="zz")


def test_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ServiceError):
        service.extract(ENGINE, "x", confidence_threshold=1.5)


def test_rejects_negative_confidence() -> None:
    # confidence_threshold is a two-sided range (ge=0.0, le=1.0); cover the lower bound.
    with pytest.raises(ServiceError):
        service.extract(ENGINE, "x", confidence_threshold=-0.1)


def test_rejects_invalid_model_name() -> None:
    with pytest.raises(ServiceError):
        service.extract(ENGINE, "x", model_name="../etc/passwd")


def test_batch_rejects_empty_items() -> None:
    with pytest.raises(ServiceError):
        service.deidentify_batch(ENGINE, [])


def test_batch_rejects_too_many_items() -> None:
    with pytest.raises(ServiceError):
        service.deidentify_batch(ENGINE, ["x"] * 101)


def test_reidentify_rejects_oversize_mapping() -> None:
    big = {str(i): "y" for i in range(validation.MAX_MAPPING_ENTRIES + 1)}
    with pytest.raises(ServiceError):
        service.reidentify(ENGINE, "x", big)


# --- acceptances (validation passes, reaches the stub engine) ---------------


def test_accepts_lang_and_model_name() -> None:
    assert service.extract(ENGINE, "x", lang="fr", model_name="OpenMed/Some-Model") == {
        "entities": []
    }


def test_accepts_date_controls() -> None:
    result = service.deidentify(ENGINE, "x", method="shift_dates", date_shift_days=180)
    assert result["method"] == "shift_dates"


# --- PHI safety + the text cap ----------------------------------------------


def test_validation_error_does_not_echo_input() -> None:
    # The offending text (possible PHI) must never appear in the user-facing message;
    # only the field location + constraint message are surfaced.
    secret = "SECRET-SSN-123-45-6789-"
    text = secret * 3000  # well over the 50k cap → a length validation error
    with pytest.raises(ServiceError) as excinfo:
        service.deidentify(ENGINE, text)
    assert secret not in str(excinfo.value)


def test_max_text_chars_env_override(monkeypatch) -> None:
    # The cap is read from OPENMED_STUDIO_MAX_TEXT_LENGTH; invalid/non-positive/unset
    # values fall back to the 50k default so a typo can't silently disable the guard.
    monkeypatch.setenv("OPENMED_STUDIO_MAX_TEXT_LENGTH", "1234")
    assert validation._max_text_chars() == 1234
    monkeypatch.setenv("OPENMED_STUDIO_MAX_TEXT_LENGTH", "not-a-number")
    assert validation._max_text_chars() == 50_000
    monkeypatch.setenv("OPENMED_STUDIO_MAX_TEXT_LENGTH", "0")
    assert validation._max_text_chars() == 50_000
    monkeypatch.delenv("OPENMED_STUDIO_MAX_TEXT_LENGTH", raising=False)
    assert validation._max_text_chars() == 50_000


def test_validation_deidmethod_matches_openmed() -> None:
    # Keep validation's method enum in sync with openmed's canonical set (no model load).
    from openmed.core.pii import DeidentificationMethod

    assert set(typing.get_args(validation.DeidMethod)) == set(
        typing.get_args(DeidentificationMethod)
    )
