"""Fast tests for OpenMed's pure-Python PII surface — no model load required."""

from __future__ import annotations

import typing

import pytest
from openmed import reidentify


def test_public_pii_api_is_importable() -> None:
    from openmed import deidentify, extract_pii, reidentify

    assert all(callable(fn) for fn in (extract_pii, deidentify, reidentify))


def test_known_deidentification_methods() -> None:
    # The canonical method set. test_validation.py::test_validation_deidmethod_matches_openmed
    # enforces that the engine/validation `DeidMethod` alias stays in sync with it.
    from openmed.core.pii import DeidentificationMethod

    assert set(typing.get_args(DeidentificationMethod)) == {
        "mask",
        "remove",
        "replace",
        "hash",
        "shift_dates",
        "format_preserve",  # added in openmed 1.7.0
    }


@pytest.mark.parametrize(
    ("deidentified", "mapping", "expected"),
    [
        ("Hi ALIAS_1.", {"ALIAS_1": "John Doe"}, "Hi John Doe."),
        (
            "ALIAS_1 lives in CITY_X",
            {"ALIAS_1": "John Doe", "CITY_X": "Springfield"},
            "John Doe lives in Springfield",
        ),
        ("nothing to restore", {}, "nothing to restore"),
    ],
)
def test_reidentify_substitutes_mapping(deidentified, mapping, expected) -> None:
    # reidentify() is a plain str.replace over {redacted: original} — no model.
    assert reidentify(deidentified, mapping) == expected


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"date_shift_days": 30}, "date_shift_days requires"),
        ({"method": "shift_dates", "shift_dates": False}, "conflicts with"),
    ],
)
def test_deidentify_rejects_invalid_options(kwargs, match) -> None:
    # Both options are validated before any model loads, so no --run-model needed.
    from openmed import deidentify

    with pytest.raises(ValueError, match=match):
        deidentify("Patient John Doe.", **kwargs)


@pytest.mark.xfail(
    reason="reidentify() applies str.replace per entry, so a key that is a prefix of "
    "another ('ALIAS_1' vs 'ALIAS_10') corrupts the longer one; correct restoration "
    "would need longest-key-first ordering. An XPASS means upstream fixed it.",
    raises=AssertionError,
    strict=True,
)
def test_reidentify_handles_overlapping_keys() -> None:
    # Currently yields "Ann and Ann0": "ALIAS_1" is replaced first and mangles "ALIAS_10".
    restored = reidentify("ALIAS_1 and ALIAS_10", {"ALIAS_1": "Ann", "ALIAS_10": "Bob"})
    assert restored == "Ann and Bob"
