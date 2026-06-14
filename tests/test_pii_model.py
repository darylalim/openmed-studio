"""Integration tests that load the OpenMed PII model.

Every test here is marked ``@pytest.mark.model`` (via ``pytestmark``), so they are
skipped unless you pass ``--run-model``. They reuse the session-scoped ``loader``
fixture so the model is initialized only once.
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.model


def _entities(result):
    """extract_pii may return a list or an object exposing the entities."""
    for attr in ("entities", "pii_entities"):
        if hasattr(result, attr):
            return getattr(result, attr)
    return result


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("first_name", "John"),
        ("last_name", "Doe"),
        ("ssn", "123-45-6789"),
        ("email", "john.doe@example.com"),
        ("phone_number", "(415) 555-0137"),
    ],
)
def test_extract_pii_detects_expected_entities(loader, note, label, text) -> None:
    from openmed import extract_pii

    found = {(e.label, e.text) for e in _entities(extract_pii(note, loader=loader))}
    assert (label, text) in found


def test_mask_removes_raw_identifiers(loader, note) -> None:
    from openmed import deidentify

    masked = deidentify(note, method="mask", loader=loader).deidentified_text
    for secret in ("John", "Doe", "123-45-6789", "john.doe@example.com"):
        assert secret not in masked
    assert "[ssn]" in masked  # replaced by a typed placeholder


def test_replace_is_deterministic_and_changes_text(loader, note) -> None:
    from openmed import deidentify

    first = deidentify(
        note, method="replace", consistent=True, seed=42, loader=loader
    ).deidentified_text
    second = deidentify(
        note, method="replace", consistent=True, seed=42, loader=loader
    ).deidentified_text
    assert first == second  # deterministic with a fixed seed
    assert first != note  # but the PII has been replaced
    assert "123-45-6789" not in first


def test_remove_deletes_identifiers(loader, note) -> None:
    from openmed import deidentify

    removed = deidentify(note, method="remove", loader=loader).deidentified_text
    for secret in ("John", "Doe", "123-45-6789", "john.doe@example.com"):
        assert secret not in removed
    assert "[ssn]" not in removed  # remove deletes spans; it leaves no placeholders


def test_hash_is_stable_and_typed(loader, note) -> None:
    from openmed import deidentify

    first = deidentify(note, method="hash", loader=loader).deidentified_text
    second = deidentify(note, method="hash", loader=loader).deidentified_text
    assert first == second  # deterministic (no seed) → enables cross-document linking
    assert "123-45-6789" not in first
    assert re.search(r"ssn_[0-9a-f]{8}", first)  # typed digest shape, e.g. ssn_01a54629


def test_round_trip_reidentify_restores_original(loader, note) -> None:
    from openmed import deidentify, reidentify

    res = deidentify(
        note,
        method="replace",
        consistent=True,
        seed=7,
        keep_mapping=True,
        loader=loader,
    )
    # Guard against a no-op (zero entities → empty mapping → trivially-true round-trip):
    assert res.deidentified_text != note  # PII was actually replaced
    assert res.mapping  # non-empty mapping (also narrows Optional → dict below)
    assert "123-45-6789" not in res.deidentified_text
    assert reidentify(res.deidentified_text, res.mapping) == note


@pytest.mark.xfail(
    reason="OpenMed 1.5.5 shift_dates matches the literal label 'DATE', but the "
    "default model emits lowercase 'date', so dates are masked, not shifted "
    "(openmed/core/pii.py:905). An XPASS here means upstream fixed it — delete this xfail.",
    raises=AssertionError,  # narrow to the expected failure; surface any other error for real
    strict=True,
)
def test_shift_dates_actually_shifts_dates(loader, note) -> None:
    from openmed import deidentify

    masked = deidentify(note, method="mask", loader=loader).deidentified_text
    shifted = deidentify(
        note, method="shift_dates", date_shift_days=180, loader=loader
    ).deidentified_text
    # If shifting worked, the shifted output would differ from plain masking.
    assert shifted != masked
