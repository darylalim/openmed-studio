"""Tests for the framework-free PIIEngine (openmed_studio.engine).

The fast tests verify the lazy-loading contract without touching a model; the
``@pytest.mark.model`` tests drive the real OpenMed model and are skipped unless
``--run-model`` is passed (reusing the session-scoped ``loader`` fixture).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from openmed_studio import DEFAULT_PII_MODEL, PIIEngine
from openmed_studio.engine import _is_date_label, shift_date_text


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


# --- Engine-side shift_dates: pure helpers (no model) -----------------------


@pytest.mark.parametrize(
    "label",
    ["date", "DATE", "date_of_birth", "DATEOFBIRTH", "date_time", "dob", "birthdate"],
)
def test_is_date_label_recognizes_date_taxonomies(label) -> None:
    assert _is_date_label(label) is True


@pytest.mark.parametrize(
    "label", ["ssn", "first_name", "email", "address", "update", "candidate"]
)
def test_is_date_label_rejects_non_dates(label) -> None:
    # Exact (letters-only) match, so 'update'/'candidate' aren't treated as dates.
    assert _is_date_label(label) is False


def test_shift_date_text_shifts_and_preserves_format() -> None:
    # One assertion per layout in _DATE_FORMATS, so every supported format round-trips.
    assert shift_date_text("03/22/2024", 180) == "09/18/2024"  # US slashes, 4-digit
    assert shift_date_text("03/22/24", 180) == "09/18/24"  # US slashes, 2-digit year
    assert shift_date_text("03-22-2024", 180) == "09-18-2024"  # US dashes
    assert shift_date_text("03.22.2024", 180) == "09.18.2024"  # US dots
    assert shift_date_text("2024-03-22", 180) == "2024-09-18"  # ISO preserved


def test_shift_date_text_keep_year_pins_year() -> None:
    # +90 days crosses into 2024; keep_year pins the year back to 2023.
    assert shift_date_text("11/05/2023", 90, keep_year=True) == "02/03/2023"
    assert shift_date_text("11/05/2023", 90, keep_year=False) == "02/03/2024"


def test_shift_date_text_keep_year_clamps_leap_day() -> None:
    # 12/01/2023 + 90 days lands on 02/29/2024; pinning back to the non-leap
    # original year 2023 has no Feb 29, so the year-pin clamps the day to the 28th.
    assert shift_date_text("12/01/2023", 90, keep_year=True) == "02/28/2023"
    assert shift_date_text("12/01/2023", 90, keep_year=False) == "02/29/2024"


def test_shift_date_text_returns_none_for_unparseable() -> None:
    assert shift_date_text("not a date", 30) is None
    assert shift_date_text("spring", 30) is None


def test_shift_date_text_matches_datetime_arithmetic() -> None:
    for days in (-400, -1, 0, 1, 45, 365):
        expected = (datetime(2021, 6, 14) + timedelta(days=days)).strftime("%m/%d/%Y")
        assert shift_date_text("06/14/2021", days, keep_year=False) == expected


# --- Engine-side shift_dates: orchestration (stubbed extract, no model) ------


def _engine_with_entities(
    monkeypatch: pytest.MonkeyPatch, entities: list[SimpleNamespace]
) -> PIIEngine:
    """A PIIEngine whose extract() returns canned entities (no model load)."""
    engine = PIIEngine()
    monkeypatch.setattr(engine, "extract", lambda *_a, **_k: entities)
    return engine


_SHIFT_NOTE = "Visit 03/22/2024; SSN 123-45-6789."
_SHIFT_ENTITIES = [
    SimpleNamespace(label="date", text="03/22/2024", start=6, end=16, confidence=0.9),
    SimpleNamespace(label="ssn", text="123-45-6789", start=22, end=33, confidence=0.9),
]


def test_shift_dates_shifts_dates_and_masks_other_pii(monkeypatch) -> None:
    engine = _engine_with_entities(monkeypatch, _SHIFT_ENTITIES)
    result = engine.deidentify(_SHIFT_NOTE, method="shift_dates", date_shift_days=10)
    assert result.deidentified_text == "Visit 04/01/2024; SSN [ssn]."
    assert result.mapping is None
    assert result.pii_entities == _SHIFT_ENTITIES


def test_shift_dates_builds_mapping_when_requested(monkeypatch) -> None:
    engine = _engine_with_entities(monkeypatch, _SHIFT_ENTITIES)
    result = engine.deidentify(
        _SHIFT_NOTE, method="shift_dates", date_shift_days=10, keep_mapping=True
    )
    assert result.mapping == {"04/01/2024": "03/22/2024", "[ssn]": "123-45-6789"}


def test_shift_dates_masks_unparseable_date_entity(monkeypatch) -> None:
    entities = [
        SimpleNamespace(label="date", text="spring", start=8, end=14, confidence=0.5)
    ]
    engine = _engine_with_entities(monkeypatch, entities)
    result = engine.deidentify(
        "Born in spring.", method="shift_dates", date_shift_days=10
    )
    assert result.deidentified_text == "Born in [date]."


def test_shift_dates_drops_overlapping_spans(monkeypatch) -> None:
    # A fragmented/overlapping span (privacy-filter-style) must not corrupt offsets.
    entities = [
        SimpleNamespace(
            label="date", text="03/22/2024", start=6, end=16, confidence=0.9
        ),
        SimpleNamespace(label="date", text="2024", start=12, end=16, confidence=0.4),
    ]
    engine = _engine_with_entities(monkeypatch, entities)
    result = engine.deidentify(_SHIFT_NOTE, method="shift_dates", date_shift_days=10)
    assert result.deidentified_text == "Visit 04/01/2024; SSN 123-45-6789."


def test_shift_dates_uses_random_offset_when_unspecified(monkeypatch) -> None:
    # date_shift_days=None -> one random offset in [-365, 365] applied to *every*
    # date, so intervals between dates within the document are preserved.
    import openmed_studio.engine as engine_module

    calls: dict[str, tuple[int, int]] = {}

    def fake_randint(low: int, high: int) -> int:
        calls["args"] = (low, high)
        return 7

    monkeypatch.setattr(engine_module.random, "randint", fake_randint)

    note = "A 01/01/2020 and B 01/11/2020 end."  # the two dates are 10 days apart
    entities = [
        SimpleNamespace(
            label="date", text="01/01/2020", start=2, end=12, confidence=0.9
        ),
        SimpleNamespace(
            label="date", text="01/11/2020", start=19, end=29, confidence=0.9
        ),
    ]
    engine = _engine_with_entities(monkeypatch, entities)
    result = engine.deidentify(note, method="shift_dates", date_shift_days=None)

    assert calls["args"] == (-365, 365)  # randomized across the documented range
    # both dates shifted by the same +7, so they stay 10 days apart
    assert result.deidentified_text == "A 01/08/2020 and B 01/18/2020 end."


def test_shift_dates_recognizes_entity_type_shape(monkeypatch) -> None:
    # privacy-filter-style entities expose UPPERCASE `entity_type` (not `label`);
    # the engine's getattr fallback + label normalization must still shift them.
    entities = [
        SimpleNamespace(
            entity_type="DATE", text="03/22/2024", start=5, end=15, confidence=0.9
        )
    ]
    engine = _engine_with_entities(monkeypatch, entities)
    result = engine.deidentify(
        "Seen 03/22/2024 today.", method="shift_dates", date_shift_days=10
    )
    assert result.deidentified_text == "Seen 04/01/2024 today."


def test_shift_dates_keep_year_threads_through_orchestration(monkeypatch) -> None:
    # keep_year must reach shift_date_text via deidentify -> _shift_dates, not only be
    # unit-tested on the helper. 11/05/2023 + 90 days crosses into 2024.
    note = "Follow-up 11/05/2023 scheduled."
    entities = [
        SimpleNamespace(
            label="date", text="11/05/2023", start=10, end=20, confidence=0.9
        )
    ]
    engine = _engine_with_entities(monkeypatch, entities)
    pinned = engine.deidentify(
        note, method="shift_dates", date_shift_days=90, keep_year=True
    )
    moved = engine.deidentify(
        note, method="shift_dates", date_shift_days=90, keep_year=False
    )
    assert pinned.deidentified_text == "Follow-up 02/03/2023 scheduled."  # year pinned
    assert moved.deidentified_text == "Follow-up 02/03/2024 scheduled."  # year moved


def test_shift_dates_skips_mapping_for_unchanged_date(monkeypatch) -> None:
    # date_shift_days=0 leaves the date identical, so the `replacement != original`
    # guard must keep it out of the mapping; masked non-date PII is still recorded.
    engine = _engine_with_entities(monkeypatch, _SHIFT_ENTITIES)
    result = engine.deidentify(
        _SHIFT_NOTE, method="shift_dates", date_shift_days=0, keep_mapping=True
    )
    assert result.deidentified_text == "Visit 03/22/2024; SSN [ssn]."
    assert result.mapping == {"[ssn]": "123-45-6789"}  # unchanged date excluded


def test_shift_dates_drops_empty_span(monkeypatch) -> None:
    # A zero-width span (end == start) must be dropped (text unchanged) while still
    # being reported in pii_entities.
    entities = [SimpleNamespace(label="date", text="", start=6, end=6, confidence=0.5)]
    engine = _engine_with_entities(monkeypatch, entities)
    result = engine.deidentify(
        "Plain note text.", method="shift_dates", date_shift_days=10
    )
    assert result.deidentified_text == "Plain note text."
    assert len(result.pii_entities) == 1


def test_shift_dates_same_start_overlap_keeps_longer_span(monkeypatch) -> None:
    # The sort key (start, -end) keeps the longer span when two share a start: the full
    # date (end=16) wins and shifts; the shorter "03/22" (end=11) wouldn't even parse.
    entities = [
        SimpleNamespace(
            label="date", text="03/22/2024", start=6, end=16, confidence=0.9
        ),
        SimpleNamespace(label="date", text="03/22", start=6, end=11, confidence=0.4),
    ]
    engine = _engine_with_entities(monkeypatch, entities)
    result = engine.deidentify(_SHIFT_NOTE, method="shift_dates", date_shift_days=10)
    assert result.deidentified_text == "Visit 04/01/2024; SSN 123-45-6789."


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


@pytest.mark.model
def test_engine_shift_dates_actually_shifts_with_default_model(loader, note) -> None:
    # The whole point: with the default model (lowercase 'date' labels) openmed's
    # shift_dates is a no-op, but the engine-side path shifts dates for real.
    engine = PIIEngine(loader=loader)
    out = engine.deidentify(
        note, method="shift_dates", date_shift_days=180
    ).deidentified_text

    assert "03/22/2024" not in out  # the appointment date was moved
    assert "[date]" not in out and "[DATE]" not in out  # NOT masked (the gotcha)
    assert shift_date_text("03/22/2024", 180) in out  # the real shifted date is present
    assert "123-45-6789" not in out  # non-date PII is still masked
