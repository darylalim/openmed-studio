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

    def analyze(self, _text, **_):
        # UPPERCASE label is the NER sentinel a PII model would never emit.
        return [
            SimpleNamespace(
                label="DISEASE", text="[[STUB-NER]]", start=8, end=16, confidence=0.97
            )
        ]

    def reidentify(self, _deidentified_text, _mapping):
        return "[[STUB-RESTORED]]"


class _RaisingEngine(_StubEngine):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def extract(self, _text, **_):
        raise self._exc

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
    assert at.title[0].value == "OpenMed Studio"
    # Detect, Clinical NER, Single note, Batch, Anonymize, Re-identify
    assert len(at.tabs) == 6
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


def test_deid_controls_live_in_tabs_not_sidebar(monkeypatch):
    # #1: the de-identification Method/Advanced moved OUT of the sidebar into the Single
    # note + Batch tabs; the sidebar keeps only the engine readout + the global Language.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception
    # Sidebar holds the global Language and no de-identification Method picker.
    assert any(s.label == "Language" for s in at.sidebar.selectbox)
    assert not list(at.sidebar.segmented_control)
    # The Method picker now lives in the de-identifying tabs, keyed per tab.
    method_keys = {s.key for s in at.segmented_control}
    assert {"single_method", "batch_method"} <= method_keys


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


def test_single_note_forwards_replace_locale_to_engine(monkeypatch):
    # The Method + its replace-only knobs now live in the Single note tab (#1), not the
    # sidebar. "Replace locale" renders only when method=replace (conditional Advanced),
    # and flows through build_base_opts -> service -> engine.
    captured: dict = {}

    class _Capturing(_StubEngine):
        def deidentify(self, _text, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                deidentified_text="[[STUB-DEID-OUTPUT]]", pii_entities=[], mapping=None
            )

    _use_engine(monkeypatch, _Capturing())
    at = AppTest.from_file(APP).run(timeout=30)
    # Set method=replace and rerun so the conditional "Replace locale" knob renders.
    next(s for s in at.segmented_control if s.key == "single_method").set_value(
        "replace"
    ).run(timeout=30)
    next(t for t in at.text_input if t.key == "single_locale").set_value("pt_BR")
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert captured.get("locale") == "pt_BR"


def test_single_note_persists_mapping_to_session_state(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    # "Keep mapping" defaults on, so the result carries a mapping.
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert at.session_state["last_mapping"] == {"PERSON_1": "John"}
    assert at.session_state["last_deidentified"] == "[[STUB-DEID-OUTPUT]]"


def test_single_note_offers_mapping_reveal_when_kept(monkeypatch):
    # With Keep mapping on (default), the result offers a "Show re-identification key" button
    # (the mapping is revealed in a dialog now, not an always-open expander).
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert any(b.label == "Show re-identification key" for b in at.button)


def test_single_note_clears_stale_mapping_when_keep_off(monkeypatch):
    # Regression guard: a run whose result has no mapping must clear a previous one.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP)
    at.session_state["last_mapping"] = {"OLD_1": "secret"}
    at.run(timeout=30)
    next(t for t in at.toggle if t.key == "single_keepmap").set_value(False)
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


def test_detect_error_renders_message(monkeypatch):
    _use_engine(monkeypatch, _RaisingEngine(ValueError("detect boom")))
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note to scan", "Patient John Doe.")
    _click(at, "Detect")

    assert not at.exception
    assert any("detect boom" in e.value for e in at.error)
    assert not at.metric


def test_batch_isolates_failed_note_in_table(monkeypatch):
    # A model ValueError on a note no longer aborts the batch (#1 batch isolation): the note
    # is shown as a Failed row and the batch completes, with a warning summarizing failures.
    _use_engine(monkeypatch, _RaisingEngine(ValueError("batch boom")))
    at = AppTest.from_file(APP).run(timeout=30)
    _click(at, "De-identify all")

    assert not at.exception
    assert not at.error  # no whole-batch abort banner
    assert any("failed" in w.value.lower() for w in at.warning)
    # the per-note error is surfaced in the results table, not swallowed (the batch editor
    # also surfaces as a dataframe, so search across them rather than assuming an index)
    assert any("batch boom" in df.value.to_string() for df in at.dataframe)


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


# --- clinical NER ------------------------------------------------------------
def test_ner_renders_entities(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note to analyze", "Patient has diabetes today.")
    _click(at, "Analyze")

    assert not at.exception
    # metric==1 proves the stub ran (the entity-type preview also renders "DISEASE", so the
    # label is no longer a stub-only sentinel); the <mark> proves the entity was highlighted.
    assert any(m.label == "Entities found" and str(m.value) == "1" for m in at.metric)
    assert "<mark" in _html(at)


def test_ner_model_picker_lists_curated_domains(monkeypatch):
    # The domain picker is the curated per-domain catalog (Disease default).
    from openmed_studio import NER_MODELS

    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    picker = next(s for s in at.selectbox if s.label == "Entity domain")
    assert list(picker.options) == list(NER_MODELS)
    assert picker.value == "Disease"  # default = first (DEFAULT_NER_MODEL's domain)


def test_ner_tab_forwards_selected_domain_model(monkeypatch):
    # Picking a (non-default) domain must resolve to that domain's curated alias and
    # forward it to the engine — guards the `model_name = NER_MODELS[domain].alias`
    # resolution (a "silently always Disease" bug would slip past test_ner_renders_entities).
    from openmed_studio import NER_MODELS

    captured: dict = {}

    class _Capturing(_StubEngine):
        def analyze(self, _text, **kwargs):
            captured.update(kwargs)
            return []

    _use_engine(monkeypatch, _Capturing())
    at = AppTest.from_file(APP).run(timeout=30)
    # The domain picker lives outside the form, so set it (and rerun) before submitting.
    next(s for s in at.selectbox if s.label == "Entity domain").set_value(
        "Anatomy"
    ).run(timeout=30)
    _set_area(at, "Clinical note to analyze", "Liver and lung findings.")
    _click(at, "Analyze")

    assert not at.exception
    assert captured["model_name"] == NER_MODELS["Anatomy"].alias
    assert (
        NER_MODELS["Anatomy"].alias != NER_MODELS["Disease"].alias
    )  # the picked domain


def test_ner_confidence_defaults_to_model_recommendation(monkeypatch):
    # The NER slider seeds from the selected model's recommended_confidence (#3), not a
    # flat 0.5 — so the first result a user sees uses the model's own threshold.
    from openmed_studio import NER_MODELS

    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    slider = next(s for s in at.slider if str(s.key).startswith("ner_conf"))
    assert slider.value == NER_MODELS["Disease"].recommended_confidence  # 0.6, not 0.5


def test_ner_preview_shows_name_and_entity_types(monkeypatch):
    # The reactive preview surfaces the friendly model name + what it detects (#4), so the
    # user sees coverage before paying a 141-434MB download.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    captions = " ".join(c.value for c in at.caption)
    assert "DiseaseDetect" in captions  # friendly display_name
    assert "DISEASE" in captions and "CONDITION" in captions  # entity-type preview


def test_ner_medical_flags_broad_coverage(monkeypatch):
    # Medical is the 434M broad model with no declared entity types — the preview flags
    # both rather than presenting it as a peer of the 141M domains (#7).
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    next(s for s in at.selectbox if s.label == "Entity domain").set_value(
        "Medical"
    ).run(timeout=30)
    captions = " ".join(c.value for c in at.caption)
    assert "broad-coverage" in captions
    assert "not declared" in captions and "434M" in captions


def test_ner_tracks_analyzed_domains_for_load_hint(monkeypatch):
    # The per-domain download warning (#8) keys off a session-state set of analyzed
    # domains (since is_loaded can't tell which specific model is resident).
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note to analyze", "Patient has diabetes.")
    _click(at, "Analyze")

    assert not at.exception
    assert at.session_state["ner_analyzed_domains"] == {"Disease"}


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


# --- anonymize ---------------------------------------------------------------
def test_anonymize_renders_synthetic_output(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note to anonymize", "Patient John Doe.")
    _click(at, "Anonymize")

    assert not at.exception
    metrics = {m.label: str(m.value) for m in at.metric}
    assert metrics.get("Entities replaced") == "2"  # stub returns 2 pii_entities
    assert metrics.get("Deterministic") == "On"  # Deterministic toggle defaults on
    body = _html(at)
    assert "[[STUB-DEID-OUTPUT]]" in body  # synthetic surrogate text rendered
    assert "<mark" in body  # original highlighted


def test_anonymize_forwards_replace_method_locale_and_keep_mapping(monkeypatch):
    # The tab pins method=replace and forwards the in-tab locale + keep_mapping=True, so the
    # call is a reversible surrogate replacement (the Anonymize tab's "Locale" is distinct
    # from the Single/Batch tabs' "Replace locale").
    captured: dict = {}

    class _Capturing(_StubEngine):
        def deidentify(self, _text, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                deidentified_text="[[STUB-DEID-OUTPUT]]", pii_entities=[], mapping=None
            )

    _use_engine(monkeypatch, _Capturing())
    at = AppTest.from_file(APP).run(timeout=30)
    next(t for t in at.text_input if t.label == "Locale").set_value("pt_BR")
    _set_area(at, "Clinical note to anonymize", "Patient John Doe.")
    _click(at, "Anonymize")

    assert not at.exception
    assert captured.get("method") == "replace"
    assert captured.get("locale") == "pt_BR"
    assert captured.get("keep_mapping") is True


def test_anonymize_feeds_reidentify_handoff(monkeypatch):
    # Anonymize is intentionally NOT a fragment, so its submit triggers a full rerun that
    # re-runs the Re-identify fragment, prefilling it from session_state (replace +
    # keep_mapping is the reversible round trip the Anonymize tab centers on).
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note to anonymize", "Patient John Doe.")
    _click(at, "Anonymize")

    assert not at.exception
    assert at.session_state["last_deidentified"] == "[[STUB-DEID-OUTPUT]]"
    assert at.session_state["last_mapping"] == {"PERSON_1": "John"}
    reid = next(t for t in at.text_area if t.label == "De-identified text")
    assert (
        reid.value == "[[STUB-DEID-OUTPUT]]"
    )  # handed off across the fragment boundary
    mapping_area = next(t for t in at.text_area if t.label == "Mapping (JSON)")
    assert "PERSON_1" in (
        mapping_area.value or ""
    )  # the mapping crosses the boundary too


def test_empty_anonymize_warns_and_skips(monkeypatch):
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note to anonymize", "   ")  # whitespace only
    _click(at, "Anonymize")

    assert not at.exception
    assert any("Enter some text" in w.value for w in at.warning)
    assert not at.metric  # no result rendered


def test_anonymize_error_renders_message(monkeypatch):
    _use_engine(monkeypatch, _RaisingEngine(ValueError("anon boom")))
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note to anonymize", "Patient John Doe.")
    _click(at, "Anonymize")

    assert not at.exception
    assert any("anon boom" in e.value for e in at.error)
    assert not at.metric


def test_anonymize_forwards_consistent_and_seed(monkeypatch):
    # "Deterministic" is on by default; the in-tab seed forwards only then, so repeated
    # mentions resolve to one stable surrogate, reproducibly across runs.
    captured: dict = {}

    class _Capturing(_StubEngine):
        def deidentify(self, _text, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                deidentified_text="[[STUB-DEID-OUTPUT]]", pii_entities=[], mapping=None
            )

    _use_engine(monkeypatch, _Capturing())
    at = AppTest.from_file(APP).run(timeout=30)
    next(n for n in at.number_input if n.key == "anon_seed").set_value(7)
    _set_area(at, "Clinical note to anonymize", "Patient John Doe.")
    _click(at, "Anonymize")

    assert not at.exception
    assert captured.get("consistent") is True
    assert captured.get("seed") == 7
    # Blank Locale → the `if locale.strip()` branch omits it, so the service forwards
    # locale=None (the seam always passes req.locale), never an empty/real locale value.
    assert captured.get("locale") is None


def test_anonymize_omits_seed_when_not_deterministic(monkeypatch):
    # With "Deterministic" off, seed is omitted so openmed uses fresh per-call surrogates
    # (guards the `if consistent: opts["seed"] = ...` branch in _render_anonymize).
    captured: dict = {}

    class _Capturing(_StubEngine):
        def deidentify(self, _text, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                deidentified_text="[[STUB-DEID-OUTPUT]]", pii_entities=[], mapping=None
            )

    _use_engine(monkeypatch, _Capturing())
    at = AppTest.from_file(APP).run(timeout=30)
    next(t for t in at.toggle if t.key == "anon_consistent").set_value(False)
    _set_area(at, "Clinical note to anonymize", "Patient John Doe.")
    _click(at, "Anonymize")

    assert not at.exception
    assert captured.get("consistent") is False
    assert captured.get("seed") is None


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


# --- fragments / cross-tab handoff (#3) --------------------------------------
def test_single_to_reidentify_handoff_across_fragments(monkeypatch):
    # The Re-identify tab is an @st.fragment; the Single tab is not, so its form
    # submit triggers a full rerun that re-runs the Re-identify fragment, which
    # re-reads last_deidentified/last_mapping (shared via session_state, not widget
    # keys) and prefills its inputs.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert at.session_state["last_deidentified"] == "[[STUB-DEID-OUTPUT]]"
    reid = next(t for t in at.text_area if t.label == "De-identified text")
    assert reid.value == "[[STUB-DEID-OUTPUT]]"
    mapping_area = next(t for t in at.text_area if t.label == "Mapping (JSON)")
    assert "PERSON_1" in (mapping_area.value or "")  # mapping handed off as JSON


def test_no_duplicate_widget_keys_across_tabs(monkeypatch):
    # Fragmenting the tabs (#3) makes a duplicate key=... a likely regression;
    # Streamlit raises StreamlitDuplicateElementKey, surfaced here as at.exception.
    # Exercise each tab so every keyed widget mounts.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception
    _set_area(at, "Clinical note to scan", "Patient John Doe.")
    _click(at, "Detect")
    assert not at.exception
    _set_area(at, "Clinical note to analyze", "Patient John Doe.")
    _click(at, "Analyze")
    assert not at.exception
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")
    assert not at.exception
    _click(at, "De-identify all")
    assert not at.exception
    _set_area(at, "Clinical note to anonymize", "Patient John Doe.")
    _click(at, "Anonymize")
    assert not at.exception


# --- theme-agnostic highlighting (#2) ----------------------------------------
def test_highlight_marks_are_theme_agnostic(monkeypatch):
    # Marks use a translucent tint + color:inherit, so they render correctly on any
    # theme with no runtime theme detection (no _is_dark / st.context.theme read).
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note to scan", "Patient John Doe.")
    _click(at, "Detect")

    assert not at.exception
    body = _html(at)
    assert "<mark" in body
    assert "color:inherit" in body
    assert "rgba(" in body


# --- review follow-ups: persistence, handoff drift, conditional Advanced, dialog ----
def test_handoff_not_overwritten_on_non_submit_rerun(monkeypatch):
    # Security-relevant invariant (H1): _set_handoff fires ONLY on submit, never on the
    # persisted-panel re-render — so a post-submit rerun (Show key / Download) can't drift
    # last_deidentified/last_mapping and resurface a stale mapping.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")
    assert at.session_state["last_mapping"] == {"PERSON_1": "John"}
    # Overwrite the handoff with sentinels, then force a NON-submit rerun (Show key).
    at.session_state["last_mapping"] = {"SENTINEL": "x"}
    at.session_state["last_deidentified"] = "SENTINEL-TEXT"
    next(b for b in at.button if b.label == "Show re-identification key").click().run(
        timeout=30
    )
    assert not at.exception
    assert at.session_state["last_mapping"] == {"SENTINEL": "x"}
    assert at.session_state["last_deidentified"] == "SENTINEL-TEXT"


def test_single_note_panel_persists_across_reruns(monkeypatch):
    # M3: the panel renders from session_state, so a post-submit rerun (Show key) must NOT
    # blank it — reverting the persistence refactor would fail here.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")
    assert any(m.label == "Entities found" for m in at.metric)
    next(b for b in at.button if b.label == "Show re-identification key").click().run(
        timeout=30
    )
    assert not at.exception
    assert any(m.label == "Entities found" for m in at.metric)
    assert "[[STUB-DEID-OUTPUT]]" in _html(at)


def test_single_note_keeps_panel_on_empty_resubmit(monkeypatch):
    # L3: after a good result, an empty re-submit warns but does NOT blank the persisted panel.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")
    assert any(m.label == "Entities found" for m in at.metric)
    _set_area(at, "Clinical note", "   ")  # whitespace only
    _click(at, "De-identify")
    assert not at.exception
    assert any("Enter some text" in w.value for w in at.warning)
    assert any(m.label == "Entities found" for m in at.metric)  # panel survived


def test_batch_panel_persists_across_reruns(monkeypatch):
    # M1: batch results are persisted, so a post-submit fragment rerun (tweaking an in-tab
    # control) no longer blanks the table.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _click(at, "De-identify all")
    assert any(m.label == "Notes de-identified" for m in at.metric)
    next(s for s in at.segmented_control if s.key == "batch_method").set_value(
        "replace"
    ).run(timeout=30)
    assert not at.exception
    assert any(m.label == "Notes de-identified" for m in at.metric)


def test_single_note_forwards_shift_dates_controls(monkeypatch):
    # M2: the shift_dates branch of the conditional Advanced renders date_shift_days/keep_year
    # and forwards them to the engine (the replace branch is covered separately).
    captured: dict = {}

    class _Capturing(_StubEngine):
        def deidentify(self, _text, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                deidentified_text="[[STUB-DEID-OUTPUT]]", pii_entities=[], mapping=None
            )

    _use_engine(monkeypatch, _Capturing())
    at = AppTest.from_file(APP).run(timeout=30)
    next(s for s in at.segmented_control if s.key == "single_method").set_value(
        "shift_dates"
    ).run(timeout=30)
    next(n for n in at.number_input if n.key == "single_shift").set_value(30)
    _set_area(at, "Clinical note", "Seen on 04/15/2024.")
    _click(at, "De-identify")

    assert not at.exception
    assert captured.get("method") == "shift_dates"
    assert captured.get("date_shift_days") == 30
    assert captured.get("keep_year") is True


def test_advanced_knobs_absent_under_default_mask(monkeypatch):
    # M2 (negative): under the default mask method, neither the replace locale nor the
    # shift_dates inputs render — the whole point of the conditional Advanced.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    keys = {w.key for w in at.text_input} | {n.key for n in at.number_input}
    assert "single_locale" not in keys
    assert "single_shift" not in keys


def test_mapping_reveal_absent_when_keep_mapping_off(monkeypatch):
    # L4: no mapping → no "Show re-identification key" button (the reveal guard).
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    next(t for t in at.toggle if t.key == "single_keepmap").set_value(False)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")

    assert not at.exception
    assert not any(b.label == "Show re-identification key" for b in at.button)


def test_mapping_dialog_renders_mapping(monkeypatch):
    # L4: clicking the reveal button opens the dialog and renders the mapping JSON.
    _use_engine(monkeypatch, _StubEngine())
    at = AppTest.from_file(APP).run(timeout=30)
    _set_area(at, "Clinical note", "Patient John Doe.")
    _click(at, "De-identify")
    next(b for b in at.button if b.label == "Show re-identification key").click().run(
        timeout=30
    )

    assert not at.exception
    assert any("PERSON_1" in str(getattr(j, "value", "")) for j in at.json)
