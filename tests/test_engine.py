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


def test_engine_default_backend_builds_eager_config(monkeypatch) -> None:
    # backend=None still auto-detects the backend (config backend stays None), but the
    # loader is always built with OpenMedConfig(torch_attention_backend="eager") so the
    # DeBERTa-v2 models load on transformers >=5.13 (which rejects the SDPA openmed's
    # "auto" backend requests). See PIIEngine.loader for the full rationale.
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

    engine = PIIEngine()
    assert isinstance(engine.loader, _FakeLoader)
    assert captured["config_kwargs"] == {
        "backend": None,
        "torch_attention_backend": "eager",
    }
    assert isinstance(captured["loader_config"], _FakeConfig)


def test_engine_backend_forwarded_via_openmedconfig(monkeypatch) -> None:
    # backend="mlx" reaches ModelLoader as OpenMedConfig(backend="mlx"), alongside the
    # eager attention pin every loader gets.
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
    assert captured["config_kwargs"] == {
        "backend": "mlx",
        "torch_attention_backend": "eager",
    }
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
    # The 1.6.0 safety sweep is wired through explicitly (on by default), and `audit` is
    # never forwarded — passing audit=True flips deidentify's return to AuditReport, which
    # service._deidentify_dict (reads .deidentified_text/.pii_entities) cannot consume.
    assert captured["use_safety_sweep"] is True
    # Smart merging is forwarded by deidentify too (default on), matching extract().
    assert captured["use_smart_merging"] is True
    assert "audit" not in captured
    # model_name is unset here, so _model_kwargs OMITS it (openmed's default is a literal
    # model string, not None) — forwarding model_name=None would make openmed load a model
    # literally named None. Pin the omission so a refactor to unconditional forwarding fails
    # in the fast suite (this shared _model_kwargs path also backs extract).
    assert "model_name" not in captured


def test_deidentify_forwards_locale_to_openmed(monkeypatch) -> None:
    # The `replace` surrogate locale must reach openmed.deidentify unchanged (the
    # validation->engine hop is pinned in test_service.py; this pins engine->openmed).
    import openmed

    captured: dict[str, object] = {}

    def fake_deidentify(_text, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(deidentified_text="ok", pii_entities=[], mapping=None)

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    engine.deidentify("x", method="replace", locale="pt_BR")
    assert captured["locale"] == "pt_BR"


def test_deidentify_forwards_every_openmed_param_or_allowlists_it(monkeypatch) -> None:
    """Drift guard: every ``openmed.deidentify`` parameter is either forwarded by the
    engine or on an explicit, documented exclusion list.

    ``PIIEngine.deidentify`` hand-lists the kwargs it threads into
    ``openmed.deidentify`` (and ``validation._DeidentifyOptions`` /
    ``service._deidentify_call`` mirror that list). Nothing pins that hand-list to
    openmed's real signature, so a parameter openmed *adds* — or one the engine
    silently stops forwarding — would drift unnoticed. This captures the kwargs the
    engine actually passes and asserts they cover openmed's signature, minus the
    parameters we deliberately don't forward (each justified below). It is the
    parameter-set analogue of ``test_validation_deidmethod_matches_openmed``.
    """
    import inspect

    import openmed

    # The real signature must be read *before* the monkeypatch below replaces
    # openmed.deidentify with the capturing stub (whose signature is just (text, **kw)).
    openmed_params = set(inspect.signature(openmed.deidentify).parameters)

    # openmed.deidentify params the engine intentionally does not forward. Each must
    # stay justified: starting to forward one (or openmed dropping one) must update
    # this set, which the assertions below enforce.
    intentionally_not_forwarded = {
        # Permanent exclusions — forwarding these would break the engine's contract:
        "audit",  # flips the return to AuditReport, which service._deidentify_dict
        # (reads .deidentified_text/.pii_entities) cannot consume.
        "config",  # the engine owns model loading via a shared ModelLoader threaded
        # as loader=; config is openmed's alternative construction path and bypasses it.
        # Not yet wired into the app's request models — listed so the guard stays green
        # until each is consciously exposed (then move it out of this set):
        "shift_dates",  # legacy bool toggle, distinct from method="shift_dates"
        "normalize_accents",
        "policy",
        "calibration_thresholds_path",
        # openmed 1.7.0 additions — advanced date-shift / surrogate / recognizer plumbing
        # and result caching the engine deliberately doesn't thread (yet):
        "patient_key",  # deterministic per-patient date-shift key
        "date_shift_max_days",  # bounds the random date shift
        "date_shift_secret",  # HMAC secret for keyed date shifting
        "surrogate_vault",  # external surrogate-consistency store for replace
        "custom_recognizer",  # caller-supplied entity recognizer
        "cache_results",  # openmed-side result memoization
        "max_cache_entries",  # cache size bound (only meaningful with cache_results)
    }

    captured: dict[str, object] = {}

    def fake_deidentify(text, **kwargs):
        captured.update(kwargs)
        captured["text"] = text
        return SimpleNamespace(deidentified_text="ok", pii_entities=[], mapping=None)

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    # A non-None loader short-circuits the lazy loader, so no model is built.
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    # Pass model_name so the optional model_name kwarg is actually threaded through
    # (_model_kwargs only adds it when set); every other forwarded kwarg is unconditional.
    engine.deidentify("x", model_name="OpenMed/Some-Model")
    forwarded = set(captured)  # includes "text", which is captured positionally

    # The guard: nothing openmed accepts is left unaccounted for.
    uncovered = openmed_params - forwarded - intentionally_not_forwarded
    assert not uncovered, (
        "openmed.deidentify params neither forwarded nor allowlisted "
        f"(forward them in PIIEngine.deidentify or justify them in the exclusion "
        f"set): {sorted(uncovered)}"
    )
    # The exclusion list can't go stale: every entry must still be a real openmed
    # param, and none may also be forwarded (a contradiction once one gets wired).
    assert intentionally_not_forwarded <= openmed_params, (
        "stale exclusion(s) no longer in openmed.deidentify: "
        f"{sorted(intentionally_not_forwarded - openmed_params)}"
    )
    assert not (forwarded & intentionally_not_forwarded), (
        "param is both forwarded and allowlisted — drop it from the exclusion set: "
        f"{sorted(forwarded & intentionally_not_forwarded)}"
    )


def test_reidentify_orders_overlapping_keys_longest_first() -> None:
    # The engine reorders the mapping longest-key-first before delegating, so a key that
    # is a substring of another (ALIAS_1 vs ALIAS_10) restores correctly despite openmed's
    # per-entry str.replace. (The raw-openmed limitation stays pinned in test_pii_pure.py.)
    restored = PIIEngine.reidentify(
        "ALIAS_1 and ALIAS_10",
        {"ALIAS_1": "Ann", "ALIAS_10": "Bob"},
    )
    assert restored == "Ann and Bob"


def test_reidentify_does_not_re_substitute_a_value_containing_another_key() -> None:
    # Single-pass restoration: a replacement value that contains another key is not
    # re-scanned, so it can't be clobbered. (Sequential str.replace would corrupt the
    # "X2" inside the restored "see X2" into "Bob".)
    restored = PIIEngine.reidentify(
        "X1 and X2",
        {"X1": "see X2", "X2": "Bob"},
    )
    assert restored == "see X2 and Bob"


# --- analyze (clinical NER) delegation (no model) ---------------------------


def test_analyze_delegates_to_openmed(monkeypatch) -> None:
    # engine.analyze wraps openmed.analyze_text the way extract wraps extract_pii:
    # forward model_name/confidence/aggregation/group_entities/output_format='dict'/loader,
    # and unwrap analyze_text's PredictionResult (an OBJECT with .entities, NOT a bare list
    # — _entities reads .entities rather than iterating the object).
    import openmed

    captured: dict[str, object] = {}

    def fake_analyze_text(text, **kwargs):
        captured.update(kwargs)
        captured["text"] = text
        return SimpleNamespace(
            entities=[
                SimpleNamespace(
                    label="DISEASE", text="diabetes", start=0, end=8, confidence=0.97
                )
            ]
        )

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze_text)
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    entities = engine.analyze(
        "diabetes today",
        model_name="disease_detection_superclinical_141m",
        confidence_threshold=0.6,
    )

    assert captured["text"] == "diabetes today"
    assert captured["model_name"] == "disease_detection_superclinical_141m"
    assert captured["confidence_threshold"] == 0.6
    assert captured["aggregation_strategy"] == "simple"
    assert captured["group_entities"] is False
    assert captured["output_format"] == "dict"  # the object-not-dict path
    assert captured["loader"] is engine.loader  # shared loader threaded through
    assert "lang" not in captured  # analyze_text has no lang param
    assert [e.label for e in entities] == ["DISEASE"]  # PredictionResult unwrapped


def test_analyze_forwards_every_openmed_param_or_allowlists_it(monkeypatch) -> None:
    """Drift guard: every named ``openmed.analyze_text`` parameter is either forwarded by
    ``PIIEngine.analyze`` or on an explicit exclusion list — the NER analogue of
    ``test_deidentify_forwards_every_openmed_param_or_allowlists_it``.

    This matters more than the deidentify case: ``analyze_text`` declares
    ``**pipeline_kwargs``, so a renamed/removed forwarded param — notably
    ``output_format="dict"``, which the ``_entities`` unwrap depends on — would NOT raise.
    It would be silently swallowed into ``pipeline_kwargs``, openmed would use its default,
    and analyze would return wrong/empty results that only a ``--run-model`` test (skipped
    in CI) could catch. Pinning the forwarded set to the real signature closes that gap.
    """
    import inspect

    import openmed

    # Read the real signature BEFORE the monkeypatch swaps in the stub. Exclude
    # **pipeline_kwargs (VAR_KEYWORD) — it absorbs anything, so it can't be "uncovered".
    sig = inspect.signature(openmed.analyze_text)
    openmed_params = {
        name
        for name, p in sig.parameters.items()
        if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }

    # analyze_text params the engine intentionally does not forward (the alternate
    # construction path, sentence/tokenizer tuning, formatter/metadata plumbing). Forwarding
    # one later — or openmed dropping one — must update this set; the assertions enforce that.
    intentionally_not_forwarded = {
        "model_id",  # alias for model_name; the engine passes model_name
        "config",  # alternate construction path; the engine owns loading via loader=
        "include_confidence",
        "formatter_kwargs",
        "metadata",
        "use_fast_tokenizer",
        "sentence_detection",
        "sentence_language",
        "sentence_clean",
        "sentence_segmenter",
        # openmed 1.7.0 additions — openmed-side result caching the engine doesn't thread:
        "cache_results",
        "max_cache_entries",
    }

    captured: dict[str, object] = {}

    def fake_analyze_text(text, **kwargs):
        captured.update(kwargs)
        captured["text"] = text
        return SimpleNamespace(entities=[])

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze_text)
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    engine.analyze("x", model_name="disease_detection_superclinical_141m")
    forwarded = set(captured)  # includes "text", captured positionally

    uncovered = openmed_params - forwarded - intentionally_not_forwarded
    assert not uncovered, (
        "openmed.analyze_text params neither forwarded nor allowlisted "
        f"(forward them in PIIEngine.analyze or justify them): {sorted(uncovered)}"
    )
    assert intentionally_not_forwarded <= openmed_params, (
        "stale exclusion(s) no longer in openmed.analyze_text: "
        f"{sorted(intentionally_not_forwarded - openmed_params)}"
    )
    assert not (forwarded & intentionally_not_forwarded), (
        "param is both forwarded and allowlisted — drop it from the exclusion set: "
        f"{sorted(forwarded & intentionally_not_forwarded)}"
    )


def test_extract_zero_shot_delegates_to_openmed(monkeypatch) -> None:
    # engine.extract_zero_shot resolves the alias -> HF repo id, fabricates a one-entry
    # in-memory ModelIndex (openmed.ner.infer's default on-disk index isn't shipped), and
    # forwards the labels/threshold. Crucially it must NOT touch the shared loader: the
    # GLiNER path bypasses ModelLoader, so is_loaded stays False afterwards.
    import openmed
    import openmed.ner as ner

    repo_id = "OpenMed/OpenMed-ZeroShot-NER-Disease-Small-166M"
    monkeypatch.setattr(
        openmed,
        "get_all_models",
        lambda: {"zeroshot_disease_small_166m": SimpleNamespace(model_id=repo_id)},
    )

    captured: dict[str, object] = {}

    def fake_infer(request, *, index):
        captured["model_id"] = request.model_id
        captured["labels"] = request.labels
        captured["threshold"] = request.threshold
        captured["index_ids"] = [r.id for r in index.models]
        captured["index_families"] = [r.family for r in index.models]
        return SimpleNamespace(
            entities=[
                SimpleNamespace(
                    label="Problem", text="diabetes", start=0, end=8, score=0.91
                )
            ]
        )

    monkeypatch.setattr(ner, "infer", fake_infer)

    engine = PIIEngine()  # NO loader passed
    entities = engine.extract_zero_shot(
        "diabetes today",
        model_name="zeroshot_disease_small_166m",
        labels=["Problem", "Treatment"],
        confidence_threshold=0.6,
    )

    assert captured["model_id"] == repo_id  # alias resolved to the HF repo id
    assert captured["index_ids"] == [repo_id]  # index points at the same repo id
    assert captured["index_families"] == ["gliner"]  # routed down the GLiNER branch
    assert captured["labels"] == ["Problem", "Treatment"]
    assert captured["threshold"] == 0.6
    assert [e.label for e in entities] == ["Problem"]  # NerResponse.entities unwrapped
    assert engine.is_loaded is False  # zero-shot never built the shared loader


def test_extract_zero_shot_unknown_model_raises_value_error(monkeypatch) -> None:
    # A model_name that passes validation's format check but isn't a registry alias must
    # raise a clear ValueError (the seam maps it to a pass-through message) rather than a
    # bare KeyError that surfaces as the opaque "failed unexpectedly".
    import openmed

    monkeypatch.setattr(openmed, "get_all_models", dict)  # empty registry
    engine = PIIEngine()
    with pytest.raises(ValueError, match="unknown zero-shot model"):
        engine.extract_zero_shot(
            "x", model_name="org/not-a-real-alias", labels=["Problem"]
        )


def test_zero_shot_available_and_default_labels_delegate(monkeypatch) -> None:
    import openmed.ner as ner

    monkeypatch.setattr(ner, "is_gliner_available", lambda: True)
    monkeypatch.setattr(ner, "get_default_labels", lambda _domain: ["Problem", "Test"])

    assert PIIEngine.zero_shot_available() is True
    assert PIIEngine.default_labels("clinical") == ["Problem", "Test"]


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


@pytest.mark.model
def test_engine_analyze_detects_clinical_entities(loader) -> None:
    # Real clinical NER: the default (Disease) model finds the disease mention. Uses a
    # different model than the PII fixture, loaded into the same shared loader by model_name.
    from openmed_studio.engine import DEFAULT_NER_MODEL

    engine = PIIEngine(loader=loader)
    entities = engine.analyze(
        "The patient was diagnosed with diabetes mellitus.",
        model_name=DEFAULT_NER_MODEL,
        confidence_threshold=0.5,
    )
    assert entities  # at least one entity detected
    assert any("diabetes" in e.text.lower() for e in entities)
    assert all(e.label.isupper() for e in entities)  # NER labels are UPPERCASE


@pytest.mark.model
def test_engine_extract_zero_shot_detects_user_labels() -> None:
    # Real GLiNER zero-shot: needs the `gliner` extra AND --run-model. Doubly gated so CI
    # (neither present) never downloads. The zero-shot path bypasses the shared loader, so
    # this builds a bare engine rather than using the PII `loader` fixture.
    pytest.importorskip("gliner")
    from openmed_studio.engine import DEFAULT_ZERO_SHOT_MODEL

    engine = PIIEngine()
    entities = engine.extract_zero_shot(
        "The patient was diagnosed with diabetes mellitus and hypertension.",
        model_name=DEFAULT_ZERO_SHOT_MODEL,
        labels=["Problem"],
        confidence_threshold=0.3,
    )
    assert entities  # at least one span for the arbitrary "Problem" label
    assert any("diabetes" in e.text.lower() for e in entities)
    assert all(e.score is not None for e in entities)  # zero-shot exposes .score
