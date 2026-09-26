"""Tests for the framework-free PIIEngine (openmed_studio.engine).

The fast tests verify the lazy-loading contract without touching a model; the
``@pytest.mark.model`` tests drive the real OpenMed model and are skipped unless
``--run-model`` is passed (reusing the session-scoped ``loader`` fixture).
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from openmed_studio import DEFAULT_PII_MODEL, HIDDEN_POLICIES, POLICY_MODELS, PIIEngine
from openmed_studio.engine import DEFAULT_PII_MLX_MODEL

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
    # loader is always built with OpenMedConfig(torch_attention_backend="eager") so a request
    # for the SDPA kernel DeBERTa-v2 lacks can never reach transformers. openmed 2.x's "auto"
    # no longer requests SDPA, so this is belt-and-braces — pin it anyway, and pin the pin
    # here, so an openmed regression fails in the fast suite. See PIIEngine.loader.
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


def test_deidentify_forwards_policy_to_openmed(monkeypatch) -> None:
    # The policy profile name must reach openmed.deidentify unchanged (this is the whole
    # Policy de-ID feature). Left None by the method-driven tabs, and openmed treats None as
    # "no policy override", so forwarding it unconditionally is safe.
    import openmed

    captured: dict[str, object] = {}

    def fake_deidentify(_text, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(deidentified_text="ok", pii_entities=[], mapping=None)

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    engine.deidentify("x", policy="hipaa_safe_harbor")
    assert captured["policy"] == "hipaa_safe_harbor"
    # And it defaults to None (no policy) when the caller doesn't set it — the method-driven
    # de-identify path must be unaffected by this new param.
    captured.clear()
    engine.deidentify("x", method="mask")
    assert captured["policy"] is None


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
        # ("policy" is now forwarded — see below — so it's no longer excluded.)
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
        # openmed 2.x additions — India/code-mixed i18n plumbing and request accounting.
        # `code_mixed=False` and the three hook params default to off/None, so omitting them
        # preserves 1.x behavior. `abdm` does NOT: `abdm=None` means AUTO, not off —
        # openmed/core/custom_recognizer.py::abdm_mode_enabled turns the India ABDM
        # recognizers ON whenever policy=="india_dpdp_act", locale ends in "_in", or lang is
        # hi/te — all knobs this app forwards, though india_dpdp_act is one of its
        # HIDDEN_POLICIES. Leaving it unset is a deliberate choice to let openmed decide per
        # context (forcing False would gut the India recognizers an en_IN locale or a hi/te
        # note relies on), not an inert omission:
        "abdm",  # India ABDM (Ayushman Bharat) health-ID recognizers; None = auto, not off
        "code_mixed",  # code-mixed (e.g. Hinglish) detection mode
        "token_language_tags",  # caller-supplied per-token language tags
        "lid_model",  # caller-supplied language-identification hook
        "transliterated_name_config",  # transliterated-name matching config
        "budget",  # RequestBudget accounting object
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
    engine.deidentify("x", model_name=DEFAULT_PII_MLX_MODEL)
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


def test_reidentify_restores_occurrence_keyed_entries_in_document_order() -> None:
    # openmed 2.x emits `__openmed_occurrence_v1__:<ordinal>:<surface>` keys when ONE redacted
    # surface stands for SEVERAL distinct originals (method="aadhaar_mask" does this for every
    # repeated label). Matching such a key as literal text finds nothing, so a naive restore
    # silently leaves the placeholder in place. Ordinals are assigned in entity order upstream,
    # so the nth match of the surface takes the nth original.
    restored = PIIEngine.reidentify(
        "Dr [last_name] referred to Dr [last_name].",
        {
            "__openmed_occurrence_v1__:00000001:[last_name]": "Doe",
            "__openmed_occurrence_v1__:00000002:[last_name]": "Roe",
        },
    )
    assert restored == "Dr Doe referred to Dr Roe."


def test_reidentify_mixes_occurrence_and_plain_keys_in_one_pass() -> None:
    # A single mapping carries both kinds (openmed only occurrence-keys the surfaces that
    # actually collide), and the longest-first single pass must still hold across the union —
    # "[last_name]" must not be eaten by a shorter overlapping plain key.
    restored = PIIEngine.reidentify(
        "[first_name] [last_name] and [first_name] [last_name], MRN [id].",
        {
            "[first_name]": "Jane",
            "[id]": "4827193",
            "__openmed_occurrence_v1__:00000002:[last_name]": "Roe",
            "__openmed_occurrence_v1__:00000001:[last_name]": "Doe",
        },
    )
    assert restored == "Jane Doe and Jane Roe, MRN 4827193."


def test_reidentify_leaves_surplus_occurrences_untouched() -> None:
    # More matches in the text than mapped originals: restore what we can and leave the rest
    # verbatim rather than raising or reusing the last value (openmed's reader does the same).
    restored = PIIEngine.reidentify(
        "[last_name], [last_name], [last_name]",
        {"__openmed_occurrence_v1__:00000001:[last_name]": "Doe"},
    )
    assert restored == "Doe, [last_name], [last_name]"


def test_reidentify_treats_a_malformed_occurrence_key_as_literal_text() -> None:
    # A caller can paste any mapping into the Re-identify tab. A key that merely *looks* like
    # the protocol (no ordinal) is not protocol — restore it literally instead of dropping it.
    bad = "__openmed_occurrence_v1__:notanordinal:[x]"
    assert PIIEngine.reidentify(f"see {bad} here", {bad: "Ann"}) == "see Ann here"


@pytest.mark.xfail(
    reason="PIIEngine.reidentify restores from the mapping alone, which carries no "
    "span offsets, so ordinary text equal to a plain surrogate key is 'restored' too "
    "(here the dose '10 mg' becomes '40 mg'). An XPASS means the restore became "
    "span-aware.",
    raises=AssertionError,
    strict=True,
)
def test_reidentify_restores_only_the_surrogate_spans() -> None:
    # Real model output from the Anonymize tab's defaults (replace, Deterministic,
    # seed 42): the model tags the age "40", whose surrogate is "10" — and the
    # untouched dose is also "10". Nothing in {surrogate: original} says which "10"
    # was the age, so the single pass restores both. Currently yields
    # "... Started amlodipine 40 mg once daily. ...".
    restored = PIIEngine.reidentify(
        "Mr. Amanda Coffey, 10, reviewed in hypertension clinic. BP 150/95. "
        "Started amlodipine 10 mg once daily. Recheck in 4 weeks.",
        {"10": "40", "Coffey": "Brooks", "Amanda": "Daniel"},
    )
    assert restored == (
        "Mr. Daniel Brooks, 40, reviewed in hypertension clinic. BP 150/95. "
        "Started amlodipine 10 mg once daily. Recheck in 4 weeks."
    )


# --- analyze (clinical NER) delegation (no model) ---------------------------


def test_analyze_delegates_to_openmed(monkeypatch) -> None:
    # engine.analyze wraps openmed.analyze_text the way extract wraps extract_pii:
    # forward model_name/confidence/aggregation/group_entities/output_format='dict'/loader,
    # and unwrap analyze_text's AnalyzeResult (an OBJECT with .entities, NOT a bare list
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
    assert [e.label for e in entities] == ["DISEASE"]  # AnalyzeResult unwrapped


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
        # openmed 2.x additions — sentence-segmenter selection and assertion-status
        # detection, both tuning knobs the NER tab doesn't expose:
        "sentence_backend",
        "assert_context",
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


def test_extract_zero_shot_unregistered_alias_raises_value_error(monkeypatch) -> None:
    # An allowed model_name that isn't (or is no longer) a registry alias must raise a
    # clear ValueError (the seam maps it to a pass-through message) rather than a bare
    # KeyError that surfaces as the opaque "failed unexpectedly".
    import openmed

    monkeypatch.setattr(openmed, "get_all_models", dict)  # empty registry
    engine = PIIEngine()
    with pytest.raises(ValueError, match="must be an openmed registry alias"):
        engine.extract_zero_shot(
            "x", model_name="zeroshot_disease_small_166m", labels=["Problem"]
        )


@pytest.mark.parametrize(
    "name",
    [
        "zeroshot_disease_large_459m",  # a registry zero-shot alias outside the ten
        "disease_detection_superclinical_141m",  # a token-classification model
        "OpenMed/OpenMed-ZeroShot-NER-Disease-Small-166M",  # a curated alias's repo id
        "SECRET-MRN-4471",
    ],
)
def test_extract_zero_shot_refuses_a_name_the_allowlist_does_not_admit(
    monkeypatch, name
) -> None:
    # extract_zero_shot resolves ANY registry alias and forces family="gliner" onto it,
    # so it re-checks validation's zero-shot allowlist itself: a caller that skips the
    # request model still can't reach openmed's other ~3,300 aliases. openmed is never
    # consulted — not its registry, not infer — and the message never quotes the name.
    import openmed
    import openmed.ner as ner

    def fail(*_args, **_kwargs):
        raise AssertionError("openmed was reached")

    monkeypatch.setattr(openmed, "get_all_models", fail)
    monkeypatch.setattr(ner, "infer", fail)
    with pytest.raises(ValueError, match="not an allowed zero-shot model") as excinfo:
        PIIEngine().extract_zero_shot("x", model_name=name, labels=["Problem"])
    assert name not in str(excinfo.value)


def test_extract_zero_shot_accepts_an_operator_extra(monkeypatch) -> None:
    # The engine checks the same set requests are validated against, extras included —
    # it doesn't keep a narrower copy of its own. (The set is built at import from
    # OPENMED_STUDIO_EXTRA_MODELS; patching it stands in for a relaunch.)
    import openmed
    import openmed.ner as ner

    from openmed_studio import validation

    monkeypatch.setattr(
        validation,
        "ZERO_SHOT_MODEL_NAMES",
        validation.ZERO_SHOT_MODEL_NAMES | {"zeroshot_disease_large_459m"},
    )
    repo_id = "OpenMed/OpenMed-ZeroShot-NER-Disease-Large-459M"
    monkeypatch.setattr(
        openmed,
        "get_all_models",
        lambda: {"zeroshot_disease_large_459m": SimpleNamespace(model_id=repo_id)},
    )
    captured: dict[str, object] = {}

    def fake_infer(request, *, index):
        captured["model_id"] = request.model_id
        return SimpleNamespace(entities=[])

    monkeypatch.setattr(ner, "infer", fake_infer)
    PIIEngine().extract_zero_shot(
        "x", model_name="zeroshot_disease_large_459m", labels=["Problem"]
    )
    assert captured["model_id"] == repo_id


def test_zero_shot_available_and_default_labels_delegate(monkeypatch) -> None:
    import openmed.ner as ner

    monkeypatch.setattr(ner, "is_gliner_available", lambda: True)
    monkeypatch.setattr(ner, "get_default_labels", lambda _domain: ["Problem", "Test"])

    assert PIIEngine.zero_shot_available() is True
    assert PIIEngine.default_labels("clinical") == ["Problem", "Test"]


# --- Local-path guard: a CWD entry named like a model is refused (no model) ---

_ZERO_SHOT_REPO = "OpenMed/OpenMed-ZeroShot-NER-Disease-Small-166M"


def _fail_if_openmed_runs(monkeypatch) -> None:
    """Make every openmed load/inference entry point the engine calls raise if reached."""
    import openmed
    import openmed.ner as ner

    def fail(*_args, **_kwargs):
        raise AssertionError("openmed was called")

    for name in ("extract_pii", "deidentify", "analyze_text"):
        monkeypatch.setattr(openmed, name, fail)
    monkeypatch.setattr(ner, "infer", fail)
    # The zero-shot path resolves its alias first (registry metadata, no load).
    monkeypatch.setattr(
        openmed,
        "get_all_models",
        lambda: {
            "zeroshot_disease_small_166m": SimpleNamespace(model_id=_ZERO_SHOT_REPO)
        },
    )


def _call(engine: PIIEngine, method: str, **kwargs):
    if method == "analyze":
        kwargs.setdefault("model_name", "disease_detection_superclinical_141m")
    if method == "extract_zero_shot":
        kwargs.setdefault("model_name", "zeroshot_disease_small_166m")
        kwargs.setdefault("labels", ["Problem"])
    return getattr(engine, method)("x", **kwargs)


_MODEL_METHODS = ["extract", "deidentify", "analyze", "extract_zero_shot"]


def _no_namespace_check(monkeypatch) -> None:
    # The "OpenMed"/"openai" namespace check would refuse any "OpenMed/..." directory on
    # its own; switch it off where a test must prove a specific name is enumerated.
    from openmed_studio import engine as engine_module

    monkeypatch.setattr(engine_module, "_OPENMED_NAMESPACES", ())


@pytest.mark.parametrize("method", _MODEL_METHODS)
@pytest.mark.parametrize("entry", ["OpenMed", "openai"])
def test_a_local_openmed_namespace_refuses_every_model_call(
    monkeypatch, tmp_path, entry, method
) -> None:
    # A CWD entry named like one of openmed's own namespaces shadows names openmed swaps
    # in by itself (MLX builds, language defaults, the privacy-filter fallback), so its
    # mere presence refuses every model call — before openmed is touched.
    from openmed_studio.engine import LocalModelPathError

    monkeypatch.chdir(tmp_path)
    (tmp_path / entry).mkdir()
    _fail_if_openmed_runs(monkeypatch)
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    with pytest.raises(LocalModelPathError) as excinfo:
        _call(engine, method)
    assert repr(entry) in str(excinfo.value)
    assert os.getcwd() in str(excinfo.value)  # the log names where, for the operator
    assert engine._lock.locked() is False


@pytest.mark.parametrize("method", ["extract", "deidentify"])
@pytest.mark.parametrize(
    "model_name", [None, DEFAULT_PII_MODEL, DEFAULT_PII_MLX_MODEL, "acme/extra-pii"]
)
def test_pii_calls_refuse_a_local_dir_named_like_the_model(
    monkeypatch, tmp_path, method, model_name
) -> None:
    # The effective name — the requested one, or the default when none is given (the
    # Streamlit UI never sends one) — is refused when it exists under the CWD.
    from openmed_studio.engine import LocalModelPathError

    _no_namespace_check(monkeypatch)
    monkeypatch.chdir(tmp_path)
    effective = model_name or DEFAULT_PII_MODEL
    (tmp_path / effective).mkdir(parents=True)
    _fail_if_openmed_runs(monkeypatch)
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    with pytest.raises(LocalModelPathError, match=re.escape(repr(effective))):
        _call(engine, method, model_name=model_name)


@pytest.mark.parametrize("method", ["extract", "deidentify"])
def test_pii_calls_refuse_a_local_dir_named_like_the_language_default(
    monkeypatch, tmp_path, method
) -> None:
    # For lang != "en" openmed swaps the default model for that language's own
    # (lang="fr" -> the French model) — a name the request never carried, so the engine
    # must enumerate it. The same directory is harmless for an English request.
    from openmed.core.model_registry import get_default_pii_model

    from openmed_studio.engine import LocalModelPathError

    french = get_default_pii_model("fr")
    assert french and french != DEFAULT_PII_MODEL
    _no_namespace_check(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / french).mkdir(parents=True)
    _fail_if_openmed_runs(monkeypatch)
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    with pytest.raises(LocalModelPathError, match=re.escape(repr(french))):
        _call(engine, method, lang="fr")
    with pytest.raises(AssertionError, match="openmed was called"):
        _call(
            engine, method, lang="en"
        )  # guard passes; the (failing) openmed call runs


@pytest.mark.parametrize(
    "local", ["zeroshot_disease_small_166m", _ZERO_SHOT_REPO], ids=["alias", "repo"]
)
def test_zero_shot_refuses_a_local_dir_named_like_the_alias_or_repo(
    monkeypatch, tmp_path, local
) -> None:
    from openmed_studio.engine import LocalModelPathError

    _no_namespace_check(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / local).mkdir(parents=True)
    _fail_if_openmed_runs(monkeypatch)
    with pytest.raises(LocalModelPathError, match=re.escape(repr(local))):
        _call(PIIEngine(), "extract_zero_shot")


def test_analyze_refuses_a_local_dir_named_like_the_alias(
    monkeypatch, tmp_path
) -> None:
    from openmed_studio.engine import LocalModelPathError

    _no_namespace_check(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "disease_detection_superclinical_141m").mkdir()
    _fail_if_openmed_runs(monkeypatch)
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    with pytest.raises(LocalModelPathError):
        _call(engine, "analyze")


def test_guard_refuses_a_privacy_filter_shaped_dir_openmed_would_trust(
    monkeypatch, tmp_path
) -> None:
    # The concrete hazard, with a harmless fixture (a config.json and nothing to run): a
    # CWD directory named like the default model whose config names the privacy-filter
    # family. By name the default is an ordinary model, but openmed's artifact check says
    # it would route THIS directory to create_privacy_filter_pipeline, which loads with
    # trust_remote_code=True. The engine refuses before openmed sees it. (__wrapped__
    # skips that check's lru_cache: caching True for the default's relative name would
    # route every later call in this process through the privacy filter.)
    import json

    from openmed.core.pii import (
        _is_privacy_filter_artifact_path,
        _looks_like_privacy_filter_identifier,
    )

    from openmed_studio.engine import LocalModelPathError

    monkeypatch.chdir(tmp_path)
    fixture = tmp_path / DEFAULT_PII_MODEL
    fixture.mkdir(parents=True)
    (fixture / "config.json").write_text(json.dumps({"family": "privacy-filter"}))
    assert not _looks_like_privacy_filter_identifier(DEFAULT_PII_MODEL)
    assert _is_privacy_filter_artifact_path.__wrapped__(DEFAULT_PII_MODEL)

    _fail_if_openmed_runs(monkeypatch)
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    for method in ("extract", "deidentify"):
        with pytest.raises(LocalModelPathError):
            _call(engine, method)


def test_guard_refuses_a_dangling_symlink(monkeypatch, tmp_path) -> None:
    # lexists, not exists: a symlink whose target doesn't exist yet is refused too.
    from openmed_studio.engine import LocalModelPathError

    monkeypatch.chdir(tmp_path)
    (tmp_path / "openai").symlink_to(tmp_path / "not-there-yet")
    _fail_if_openmed_runs(monkeypatch)
    with pytest.raises(LocalModelPathError):
        _call(PIIEngine(loader=cast("ModelLoader", object())), "extract")


@pytest.mark.parametrize("method", _MODEL_METHODS)
def test_guard_lets_a_clean_working_directory_through(
    monkeypatch, tmp_path, method
) -> None:
    # No false positives from unrelated entries: the call reaches openmed (here a fake
    # that raises, so reaching it is the proof).
    for unrelated in ("models", "tests", "openmed_studio", "notes.txt"):
        (tmp_path / unrelated).touch()
    monkeypatch.chdir(tmp_path)
    _fail_if_openmed_runs(monkeypatch)
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    with pytest.raises(AssertionError, match="openmed was called"):
        _call(engine, method)


def test_pii_model_names_match_openmeds_resolution() -> None:
    # The guard's language rule mirrors core/pii.py::_resolve_effective_pii_model: for
    # every language the app offers, and for both default PII ids, the names the engine
    # checks include the one openmed actually resolves — and the default model id is the
    # one openmed substitutes when model_name is omitted. Registry metadata only.
    import inspect
    import typing

    import openmed
    from openmed.core.pii import _DEFAULT_EN_MODEL, _resolve_effective_pii_model

    from openmed_studio.validation import Lang

    assert _DEFAULT_EN_MODEL == DEFAULT_PII_MODEL
    for func in (openmed.extract_pii, openmed.deidentify):
        assert inspect.signature(func).parameters["model_name"].default == (
            DEFAULT_PII_MODEL
        )
    engine = PIIEngine()
    for lang in typing.get_args(Lang):
        for model_name in (None, DEFAULT_PII_MODEL, DEFAULT_PII_MLX_MODEL):
            names = engine._pii_model_names(lang=lang, model_name=model_name)
            resolved = _resolve_effective_pii_model(
                model_name or DEFAULT_PII_MODEL, lang
            )
            assert resolved in names, (lang, model_name, names, resolved)


def test_guarded_namespaces_cover_every_model_the_app_resolves() -> None:
    # The namespace check stands in for names the engine doesn't enumerate — the repo
    # id a curated NER alias resolves to, MLX swaps — so every such name must live in a
    # guarded namespace: the curated aliases' repo ids, the per-language defaults, and
    # every target of openmed's MLX map. Registry metadata only.
    import typing

    import openmed
    from openmed.core.model_registry import get_default_pii_model
    from openmed.mlx.inference import _MLX_MODEL_MAP

    from openmed_studio.engine import (
        _OPENMED_NAMESPACES,
        NER_MODELS,
        ZERO_SHOT_MODELS,
    )
    from openmed_studio.validation import Lang

    catalog = openmed.get_all_models()
    names = {DEFAULT_PII_MODEL, DEFAULT_PII_MLX_MODEL, *_MLX_MODEL_MAP.values()}
    names |= {catalog[m.alias].model_id for m in NER_MODELS.values()}
    names |= {catalog[m.alias].model_id for m in ZERO_SHOT_MODELS.values()}
    names |= {get_default_pii_model(lang) or "" for lang in typing.get_args(Lang)}
    outside = sorted(n for n in names if n.split("/")[0] not in _OPENMED_NAMESPACES)
    assert not outside, outside


# --- Concurrency: model methods serialize on the engine's internal lock ------


@pytest.mark.parametrize("method", ["extract", "analyze", "deidentify"])
def test_model_methods_hold_lock_during_inference(monkeypatch, method) -> None:
    # The engine is shared across threads (FastAPI requests, cached Streamlit sessions),
    # so every model-calling method must run its openmed call while holding self._lock —
    # concurrent inference serializes. We record the lock state from inside the
    # monkeypatched openmed call, where the lock should be held.
    import openmed

    engine = PIIEngine(loader=cast("ModelLoader", object()))
    seen: dict[str, bool] = {}

    def record(*_args, **_kwargs):
        seen["locked"] = engine._lock.locked()
        return SimpleNamespace(
            entities=[], deidentified_text="ok", pii_entities=[], mapping=None
        )

    monkeypatch.setattr(openmed, "extract_pii", record)
    monkeypatch.setattr(openmed, "analyze_text", record)
    monkeypatch.setattr(openmed, "deidentify", record)

    if method == "extract":
        engine.extract("x")
    elif method == "analyze":
        engine.analyze("x", model_name="disease_detection_superclinical_141m")
    else:
        engine.deidentify("x")

    assert seen["locked"] is True  # the openmed call ran under the lock
    assert engine._lock.locked() is False  # and the lock was released afterward


def test_extract_zero_shot_holds_lock_during_inference(monkeypatch) -> None:
    # The 4th model-calling method — the GLiNER path that bypasses the shared loader — must
    # also run its openmed.ner.infer call under self._lock. It needs its own test (not the
    # parametrize above) because it monkeypatches get_all_models + openmed.ner.infer instead
    # of a top-level openmed.* function.
    import openmed
    import openmed.ner as ner

    repo_id = "OpenMed/OpenMed-ZeroShot-NER-Disease-Small-166M"
    monkeypatch.setattr(
        openmed,
        "get_all_models",
        lambda: {"zeroshot_disease_small_166m": SimpleNamespace(model_id=repo_id)},
    )

    engine = PIIEngine()
    seen: dict[str, bool] = {}

    def fake_infer(_request, *, index):
        seen["locked"] = engine._lock.locked()
        return SimpleNamespace(entities=[])

    monkeypatch.setattr(ner, "infer", fake_infer)
    engine.extract_zero_shot(
        "x", model_name="zeroshot_disease_small_166m", labels=["Problem"]
    )

    assert seen["locked"] is True
    assert engine._lock.locked() is False


def test_reidentify_is_lock_free() -> None:
    # reidentify does no model work, so it must never touch the lock — it returns even
    # while the lock is held (a lock-acquiring impl would deadlock this very call).
    engine = PIIEngine()
    with engine._lock:
        assert engine.reidentify("[a]", {"[a]": "b"}) == "b"


# --- Model-backed tests (real OpenMed engine; need --run-model) -------------


@pytest.mark.model
def test_engine_aadhaar_mask_roundtrips_through_reidentify(loader, note) -> None:
    # The regression that motivated the occurrence-key support: aadhaar_mask is NOT in
    # openmed's unique-placeholder set, so every same-label entity collapses onto one
    # placeholder and the WHOLE mapping comes back occurrence-keyed. Only a real model
    # produces those keys, so the fast tests above use synthetic ones and this pins the
    # end-to-end round trip on the method that actually triggers it.
    engine = PIIEngine(loader=loader)
    result = engine.deidentify(note, method="aadhaar_mask", keep_mapping=True)
    mapping = result.mapping or {}
    assert any(k.startswith("__openmed_occurrence_v1__:") for k in mapping), (
        "expected occurrence-keyed entries from aadhaar_mask; if openmed changed this, the "
        "fast occurrence tests still stand but this guard no longer covers the real path"
    )
    assert engine.reidentify(result.deidentified_text, mapping) == note


@pytest.mark.model
def test_engine_refuses_a_poisoned_working_directory_then_recovers(
    loader, note, tmp_path, monkeypatch
) -> None:
    # The real engine and loader: under a working directory holding a privacy-filter-shaped
    # directory named like the default model (a config.json, nothing to run), both PII
    # calls are refused before openmed sees the name, and back in a clean directory the
    # same engine detects PII normally. (Had openmed seen it, its lru_cache'd artifact
    # check could have cached "privacy filter" for the default's relative name, outliving
    # the directory.)
    import json

    from openmed_studio.engine import LocalModelPathError

    engine = PIIEngine(loader=loader)
    home = os.getcwd()
    monkeypatch.chdir(tmp_path)
    fixture = tmp_path / DEFAULT_PII_MODEL
    fixture.mkdir(parents=True)
    (fixture / "config.json").write_text(json.dumps({"family": "privacy-filter"}))
    with pytest.raises(LocalModelPathError):
        engine.extract(note)
    with pytest.raises(LocalModelPathError):
        engine.deidentify(note, method="mask")

    monkeypatch.chdir(home)
    found = {(e.label, e.text) for e in engine.extract(note)}
    assert ("ssn", "123-45-6789") in found


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
@pytest.mark.parametrize(
    "reversible", [m.name for m in POLICY_MODELS.values() if m.keep_mapping]
)
def test_engine_deidentify_policy_masks_and_pseudonymizes(
    loader, note, reversible
) -> None:
    # Real policy-driven anonymization proves two things the service path relies on: the policy
    # (not the flat method) drives the per-label action, AND reversibility is the policy's call.
    # Both calls pass keep_mapping=False (what service.anonymize_policy sends) and leave method at
    # its default — the policy overrides it. The default policy (HIPAA Safe Harbor) masks the SSN
    # irreversibly (no mapping); each reversible profile offered (GDPR Art. 9 Health, China
    # PIPL) replaces it with a surrogate and FORCES a mapping (openmed ORs the profile's own
    # keep_mapping) that round-trips.
    from openmed_studio.engine import DEFAULT_POLICY_MODEL

    engine = PIIEngine(loader=loader)

    masked = engine.deidentify(note, policy=DEFAULT_POLICY_MODEL, keep_mapping=False)
    assert "123-45-6789" not in masked.deidentified_text  # SSN redacted
    assert (
        not masked.mapping
    )  # Safe Harbor is irreversible — the policy keeps no mapping

    pseudo = engine.deidentify(note, policy=reversible, keep_mapping=False)
    assert "123-45-6789" not in pseudo.deidentified_text  # replaced by a surrogate
    assert (
        pseudo.mapping
    )  # reversible: the policy forces a key despite keep_mapping=False
    # The kept mapping round-trips: re-identifying restores the original SSN.
    restored = PIIEngine.reidentify(pseudo.deidentified_text, pseudo.mapping)
    assert "123-45-6789" in restored


# A synthetic note for the policy pins below; every identifier is fabricated. "devout
# Catholic" is a sensitive trait (the PII model's religious_belief label).
_POLICY_NOTE = (
    "Mr. Tom Hale, a devout Catholic and 54-year-old teacher from Dayton, Ohio 45402, was "
    "seen at Mercy Clinic on 03/14/2024, at Riverside General Hospital on 2024-03-14, "
    "March 14, 2024 and 14 March 2024, and again on March 14th, 2024 and in June 2023 for "
    "type 2 diabetes. Phone 937-555-0142, SSN 123-45-6789, billing ZIP code: 45409."
)
# Identifiers the PII model detects but openmed's normalize_label files under the catch-all
# OTHER label (certificate_license_number, tax_id, company_name, religious_belief) — the
# reason HIDDEN_POLICIES exists. Fabricated.
_OTHER_NOTE = (
    "Mr. Tom Hale, driver's license D4417-2290, tax ID 98-7654321, works at Brightline "
    "Logistics and is a devout Catholic."
)
_OTHER_SURFACES = ("D4417-2290", "98-7654321", "Brightline Logistics", "Catholic")


def _anonymize_note(engine: PIIEngine, policy: str, text: str = _POLICY_NOTE):
    # What service.anonymize_policy sends (no method, keep_mapping=False) at the Policy de-ID
    # tab's defaults: confidence 0.5 (the engine's own default is 0.7), deterministic
    # surrogates with seed 42, and the sweep on.
    return engine.deidentify(
        text,
        policy=policy,
        keep_mapping=False,
        confidence_threshold=0.5,
        consistent=True,
        seed=42,
        use_safety_sweep=True,
    )


@pytest.mark.model
@pytest.mark.parametrize(
    "policy", sorted({m.name for m in POLICY_MODELS.values()} | HIDDEN_POLICIES)
)
def test_engine_hidden_policies_keep_other_identifiers_verbatim(loader, policy) -> None:
    # Pins the reason for HIDDEN_POLICIES with a real run, one openmed profile per case: a
    # hidden profile (OTHER = keep) passes a detected license number, tax ID, employer and
    # religion through verbatim AND leaves them out of pii_entities (the tab's entity table),
    # while every offered profile masks all four and lists them. A hidden profile's case
    # failing here is the signal that openmed fixed it and it can be re-exposed (see
    # HIDDEN_POLICIES in engine.py). The sweep-vs-keep description pins for the four
    # keep-dates profiles are in commit 8e1e50f; all four are hidden, and no offered
    # profile keeps a label for the sweep to override.
    engine = PIIEngine(loader=loader)
    result = _anonymize_note(engine, policy, _OTHER_NOTE)
    out = result.deidentified_text
    listed = {e.text for e in result.pii_entities}
    hidden = policy in HIDDEN_POLICIES
    for surface in _OTHER_SURFACES:
        assert (surface in out) is hidden, (policy, surface)
        assert (surface in listed) is not hidden, (policy, surface)
    assert "Tom Hale" not in out  # every profile handles the name either way


@pytest.mark.model
@pytest.mark.parametrize(
    "policy", ["africa_malabo_baseline", "ke_dpa", "eg_pdpl", "ma_law_09_08"]
)
def test_engine_mask_all_profiles_match_strict_no_leak(loader, policy) -> None:
    # Pins "identical to Strict No-Leak" in these four descriptions, and "every detected
    # span" in all five: their rules mask the clinical labels too, but the PII model has
    # none, so a diagnosis is never a detected span. If clinical text ever starts being
    # masked here, the descriptions can say so again.
    engine = PIIEngine(loader=loader)
    baseline = _anonymize_note(engine, "strict_no_leak").deidentified_text
    assert _anonymize_note(engine, policy).deidentified_text == baseline
    for masked in (
        "Tom Hale",
        "Catholic",
        "54-year-old",
        "teacher",
        "Dayton",
        "03/14/2024",
    ):
        assert masked not in baseline, masked
    assert "type 2 diabetes" in baseline


@pytest.mark.model
@pytest.mark.parametrize(
    ("policy", "surrogated", "masked"),
    [
        *(
            (
                twin,
                {
                    "first_name": "Tom Hale",
                    "phone_number": "937-555-0142",
                    "ssn": "123-45-6789",
                },
                {
                    "city": "Dayton",
                    "date": "03/14/2024",
                    "religious_belief": "Catholic",
                },
            )
            for twin in ("china_pipl", "gdpr_art9_health")  # identical action maps
        ),
        (
            "ng_ndpa",
            {"first_name": "Tom Hale", "phone_number": "937-555-0142"},
            {
                "ssn": "123-45-6789",
                "city": "Dayton",
                "date": "03/14/2024",
                "religious_belief": "Catholic",
            },
        ),
        (
            "za_popia",
            {
                "first_name": "Tom Hale",
                "phone_number": "937-555-0142",
                "city": "Dayton",
            },
            {
                "ssn": "123-45-6789",
                "age": "54-year-old",
                "occupation": "teacher",
                "date": "03/14/2024",
                "religious_belief": "Catholic",
            },
        ),
    ],
)
def test_engine_surrogate_profiles_split_as_described(
    loader, policy, surrogated, masked
) -> None:
    # Pins the surrogate-vs-mask split these three descriptions state, keyed label -> surface:
    # a surrogate removes the surface and leaves no "[label]" placeholder, a mask leaves one.
    # Clinical text passes through all of them, which is why none may say "mask all else".
    engine = PIIEngine(loader=loader)
    out = _anonymize_note(engine, policy).deidentified_text
    for label, surface in surrogated.items():
        assert surface not in out and f"[{label}]" not in out, (policy, label)
    for label, surface in masked.items():
        assert surface not in out and f"[{label}]" in out, (policy, label)
    assert "type 2 diabetes" in out


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
