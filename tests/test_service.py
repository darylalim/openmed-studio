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
from openmed_studio.engine import DEFAULT_PII_MLX_MODEL
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

    def extract_zero_shot(self, _text, **_):
        # openmed's zero-shot Entity exposes .score (not .confidence) — the adapter must
        # normalize it. No .confidence attribute here, on purpose.
        return [
            SimpleNamespace(
                label="Problem", text="diabetes", start=0, end=8, score=0.88
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

    def extract_zero_shot(self, _text, **_):
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
        if method in ("analyze", "extract_zero_shot")
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


# --- policy-driven anonymization (stub engine) ------------------------------


def test_anonymize_policy_returns_deidentify_dict() -> None:
    # Flows through engine.deidentify + _deidentify_dict (Option A), so it yields the same
    # {deidentified_text, method, entities, mapping} shape — with the policy name in `method`.
    result = service.anonymize_policy(
        _stub(), "Patient John.", policy="hipaa_safe_harbor"
    )
    assert result["deidentified_text"] == "[first_name] A. Doe"
    assert (
        result["method"] == "hipaa_safe_harbor"
    )  # the policy name rides the method slot
    assert result["entities"] == [
        {
            "label": "first_name",
            "text": "John",
            "start": 0,
            "end": 4,
            "confidence": 0.99,
        }
    ]


def test_anonymize_policy_forwards_policy_and_lets_policy_decide_mapping() -> None:
    # The seam forwards the validated policy but does NOT force keep_mapping: reversibility is the
    # policy's call (openmed ORs the profile's own flag), so masking policies stay irreversible.
    # It sends NO method (the policy overrides it).
    engine, captured = _capturing()  # captures engine.deidentify kwargs
    service.anonymize_policy(
        engine, "x", policy="gdpr_art9_health", consistent=True, seed=7
    )
    assert captured["policy"] == "gdpr_art9_health"
    assert captured["keep_mapping"] is False  # not forced on — the policy decides
    assert captured["consistent"] is True
    assert captured["seed"] == 7
    assert "method" not in captured  # policy overrides method; none is sent


def test_anonymize_policy_surfaces_a_policy_forced_mapping() -> None:
    # A reversible policy makes openmed return a mapping even though the seam passes
    # keep_mapping=False; the dict adapter surfaces it to the UI. Modeled with a stub whose
    # deidentify returns a mapping regardless (as a reversible profile does).
    class _Reversible(_StubEngine):
        def deidentify(self, _text, **_):
            return SimpleNamespace(
                deidentified_text="ok", pii_entities=[], mapping={"Wong": "John"}
            )

    engine = cast("PIIEngine", _Reversible())
    result = service.anonymize_policy(engine, "x", policy="gdpr_art9_health")
    assert result["mapping"] == {"Wong": "John"}


def test_anonymize_policy_masking_policy_yields_no_mapping() -> None:
    # A masking policy leaves the seam's keep_mapping=False in effect, so openmed returns no
    # mapping and the adapter surfaces None (the default _StubEngine gives mapping only when
    # keep_mapping is requested — which the policy path does not).
    result = service.anonymize_policy(
        _stub(), "Patient John.", policy="hipaa_safe_harbor"
    )
    assert result["mapping"] is None


def test_anonymize_policy_value_error_maps_to_service_error() -> None:
    with pytest.raises(ServiceError, match="bad option"):
        service.anonymize_policy(
            _raising(ValueError("bad option")), "x", policy="hipaa_safe_harbor"
        )


def test_anonymize_policy_backend_failure_does_not_leak() -> None:
    with pytest.raises(ServiceError) as excinfo:
        service.anonymize_policy(
            _raising(RuntimeError("policy model exploded")),
            "x",
            policy="hipaa_safe_harbor",
        )
    message = str(excinfo.value)
    assert "exploded" not in message
    assert "unavailable" in message.lower()


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


def test_zero_shot_returns_entity_dicts_with_score_as_confidence() -> None:
    # Zero-shot flows through _entity_dict, whose .score fallback maps openmed.ner.Entity's
    # .score (it has no .confidence) into the UI's confidence field.
    result = service.extract_zero_shot(
        _stub(),
        "Has diabetes.",
        model_name="zeroshot_disease_small_166m",
        labels=["Problem"],
    )
    assert result["entities"] == [
        {
            "label": "Problem",
            "text": "diabetes",
            "start": 0,
            "end": 8,
            "confidence": 0.88,  # came from .score, not .confidence
        }
    ]


def test_zero_shot_forwards_options_to_engine() -> None:
    engine, captured = _capturing("extract_zero_shot")
    service.extract_zero_shot(
        engine,
        "x",
        model_name="zeroshot_anatomy_small_166m",
        labels=["Organ", "Organ", " organ "],  # deduped by validation before the engine
        confidence_threshold=0.4,
    )
    assert captured["model_name"] == "zeroshot_anatomy_small_166m"
    assert captured["labels"] == ["Organ"]  # normalized/deduped upstream of the engine
    assert captured["confidence_threshold"] == 0.4


def test_zero_shot_missing_gliner_maps_to_actionable_service_error() -> None:
    # openmed raises an ImportError subclass (MissingDependencyError) when the gliner extra
    # isn't installed; _run passes its message through, so a /zero-shot caller sees
    # openmed's own install hint (the Streamlit tab checks zero_shot_available() first and
    # shows its own `uv sync --extra gliner` hint). The text mirrors openmed 2.5's.
    hint = (
        "Optional dependency 'gliner' is required for this operation. "
        "Install with `pip install openmed[gliner]`."
    )
    with pytest.raises(ServiceError, match="gliner") as excinfo:
        service.extract_zero_shot(
            _raising(ImportError(hint)),
            "x",
            model_name="zeroshot_disease_small_166m",
            labels=["Problem"],
        )
    assert excinfo.value.kind == "dependency"
    assert str(excinfo.value) == hint


def test_zero_shot_backend_failure_does_not_leak() -> None:
    with pytest.raises(ServiceError) as excinfo:
        service.extract_zero_shot(
            _raising(RuntimeError("gliner model exploded")),
            "x",
            model_name="zeroshot_disease_small_166m",
            labels=["Problem"],
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
    # A RuntimeError backend failure (openmed's offline-mode/model-integrity errors) isn't
    # note-specific (it fails every note identically), so it aborts the whole batch via
    # _run rather than spamming N failed rows — and never leaks.
    with pytest.raises(ServiceError) as excinfo:
        service.deidentify_batch(_raising(RuntimeError("kaboom")), ["x", "y"])
    assert "kaboom" not in str(excinfo.value)


class _LoadError(ImportError, ValueError):
    """Stand-in for openmed's ``ModelLoadError`` — an ``ImportError`` *and* ``ValueError``.

    Mirrors only the two builtin bases that decide ``_run``'s ``except`` routing, so these
    tests pin that order without importing openmed. The real class (2.3+) also has
    openmed's ``CapabilityError``/``OpenMedError`` bases, which no ``except`` here names.
    """


def test_run_classifies_model_load_error_as_bad_options() -> None:
    # A model_name that fails to load is the caller's to fix, so _run must catch ValueError
    # BEFORE ImportError — swap them and this becomes a "dependency" 503 telling the caller
    # to install something. openmed's message is PHI-free by contract, so it passes through.
    # e.g. the default's pre-converted MLX build requested on a host without MLX.
    message = f"Could not load model {DEFAULT_PII_MLX_MODEL}. Verify the model ID."
    with pytest.raises(ServiceError) as excinfo:
        service.extract(
            _raising(_LoadError(message)), "x", model_name=DEFAULT_PII_MLX_MODEL
        )
    assert excinfo.value.kind == "bad_options"
    assert str(excinfo.value) == message


def test_batch_isolates_model_load_error_per_note() -> None:
    # In a batch the same error meets the per-note `except ValueError` net first, so every
    # note gets its own ok=False row instead of the batch aborting (unlike the RuntimeError
    # above). Guards against an `except ImportError: raise` sneaking in ahead of that net.
    result = service.deidentify_batch(
        _raising(_LoadError("could not load")), ["x", "y"], method="mask"
    )
    results = result["results"]
    assert [r["ok"] for r in results] == [False, False]
    assert all(r["error"] == "could not load" for r in results)


class _CodedRuntimeError(RuntimeError):
    """Stand-in for openmed's ``RuntimeError``-based errors, which carry a stable ``.code``.

    ``_run`` duck-types on that code rather than importing openmed, so the builtin base and
    the attribute are all a stand-in needs. The real classes are pinned further down, by
    ``test_openmed_internal_codes_match_openmed``.
    """

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


# What an internal error might carry: it must reach the server log, never the caller.
_RAW_DETAIL = "invariant failed near RAW-DETAIL-4471"


@pytest.mark.parametrize("code", ["internal_error", "inference_error"])
def test_run_classifies_openmed_internal_error_as_internal(code, caplog) -> None:
    # openmed's InternalError (and its InferenceError subclass) is a RuntimeError, so
    # without the code check it reads as a 503 "the model failed to load" — e.g. openmed
    # 2.5's safety-sweep invariant, which one note's content trips on a healthy model.
    exc = _CodedRuntimeError(_RAW_DETAIL, code)
    with (
        caplog.at_level(logging.ERROR, logger="openmed_studio"),
        pytest.raises(ServiceError) as excinfo,
    ):
        service.deidentify(_raising(exc), "x", method="mask")
    assert excinfo.value.kind == "internal"
    assert str(excinfo.value) == "The request failed unexpectedly."
    assert "RAW-DETAIL-4471" in caplog.text  # the detail goes to the log instead


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("kaboom"),  # like ModelIntegrityError/OfflineModeError: no .code
        _CodedRuntimeError("over budget", "budget_exceeded"),  # BudgetExceededError's
    ],
)
def test_run_keeps_other_runtime_errors_unavailable(exc) -> None:
    # Only openmed's internal codes are carved out of the RuntimeError branch: an outage
    # and openmed's budget error (a 503 in openmed's own REST service too) stay there.
    with pytest.raises(ServiceError) as excinfo:
        service.deidentify(_raising(exc), "x", method="mask")
    assert excinfo.value.kind == "unavailable"


def test_batch_isolates_openmed_internal_error_per_note(caplog) -> None:
    # Unlike an outage, the internal error's known trigger is one note's content, so it
    # gets its own ok=False row (the other notes still complete) — carrying the generic
    # message, never openmed's text. A code-less RuntimeError still aborts the batch
    # (test_batch_backend_failure_aborts_and_does_not_leak).
    class _Mixed(_StubEngine):
        def deidentify(self, _text, **kwargs):
            if "MRZ" in _text:
                raise _CodedRuntimeError(_RAW_DETAIL, "internal_error")
            return super().deidentify(_text, **kwargs)

    engine = cast("PIIEngine", _Mixed())
    with caplog.at_level(logging.ERROR, logger="openmed_studio"):
        result = service.deidentify_batch(
            engine, ["Patient John.", "MRZ note", "Patient Jane."], method="mask"
        )
    results = result["results"]
    assert [r["ok"] for r in results] == [True, False, True]
    assert results[1]["error"] == "The request failed unexpectedly."
    assert "RAW-DETAIL-4471" in caplog.text


def test_openmed_internal_codes_match_openmed() -> None:
    # service.py bakes openmed's internal codes (it stays openmed-free), so pin the copy
    # against the real classes, driven through the seam: the whole InternalError family
    # must classify "internal", while the RuntimeError-based errors that mean the backend
    # can't serve — and the budget error — must stay "unavailable". A renamed code, or a
    # new InternalError subclass with its own code, fails here. No model is loaded.
    from openmed.core import errors
    from openmed.core.model_integrity import ModelIntegrityError
    from openmed.core.offline import OfflineModeError

    family: set[type[errors.InternalError]] = set()
    pending: list[type[errors.InternalError]] = [errors.InternalError]
    while pending:
        cls = pending.pop()
        family.add(cls)
        pending.extend(cls.__subclasses__())
    assert {cls.code for cls in family} == set(service._OPENMED_INTERNAL_CODES)

    def kind_of(exc: Exception) -> str:
        with pytest.raises(ServiceError) as excinfo:
            service.deidentify(_raising(exc), "x", method="mask")
        return excinfo.value.kind

    for cls in family:
        assert kind_of(cls("invariant failed")) == "internal"
    integrity = ModelIntegrityError("org/m", expected_sha256="a", actual_sha256="b")
    for exc in (integrity, OfflineModeError("offline"), errors.BudgetExceededError()):
        assert kind_of(exc) == "unavailable"


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


# --- ServiceError.kind (the transport-neutral classification the API maps) ---


def test_service_error_defaults_to_internal_kind() -> None:
    # An unclassified failure is the server's fault, not the caller's — so the default is
    # "internal" (500), never "bad_options" (400).
    assert ServiceError("x").kind == "internal"
    assert ServiceError("x", kind="bad_options").kind == "bad_options"


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (ValueError("bad option"), "bad_options"),
        (RuntimeError("kaboom"), "unavailable"),
        (OSError("io error"), "unavailable"),
        (ImportError("run uv sync --extra gliner"), "dependency"),
        (KeyError("leak-me"), "internal"),
    ],
)
def test_run_classifies_engine_failures_by_kind(exc, kind) -> None:
    # The kind the FastAPI layer maps to an HTTP status is set from the exception class.
    with pytest.raises(ServiceError) as excinfo:
        service.extract(_raising(exc), "x")
    assert excinfo.value.kind == kind


def test_validation_failure_has_validation_kind() -> None:
    # A pre-engine schema rejection carries kind="validation" (a bad request body).
    with pytest.raises(ServiceError) as excinfo:
        service.extract(_stub(), "x", confidence_threshold=5.0)
    assert excinfo.value.kind == "validation"


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
