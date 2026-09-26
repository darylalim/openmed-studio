"""Tests for the input guarantees the Pydantic request models enforce.

The request models validate every request in-process via ``openmed_studio.service``
(and, unchanged, serve as the FastAPI request bodies); the seam raises ``ServiceError``
on rejection (validation runs before the stub engine is reached). This pins the
text/batch/mapping caps, the value/enum/format checks, the per-capability ``model_name``
allowlists, the ``OPENMED_STUDIO_MAX_TEXT_LENGTH`` and ``OPENMED_STUDIO_EXTRA_MODELS``
knobs, the ``DeidMethod``↔openmed sync, and that rejection messages never echo the
offending input (possible PHI).
"""

from __future__ import annotations

import os
import subprocess
import sys
import typing
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from openmed_studio import PIIEngine, engine, service, validation
from openmed_studio.service import ServiceError

_REPO_ROOT = Path(__file__).resolve().parents[1]


class _StubEngine:
    """Returns canned results; only reached when validation passes."""

    def extract(self, _text, **_):
        return []

    def deidentify(self, _text, **_):
        return SimpleNamespace(deidentified_text="ok", pii_entities=[], mapping=None)

    def analyze(self, _text, **_):
        return []

    def extract_zero_shot(self, _text, **_):
        return []

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


def test_rejects_malformed_locale() -> None:
    # locale flows into Faker; reject obviously-malformed values up front with a
    # PHI-safe message rather than letting Faker raise mid-call. (A well-formed but
    # unknown locale is openmed/Faker's to reject at call time, not validation's.)
    with pytest.raises(ServiceError):
        service.deidentify(ENGINE, "x", method="replace", locale="not a locale!")


_NER_MODEL = "disease_detection_superclinical_141m"


def test_ner_rejects_missing_model_name() -> None:
    # model_name is REQUIRED for NER — an absent one would silently fall back to
    # openmed's disease-only default, so validation must reject it.
    with pytest.raises(ServiceError):
        service.analyze(ENGINE, "x")


def test_ner_rejects_invalid_model_name() -> None:
    with pytest.raises(ServiceError):
        service.analyze(ENGINE, "x", model_name="../etc/passwd")


def test_ner_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ServiceError):
        service.analyze(ENGINE, "x", model_name=_NER_MODEL, confidence_threshold=1.5)


def test_ner_rejects_bad_aggregation_strategy() -> None:
    with pytest.raises(ServiceError):
        service.analyze(
            ENGINE, "x", model_name=_NER_MODEL, aggregation_strategy="bogus"
        )


def test_ner_rejects_unknown_field() -> None:
    with pytest.raises(ServiceError):
        service.analyze(ENGINE, "x", model_name=_NER_MODEL, bogus=1)


# --- model_name path guard --------------------------------------------------

# Names that fit the one-"/" cap but have a segment opening with ".". openmed checks the
# filesystem before its registry, so any of them that exists would otherwise load as a
# local path: the parent dir, the working dir, a sibling, or a hidden entry.
_DOT_SEGMENT_NAMES = [
    "..",
    ".",
    "../x",
    "x/..",
    "./x",
    "x/.",
    ".venv",
    ".streamlit/config.toml",
]


@pytest.mark.parametrize("name", _DOT_SEGMENT_NAMES)
def test_rejects_dot_segment_model_name(name) -> None:
    with pytest.raises(ServiceError) as excinfo:
        service.extract(ENGINE, "x", model_name=name)
    assert "must not start with '.'" in str(excinfo.value)


def test_three_segment_model_name_fails_the_format_check_first() -> None:
    # "a/./b" (like "../etc/passwd" above) is three segments, which the one-"/" cap
    # already rejects; it never reaches the dot rule.
    with pytest.raises(ServiceError) as excinfo:
        service.extract(ENGINE, "x", model_name="a/./b")
    assert "must look like 'org/model' or 'model'" in str(excinfo.value)


def test_model_name_rejection_does_not_echo_input() -> None:
    # Both messages (dot rule, format) name the rule, never the value — a model_name
    # field is one stray paste away from holding note text.
    for name in ("../SECRET-MRN-4471", "SECRET MRN 4471"):
        with pytest.raises(ServiceError) as excinfo:
            service.extract(ENGINE, "x", model_name=name)
        assert "SECRET" not in str(excinfo.value)


def _models_with_a_model_name() -> set[type[BaseModel]]:
    """Every request model — the seam's and the /compat bodies in main.py — with one."""
    from openmed_studio import main

    models = {
        obj
        for module in (validation, main)
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and "model_name" in obj.model_fields
    }
    found = {model.__name__ for model in models}
    assert {
        "ExtractRequest",
        "NerRequest",
        "ZeroShotRequest",
        "AnonymizePolicyRequest",
        "DeidentifyRequest",
        "DeidentifyBatchRequest",
        "CompatExtractRequest",
        "CompatDeidentifyRequest",
    } <= found
    return models


def _model_name_error(model: type[BaseModel], name: str) -> str:
    with pytest.raises(ValidationError) as excinfo:
        model.model_validate({"model_name": name})
    messages = {e["loc"]: e["msg"] for e in excinfo.value.errors()}
    return messages.get(("model_name",), "")


def test_every_model_name_field_applies_the_dot_rule() -> None:
    # One rule on every surface: each request model that declares a model_name must
    # route it through _check_model_name, so a new model can't reopen "../x" with a bare
    # `str` field.
    for model in _models_with_a_model_name():
        assert "must not start with '.'" in _model_name_error(model, "../x"), (
            f"{model.__name__}.model_name skips the dot rule"
        )


def test_every_model_name_field_applies_an_allowlist() -> None:
    # ...and an allowlist after it, so a new model can't reopen arbitrary Hub ids (or
    # the privacy-filter names openmed loads with trust_remote_code=True) with a
    # format-only field.
    for model in _models_with_a_model_name():
        assert "is not an allowed" in _model_name_error(
            model, "openai/privacy-filter"
        ), f"{model.__name__}.model_name skips the allowlist"


def test_every_lang_field_is_the_apps_lang_literal() -> None:
    # A `lang` openmed supports but the app doesn't list can swap openmed's
    # privacy-filter model in for the default (fa -> OpenMed/privacy-filter-multilingual,
    # loaded with trust_remote_code=True) without any model_name. So every request model
    # with a lang — the /compat bodies included — takes the Lang Literal, not a str.
    from openmed_studio import main

    models = {
        obj
        for module in (validation, main)
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and "lang" in obj.model_fields
    }
    assert {"ExtractRequest", "CompatExtractRequest", "CompatDeidentifyRequest"} <= {
        model.__name__ for model in models
    }
    for model in models:
        assert model.model_fields["lang"].annotation == validation.Lang, model.__name__
        with pytest.raises(ValidationError) as excinfo:
            model.model_validate({"lang": "fa"})
        assert ("lang",) in {e["loc"] for e in excinfo.value.errors()}, model.__name__


def test_accepts_every_openmed_registry_model_name() -> None:
    # The dot rule must cost no real model: every alias and HF model id in openmed's
    # registry, plus the ids engine.py bakes, passes unchanged. (The Hub itself forbids
    # a leading "." in a repo id.) Registry metadata only — no model download.
    import openmed

    catalog = openmed.get_all_models()  # dict[alias -> ModelInfo]
    names = set(catalog) | {info.model_id for info in catalog.values()}
    names |= {engine.DEFAULT_PII_MODEL, engine.DEFAULT_PII_MLX_MODEL}
    names |= {engine.DEFAULT_NER_MODEL}
    names |= {engine.DEFAULT_ZERO_SHOT_MODEL}
    names |= {model.alias for model in engine.NER_MODELS.values()}
    names |= {model.alias for model in engine.ZERO_SHOT_MODELS.values()}
    rejected = []
    for name in names:
        try:
            if validation._check_model_name(name) != name:
                rejected.append(name)
        except ValueError:
            rejected.append(name)
    assert not rejected, sorted(rejected)[:10]


# --- model_name allowlists (per capability) ---------------------------------

# A format-valid model_name that could be a pasted MRN; no rejection may quote it.
_SENTINEL = "SECRET-MRN-4471"

# Every PII entry point, keyed for test ids; each forwards a model_name to the engine.
_PII_CALLS = {
    "extract": lambda **kw: service.extract(ENGINE, "x", **kw),
    "deidentify": lambda **kw: service.deidentify(ENGINE, "x", **kw),
    "deidentify_batch": lambda **kw: service.deidentify_batch(ENGINE, ["x"], **kw),
    "anonymize_policy": lambda **kw: service.anonymize_policy(
        ENGINE, "x", policy="hipaa_safe_harbor", **kw
    ),
}


def _ner(**kw):
    return service.analyze(ENGINE, "x", **kw)


def _zero_shot(**kw):
    return service.extract_zero_shot(ENGINE, "x", labels=["Problem"], **kw)


# Format-valid names no PII field may accept.
_DISALLOWED_PII_NAMES = [
    # openmed routes these to create_privacy_filter_pipeline, which loads with
    # trust_remote_code=True and no revision pin, and its prefix check is
    # case-INsensitive — so every casing must fail here, not just the canonical one.
    "openai/privacy-filter",
    "OpenAI/Privacy-Filter",
    "OpenMed/privacy-filter-multilingual",
    "openmed/privacy-filter-multilingual",
    "OPENMED/PRIVACY-FILTER-MULTILINGUAL",
    "OpenMed/Privacy-Filter-Multilingual",
    "OpenMed/privacy-filter-nemotron",
    "OpenMed/privacy-filter-mlx",
    "privacy-filter",  # openmed's bare family aliases route there too
    "openai-privacy-filter",
    # -mlx builds other than the default's own
    "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-141M-mlx",
    "OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx",
    # case-variants of the allowed names (openmed's registry lookup is exact)
    engine.DEFAULT_PII_MODEL.lower(),
    engine.DEFAULT_PII_MLX_MODEL.upper(),
    # a registry alias of the default: one accepted spelling per model
    "pii_superclinical_small",
    # a language default: openmed swaps it in for lang="fr"; it isn't requested by name
    "OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1",
    # arbitrary Hub ids
    "attacker/pii-model",
    "bert-base-uncased",
    _SENTINEL,
    # other capabilities' models
    engine.DEFAULT_NER_MODEL,
    engine.DEFAULT_ZERO_SHOT_MODEL,
]


@pytest.mark.parametrize(
    "name", [None, engine.DEFAULT_PII_MODEL, engine.DEFAULT_PII_MLX_MODEL]
)
@pytest.mark.parametrize("call", sorted(_PII_CALLS))
def test_pii_fields_accept_the_default_and_its_mlx_build(call, name) -> None:
    # None (openmed's default, per language) and the two spellings of the default model
    # reach the stub engine.
    _PII_CALLS[call](model_name=name)


@pytest.mark.parametrize("name", _DISALLOWED_PII_NAMES)
def test_pii_fields_reject_every_other_model(name) -> None:
    for call in _PII_CALLS.values():
        with pytest.raises(ServiceError) as excinfo:
            call(model_name=name)
        assert excinfo.value.kind == "validation"
        assert "not an allowed PII model" in str(excinfo.value)


@pytest.mark.parametrize("alias", sorted(m.alias for m in engine.NER_MODELS.values()))
def test_ner_accepts_every_curated_alias(alias) -> None:
    assert _ner(model_name=alias) == {"entities": []}


@pytest.mark.parametrize(
    "name",
    [
        # the Disease alias's own repo id: NER accepts aliases only (see NER_MODEL_NAMES)
        "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-141M",
        engine.DEFAULT_NER_MODEL.upper(),
        "disease_detection_superclinical",  # a registry alias outside the curated ten
        engine.DEFAULT_PII_MODEL,  # a PII model on the NER field
        engine.DEFAULT_ZERO_SHOT_MODEL,
        "openai/privacy-filter",
        "attacker/ner-model",
    ],
)
def test_ner_rejects_every_other_model(name) -> None:
    with pytest.raises(ServiceError) as excinfo:
        _ner(model_name=name)
    assert excinfo.value.kind == "validation"
    assert "not an allowed clinical NER model" in str(excinfo.value)


@pytest.mark.parametrize(
    "alias", sorted(m.alias for m in engine.ZERO_SHOT_MODELS.values())
)
def test_zero_shot_accepts_every_curated_alias(alias) -> None:
    assert _zero_shot(model_name=alias) == {"entities": []}


@pytest.mark.parametrize(
    "name",
    [
        "zeroshot_disease_large_459m",  # a zero-shot registry alias outside the curated ten
        "OpenMed/OpenMed-ZeroShot-NER-Disease-Small-166M",  # a curated alias's repo id
        engine.DEFAULT_ZERO_SHOT_MODEL.upper(),
        engine.DEFAULT_NER_MODEL,  # a token-classification model forced into GLiNER
        engine.DEFAULT_PII_MODEL,
        "attacker/gliner-model",
    ],
)
def test_zero_shot_rejects_every_other_model(name) -> None:
    with pytest.raises(ServiceError) as excinfo:
        _zero_shot(model_name=name)
    assert excinfo.value.kind == "validation"
    assert "not an allowed zero-shot model" in str(excinfo.value)


def test_allowlist_rejection_does_not_echo_the_model_name() -> None:
    # The allowlist message names the rule and the env var, never the value (nor the
    # allowed ids) — a model_name field is one stray paste away from holding note text.
    for call in (*_PII_CALLS.values(), _ner, _zero_shot):
        for name in (_SENTINEL, f"org/{_SENTINEL}"):
            with pytest.raises(ServiceError) as excinfo:
                call(model_name=name)
            message = str(excinfo.value)
            assert "SECRET" not in message
            assert validation.EXTRA_MODELS_ENV in message


def test_allowlists_admit_only_the_curated_names_of_openmeds_registry() -> None:
    # Of openmed's ~3,300 registry aliases and their HF model ids, each capability admits
    # exactly its curated names; every other entry — the privacy-filter and -mlx ones,
    # the other domains', the other sizes' — fails before the engine. (Before the
    # allowlist, POST /zero-shot resolved ANY of them and forced it into GLiNER.)
    # Registry metadata only — no model download.
    import openmed

    catalog = openmed.get_all_models()  # dict[alias -> ModelInfo]
    names = set(catalog) | {info.model_id for info in catalog.values()}
    extras = validation.EXTRA_MODELS & names  # empty: tests/conftest.py scrubs the knob
    expected = {
        validation.PiiModelName: {
            engine.DEFAULT_PII_MODEL,
            engine.DEFAULT_PII_MLX_MODEL,
        },
        validation.NerModelName: {m.alias for m in engine.NER_MODELS.values()},
        validation.ZeroShotModelName: {
            m.alias for m in engine.ZERO_SHOT_MODELS.values()
        },
    }
    for field_type, curated in expected.items():
        adapter = TypeAdapter(field_type)
        admitted = set()
        for name in names:
            try:
                adapter.validate_python(name)
            except ValidationError:
                continue
            admitted.add(name)
        assert admitted == curated | extras


def test_default_pii_mlx_model_is_a_registry_model() -> None:
    # The two default PII ids stay openmed registry model ids. That buys a registry hash
    # only on openmed's HF path (core/models.py::prepare_model_reference): its MLX path —
    # the only one that can load the -mlx build — verifies no model, this one or the
    # default (see engine.DEFAULT_PII_MLX_MODEL). An unregistered id is verified nowhere.
    import openmed

    model_ids = {info.model_id for info in openmed.get_all_models().values()}
    assert engine.DEFAULT_PII_MODEL in model_ids
    assert engine.DEFAULT_PII_MLX_MODEL in model_ids


def test_no_default_allowlist_entry_is_a_privacy_filter_model() -> None:
    # openmed loads a name its privacy-filter predicate matches with
    # trust_remote_code=True. No name the allowlists admit by default may match it —
    # nor any registry model id a curated alias resolves to. Uses openmed's own
    # (private) predicate, so a widened prefix list upstream fails here.
    import openmed
    from openmed.core.pii import _looks_like_privacy_filter_identifier

    catalog = openmed.get_all_models()
    admitted = (
        validation.PII_MODEL_NAMES
        | validation.NER_MODEL_NAMES
        | validation.ZERO_SHOT_MODEL_NAMES
    ) - validation.EXTRA_MODELS
    resolved = {catalog[name].model_id for name in admitted if name in catalog}
    matches = sorted(
        n for n in admitted | resolved if _looks_like_privacy_filter_identifier(n)
    )
    assert not matches, matches


# --- OPENMED_STUDIO_EXTRA_MODELS (the operator's escape hatch) --------------


def test_extra_models_env_parsing(monkeypatch) -> None:
    # Comma-separated, whitespace-trimmed, empty entries dropped (a trailing comma is
    # harmless); unset or empty means no extras. Mirrors the MAX_TEXT_LENGTH knob test:
    # the reader is called directly, since the module ran it once at import.
    env = validation.EXTRA_MODELS_ENV
    monkeypatch.delenv(env, raising=False)
    assert validation._extra_models() == frozenset()
    monkeypatch.setenv(env, "")
    assert validation._extra_models() == frozenset()
    monkeypatch.setenv(env, " , ,")
    assert validation._extra_models() == frozenset()
    monkeypatch.setenv(env, " acme/pii-model , ,zeroshot_extra,, Acme/Other-Model ,")
    assert validation._extra_models() == {
        "acme/pii-model",
        "zeroshot_extra",
        "Acme/Other-Model",  # casing kept: the allowlist matches exactly
    }


@pytest.mark.parametrize(
    "raw", ["acme/ok,../escape", "acme/ok, two words", "a/b/c", ".venv", "~/model"]
)
def test_extra_models_env_rejects_a_malformed_entry(monkeypatch, raw) -> None:
    # A malformed entry fails loudly, naming the env var, rather than being dropped
    # (which would silently refuse the model the operator meant to allow).
    monkeypatch.setenv(validation.EXTRA_MODELS_ENV, raw)
    with pytest.raises(ValueError, match=validation.EXTRA_MODELS_ENV):
        validation._extra_models()


def _import_with_extra_models(value: str, code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter with OPENMED_STUDIO_EXTRA_MODELS=value.

    A subprocess, because the env var is read once at import: a fresh process proves
    that wiring without reloading modules other tests hold references into.
    """
    env = {**os.environ, validation.EXTRA_MODELS_ENV: value}
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_extra_models_are_read_at_import_and_allowed_on_every_field() -> None:
    code = """
from openmed_studio import main, validation

extra = "Acme/Custom-Model"
assert validation.EXTRA_MODELS == {extra, "other/model"}, validation.EXTRA_MODELS
payloads = {
    validation.ExtractRequest: {"text": "x"},
    validation.DeidentifyRequest: {"text": "x"},
    validation.DeidentifyBatchRequest: {"items": ["x"]},
    validation.AnonymizePolicyRequest: {"text": "x", "policy": "hipaa_safe_harbor"},
    validation.NerRequest: {"text": "x"},
    validation.ZeroShotRequest: {"text": "x", "labels": ["Problem"]},
    main.CompatExtractRequest: {"text": "x"},
    main.CompatDeidentifyRequest: {"text": "x"},
}
for model, body in payloads.items():
    assert model.model_validate({**body, "model_name": extra}).model_name == extra
    try:
        model.model_validate({**body, "model_name": extra.lower()})
    except Exception:
        pass
    else:
        raise AssertionError(f"{model.__name__} accepted a case-variant extra")
print("OK")
"""
    result = _import_with_extra_models(" Acme/Custom-Model, other/model ,", code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_malformed_extra_models_stop_the_app_at_import() -> None:
    # Both surfaces import validation at startup (the UI via service, the API via
    # main), so a bad entry stops the app there instead of being ignored.
    result = _import_with_extra_models(
        "acme/ok,../escape", "import openmed_studio.main"
    )
    assert result.returncode != 0
    assert validation.EXTRA_MODELS_ENV in result.stderr
    assert "ValueError" in result.stderr


def test_openapi_lists_the_curated_model_names_but_no_extra() -> None:
    # A 422 names no allowed id, so the schema is where an API caller discovers them:
    # each capability's model_name lists exactly its curated names, in description and
    # examples — and never the operator's extras (a fresh process, since the extras are
    # read at import).
    code = """
import json
from openmed_studio import engine, main

spec = main.create_app().openapi()
schemas = spec["components"]["schemas"]
curated = {
    "ExtractRequest": [engine.DEFAULT_PII_MODEL, engine.DEFAULT_PII_MLX_MODEL],
    "NerRequest": [m.alias for m in engine.NER_MODELS.values()],
    "ZeroShotRequest": [m.alias for m in engine.ZERO_SHOT_MODELS.values()],
}
for model, names in curated.items():
    field = schemas[model]["properties"]["model_name"]
    assert field["examples"] == names, (model, field)
    assert all(name in field["description"] for name in names), model
    assert "OPENMED_STUDIO_EXTRA_MODELS" in field["description"], model
    assert "enum" not in json.dumps(field), model
assert "Acme/Secret-Extra" not in json.dumps(spec)
print("OK")
"""
    result = _import_with_extra_models("Acme/Secret-Extra", code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


# The operator knobs tests/conftest.py scrubs before anything imports openmed_studio.
_SCRUBBED_KNOBS = {
    "OPENMED_STUDIO_EXTRA_MODELS": "acme/ok,../escape",  # malformed: stops collection
    "OPENMED_STUDIO_MAX_TEXT_LENGTH": "1000",
    "OPENMED_STUDIO_API_KEY": "exported-in-the-shell",
    "OPENMED_STUDIO_COMPAT": "1",
    "OPENMED_STUDIO_PRELOAD": "1",
}


def test_operator_knobs_are_absent_and_their_defaults_hold() -> None:
    # What the fast suite assumes: none of the scrubbed knobs is set, so the import-time
    # ones took their defaults. The knob list here must be conftest's, or a knob added there
    # would never be exercised by the subprocess test below.
    import conftest

    assert set(_SCRUBBED_KNOBS) == set(conftest.SCRUBBED_KNOBS)
    assert not set(_SCRUBBED_KNOBS) & set(os.environ)
    assert validation.EXTRA_MODELS == frozenset()
    assert validation.MAX_TEXT_CHARS == 50_000


def test_conftest_scrubs_operator_knobs_before_import() -> None:
    # Export every scrubbed knob (a malformed extra included, which fails collection when
    # read) and run the check above in a fresh pytest: it passes only if conftest.py pops
    # them before any test module imports openmed_studio.
    env = {**os.environ, **_SCRUBBED_KNOBS}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_validation.py::"
            "test_operator_knobs_are_absent_and_their_defaults_hold",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


# --- zero-shot (GLiNER) request guards --------------------------------------

_ZERO_SHOT_MODEL = "zeroshot_disease_small_166m"


def test_zero_shot_accepts_valid_request() -> None:
    assert service.extract_zero_shot(
        ENGINE, "x", model_name=_ZERO_SHOT_MODEL, labels=["Problem", "Treatment"]
    ) == {"entities": []}


def test_zero_shot_normalizes_and_dedups_labels() -> None:
    # strip each label, drop blanks, and dedup case-insensitively (first spelling wins).
    req = validation.ZeroShotRequest.model_validate(
        {
            "text": "x",
            "model_name": _ZERO_SHOT_MODEL,
            "labels": ["Problem", " Problem ", "problem", "", "Treatment"],
        }
    )
    assert req.labels == ["Problem", "Treatment"]


def test_zero_shot_rejects_missing_model_name() -> None:
    with pytest.raises(ServiceError):
        service.extract_zero_shot(ENGINE, "x", labels=["Problem"])


def test_zero_shot_rejects_empty_labels() -> None:
    with pytest.raises(ServiceError):
        service.extract_zero_shot(ENGINE, "x", model_name=_ZERO_SHOT_MODEL, labels=[])


def test_zero_shot_rejects_all_blank_labels() -> None:
    # A list that strips down to nothing is as empty as [].
    with pytest.raises(ServiceError):
        service.extract_zero_shot(
            ENGINE, "x", model_name=_ZERO_SHOT_MODEL, labels=["  ", ""]
        )


def test_zero_shot_rejects_too_many_labels() -> None:
    over = [f"label{i}" for i in range(validation.MAX_ZERO_SHOT_LABELS + 1)]
    with pytest.raises(ServiceError):
        service.extract_zero_shot(ENGINE, "x", model_name=_ZERO_SHOT_MODEL, labels=over)


def test_zero_shot_rejects_overlong_label_without_echoing_it() -> None:
    # The over-length label could carry pasted PHI, so the rejection names the cap, not it.
    secret = "S" + "SECRET-9999" * 20  # > MAX_ZERO_SHOT_LABEL_CHARS
    with pytest.raises(ServiceError) as excinfo:
        service.extract_zero_shot(
            ENGINE, "x", model_name=_ZERO_SHOT_MODEL, labels=[secret]
        )
    assert secret not in str(excinfo.value)


def test_zero_shot_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ServiceError):
        service.extract_zero_shot(
            ENGINE,
            "x",
            model_name=_ZERO_SHOT_MODEL,
            labels=["Problem"],
            confidence_threshold=1.5,
        )


def test_zero_shot_rejects_unknown_field() -> None:
    with pytest.raises(ServiceError):
        service.extract_zero_shot(
            ENGINE, "x", model_name=_ZERO_SHOT_MODEL, labels=["Problem"], bogus=1
        )


def test_zero_shot_default_confidence_is_0_6() -> None:
    req = validation.ZeroShotRequest.model_validate(
        {"text": "x", "model_name": _ZERO_SHOT_MODEL, "labels": ["Problem"]}
    )
    assert req.confidence_threshold == 0.6


# --- policy-driven anonymization request guards -----------------------------


def test_anonymize_policy_accepts_valid_request() -> None:
    result = service.anonymize_policy(ENGINE, "x", policy="hipaa_safe_harbor")
    assert result["deidentified_text"] == "ok"
    assert (
        result["method"] == "hipaa_safe_harbor"
    )  # the policy name rides the method slot


def test_anonymize_policy_rejects_missing_policy() -> None:
    # policy is a REQUIRED closed Literal — an absent one must be rejected, not defaulted.
    with pytest.raises(ServiceError):
        service.anonymize_policy(ENGINE, "x")


def test_anonymize_policy_rejects_unknown_policy() -> None:
    # An unknown/typo'd policy is caught by the Policy Literal before the engine.
    with pytest.raises(ServiceError):
        service.anonymize_policy(ENGINE, "x", policy="not_a_real_policy")


def test_anonymize_policy_rejects_hidden_policies_phi_safely() -> None:
    # openmed ships these profiles, but they keep some detected identifiers verbatim, so the
    # Policy Literal leaves them out: each is refused by validation — before the engine, as
    # kind "validation", and without echoing the note (possible PHI).
    from openmed_studio.engine import HIDDEN_POLICIES

    secret = "SECRET-PATIENT-NAME-98765"
    for policy in sorted(HIDDEN_POLICIES):
        with pytest.raises(ServiceError) as excinfo:
            service.anonymize_policy(ENGINE, secret, policy=policy)
        assert excinfo.value.kind == "validation", policy
        assert secret not in str(excinfo.value), policy


def test_anonymize_policy_rejects_method_field() -> None:
    # There is deliberately no `method` on the request (the policy overrides it), so passing one
    # is an unknown field that extra="forbid" rejects — not a silently-ignored control.
    with pytest.raises(ServiceError):
        service.anonymize_policy(ENGINE, "x", policy="hipaa_safe_harbor", method="mask")


def test_anonymize_policy_rejects_keep_mapping_field() -> None:
    # keep_mapping isn't a request field either (the policy decides reversibility), so it's a
    # forbidden unknown field, not a user toggle.
    with pytest.raises(ServiceError):
        service.anonymize_policy(
            ENGINE, "x", policy="hipaa_safe_harbor", keep_mapping=False
        )


def test_anonymize_policy_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ServiceError):
        service.anonymize_policy(
            ENGINE, "x", policy="hipaa_safe_harbor", confidence_threshold=1.5
        )


def test_anonymize_policy_rejects_bad_lang() -> None:
    with pytest.raises(ServiceError):
        service.anonymize_policy(ENGINE, "x", policy="hipaa_safe_harbor", lang="zz")


def test_anonymize_policy_rejects_malformed_locale() -> None:
    with pytest.raises(ServiceError):
        service.anonymize_policy(
            ENGINE, "x", policy="gdpr_art9_health", locale="not a locale!"
        )


def test_batch_rejects_empty_items() -> None:
    with pytest.raises(ServiceError):
        service.deidentify_batch(ENGINE, [])


def test_batch_rejects_too_many_items() -> None:
    with pytest.raises(ServiceError):
        service.deidentify_batch(ENGINE, ["x"] * (validation.MAX_BATCH_ITEMS + 1))


def test_reidentify_rejects_oversize_mapping() -> None:
    big = {str(i): "y" for i in range(validation.MAX_MAPPING_ENTRIES + 1)}
    with pytest.raises(ServiceError):
        service.reidentify(ENGINE, "x", big)


# --- acceptances (validation passes, reaches the stub engine) ---------------


def test_accepts_lang_and_model_name() -> None:
    assert service.extract(
        ENGINE, "x", lang="fr", model_name=engine.DEFAULT_PII_MODEL
    ) == {"entities": []}


@pytest.mark.parametrize("lang", ["ar", "ja", "tr"])
def test_accepts_newly_added_languages(lang) -> None:
    # ar/ja/tr were added to Lang to match openmed 1.6.0's SUPPORTED_LANGUAGES; they
    # must now validate (they were rejected before, even though openmed ships models).
    assert service.extract(ENGINE, "x", lang=lang) == {"entities": []}


def test_accepts_date_controls() -> None:
    result = service.deidentify(ENGINE, "x", method="shift_dates", date_shift_days=180)
    assert result["method"] == "shift_dates"


def test_ner_accepts_valid_request() -> None:
    assert service.analyze(ENGINE, "x", model_name=_NER_MODEL) == {"entities": []}


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


def test_validation_lang_subset_of_openmed() -> None:
    # Every language the app offers must be one openmed actually supports, so the app
    # never rejects a language openmed ships a model for. Subset (not equality) lets the
    # app deliberately offer fewer than openmed's full set while still catching drift if
    # openmed ever drops one the app still lists.
    from openmed.core.pii_i18n import SUPPORTED_LANGUAGES

    assert set(typing.get_args(validation.Lang)) <= set(SUPPORTED_LANGUAGES)


def test_validation_ner_models_resolve_in_openmed() -> None:
    # Pin the curated NER catalog against openmed's live registry: every domain key is a
    # real category, every alias resolves in that category, and the metadata baked into
    # NerModel for the UI (recommended_confidence, entity_types) still matches the
    # registry — so a renamed alias, dropped category, or drifted metadata fails CI rather
    # than leaving the NER tab stale. Registry metadata only — no model download.
    import openmed

    from openmed_studio.engine import NER_MODELS

    catalog = openmed.get_all_models()  # dict[alias -> ModelInfo]
    categories = set(openmed.list_model_categories())
    for domain, model in NER_MODELS.items():
        assert domain in categories, f"{domain!r} is not an openmed category"
        info = catalog.get(model.alias)
        assert info is not None, (
            f"NER alias {model.alias!r} is not in openmed's registry"
        )
        assert info.category == domain, (
            f"{model.alias!r} is category {info.category!r}, expected {domain!r}"
        )
        assert info.recommended_confidence == model.recommended_confidence, (
            f"{model.alias!r} recommended_confidence drifted: registry "
            f"{info.recommended_confidence} != baked {model.recommended_confidence}"
        )
        assert set(info.entity_types) == set(model.entity_types), (
            f"{model.alias!r} entity_types drifted: registry {sorted(info.entity_types)} "
            f"!= baked {sorted(model.entity_types)}"
        )


def test_zero_shot_models_resolve_in_openmed() -> None:
    # Pin the curated zero-shot catalog against openmed's live registry the way the NER guard
    # does: every alias resolves, its baked recommended_confidence/entity_types still match,
    # and every label_domain is a real openmed label vocabulary. Unlike NER, we don't pin
    # info.category (zero-shot models bucket into only a few broad categories, not per-domain).
    # Registry/label metadata only — no model download, no gliner extra needed.
    import openmed
    from openmed.ner import available_domains

    from openmed_studio.engine import ZERO_SHOT_MODELS

    catalog = openmed.get_all_models()  # dict[alias -> ModelInfo]
    label_domains = set(available_domains())
    for domain, model in ZERO_SHOT_MODELS.items():
        info = catalog.get(model.alias)
        assert info is not None, (
            f"zero-shot alias {model.alias!r} is not in openmed's registry"
        )
        # Pin the one registry field the runtime path actually reads: extract_zero_shot
        # resolves alias -> info.model_id to build the infer request. Without this, an
        # openmed rename of .model_id would pass CI and only fail under --run-model.
        assert info.model_id, f"{model.alias!r} has no model_id in openmed's registry"
        assert info.recommended_confidence == model.recommended_confidence, (
            f"{model.alias!r} recommended_confidence drifted: registry "
            f"{info.recommended_confidence} != baked {model.recommended_confidence}"
        )
        assert set(info.entity_types) == set(model.entity_types), (
            f"{model.alias!r} entity_types drifted: registry {sorted(info.entity_types)} "
            f"!= baked {sorted(model.entity_types)}"
        )
        assert model.label_domain in label_domains, (
            f"{domain!r} label_domain {model.label_domain!r} is not an openmed label "
            f"vocabulary (available: {sorted(label_domains)})"
        )


def test_validation_policy_matches_openmed() -> None:
    # Keep the app's policy surface in sync with openmed's canonical policy set — the policy
    # analogue of test_validation_deidmethod_matches_openmed (registry metadata only, no model
    # load). Every canonical profile is either offered (Policy) or deliberately hidden
    # (HIDDEN_POLICIES), never both and never neither, so a profile openmed adds fails here
    # until someone decides which side it belongs on.
    from openmed.core.policy import PolicyName

    from openmed_studio.engine import HIDDEN_POLICIES

    offered = set(typing.get_args(validation.Policy))
    assert offered.isdisjoint(HIDDEN_POLICIES), sorted(offered & HIDDEN_POLICIES)
    assert offered | HIDDEN_POLICIES == {p.value for p in PolicyName}


def test_hidden_policies_are_exactly_those_that_keep_other() -> None:
    # HIDDEN_POLICIES is not a taste call: it is exactly the profiles whose catch-all OTHER
    # action is "keep". openmed files much of the PII model's vocabulary (license, tax and
    # employee IDs, employers, religion, ...) under OTHER, so such a profile passes those
    # identifiers through verbatim. Pinned against the live profiles, so an openmed change to
    # a profile's OTHER action, or a new leaky profile, fails CI and forces a human decision.
    from openmed.core.policy import PolicyName, load_policy

    from openmed_studio.engine import HIDDEN_POLICIES

    keeps_other = {
        p.value for p in PolicyName if load_policy(p.value).actions["OTHER"] == "keep"
    }
    assert HIDDEN_POLICIES == keeps_other, (
        "openmed changed which profiles keep OTHER-labelled identifiers verbatim; a "
        "human must decide. Newly leaky (hide: add to HIDDEN_POLICIES, drop from "
        f"Policy/POLICY_MODELS): {sorted(keeps_other - HIDDEN_POLICIES)}. No longer "
        "leaky (re-expose after a real run; their last descriptions are in commit "
        f"8e1e50f): {sorted(HIDDEN_POLICIES - keeps_other)}."
    )
    # ...and every OFFERED profile keeps no label at all. The offered descriptions (and the
    # engine.py comments) assume nothing is kept; a profile that starts keeping, say, DATE
    # needs a real-run review of its description, even if its OTHER action is still "mask".
    offered = set(typing.get_args(validation.Policy))
    keeps_any = {
        name: sorted(k for k, a in load_policy(name).actions.items() if a == "keep")
        for name in offered
    }
    assert not any(keeps_any.values()), (
        "an offered profile now keeps labels verbatim; review its description with a real "
        f"run (and hide it if it leaks identifiers): { {n: k for n, k in keeps_any.items() if k} }"
    )


def test_policy_models_resolve_in_openmed() -> None:
    # Pin the curated policy catalog against openmed's live registry the way the NER/zero-shot
    # guards do: every POLICY_MODELS entry names a real policy, and the behavioral flags baked for
    # the UI preview (default_action/keep_mapping/safety_sweep_mandatory) still match openmed's
    # loaded PolicyProfile — so a policy-schema change fails CI rather than leaving the preview
    # stale. Registry/profile metadata only — no model download.
    from openmed.core.policy import list_policies, load_policy

    from openmed_studio.engine import POLICY_MODELS

    available = set(list_policies())
    for label, model in POLICY_MODELS.items():
        assert model.name in available, (
            f"policy {model.name!r} ({label!r}) is not in openmed's registry"
        )
        profile = load_policy(model.name)
        # default_action is a str today but compare by value in case openmed makes it an enum.
        assert (
            getattr(profile.default_action, "value", profile.default_action)
            == model.default_action
        ), f"{model.name!r} default_action drifted"
        assert profile.keep_mapping == model.keep_mapping, (
            f"{model.name!r} keep_mapping drifted"
        )
        assert profile.safety_sweep_mandatory == model.safety_sweep_mandatory, (
            f"{model.name!r} safety_sweep_mandatory drifted"
        )

    # ...and the catalog is COMPLETE. The loop above only pins POLICY_MODELS ⊆ registry, and
    # test_validation_policy_matches_openmed pins Policy (with HIDDEN_POLICIES) against
    # PolicyName — so without this a policy moved into Policy would be accepted by
    # AnonymizePolicyRequest yet missing from the UI picker.
    # openmed 2.x took the built-ins from 10 to 19 in one release (and 2.5 added a 20th), so the
    # gap is not theoretical.
    assert {model.name for model in POLICY_MODELS.values()} == set(
        typing.get_args(validation.Policy)
    ), "POLICY_MODELS must surface every policy the Policy literal accepts"

    # ...and no description promises a re-identification key the profile does not keep. The
    # descriptions are hand-authored prose, so most of their accuracy can only be reviewed by a
    # human — but this one claim is safety-critical and mechanically checkable: telling a user a
    # policy is "reversible with a key" when keep_mapping is False means they may anonymize
    # believing they can get the original back. (openmed 2.x makes this easy to get wrong: two
    # of the four `replace`-based profiles offered, ZA POPIA and NG NDPA, keep NO mapping.)
    for label, model in POLICY_MODELS.items():
        # Match the CLAIM, not one blessed wording ("reversible with a key" / "reversible").
        # Strip "irreversible" first — it contains "reversible" as a substring.
        text = model.description.casefold().replace("irreversible", "")
        promises_key = "reversible" in text
        assert promises_key == model.keep_mapping, (
            f"{label!r} description and keep_mapping disagree about reversibility: "
            f"keep_mapping={model.keep_mapping}, description={model.description!r}"
        )
