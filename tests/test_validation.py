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
            ENGINE, "x", policy="gdpr_pseudonymization", locale="not a locale!"
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
    assert service.extract(ENGINE, "x", lang="fr", model_name="OpenMed/Some-Model") == {
        "entities": []
    }


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
    # Keep the app's Policy enum in sync with openmed's canonical policy set — the policy analogue
    # of test_validation_deidmethod_matches_openmed (registry metadata only, no model load).
    from openmed.core.policy import PolicyName

    assert set(typing.get_args(validation.Policy)) == {p.value for p in PolicyName}


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
