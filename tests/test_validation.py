"""Tests for the input guarantees the Pydantic request models enforce.

The request models validate every request in-process via ``openmed_studio.service``
(and, unchanged, serve as the FastAPI request bodies); the seam raises ``ServiceError``
on rejection (validation runs before the stub engine is reached). This pins the
text/batch/mapping caps, the value/enum/format checks, the
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


def test_every_model_name_field_applies_the_dot_rule() -> None:
    # One rule on every surface: each request model that declares a model_name — the
    # seam's and the /compat bodies in main.py — must route it through
    # _check_model_name, so a new model can't reopen "../x" with a bare `str` field.
    from pydantic import BaseModel, ValidationError

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
    assert {"ExtractRequest", "NerRequest", "CompatExtractRequest"} <= found
    for model in models:
        with pytest.raises(ValidationError) as excinfo:
            model.model_validate({"model_name": "../x"})
        messages = {e["loc"]: e["msg"] for e in excinfo.value.errors()}
        assert "must not start with '.'" in messages.get(("model_name",), ""), (
            f"{model.__name__}.model_name skips the dot rule"
        )


def test_accepts_every_openmed_registry_model_name() -> None:
    # The dot rule must cost no real model: every alias and HF model id in openmed's
    # registry, plus the ids engine.py bakes, passes unchanged. (The Hub itself forbids
    # a leading "." in a repo id.) Registry metadata only — no model download.
    import openmed

    from openmed_studio import engine

    catalog = openmed.get_all_models()  # dict[alias -> ModelInfo]
    names = set(catalog) | {info.model_id for info in catalog.values()}
    names |= {engine.DEFAULT_PII_MODEL, engine.DEFAULT_NER_MODEL}
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
