"""Tests for the FastAPI service (``openmed_studio.main``).

These inject a model-free stub engine via FastAPI ``dependency_overrides``, so they
validate the HTTP transport layer — routing to each ``service.*`` function, the
``ServiceError.kind`` -> HTTP-status mapping, the uniform error envelope, PHI-safe 422s,
``X-API-Key`` auth, and the opt-in ``/compat`` surface — with no ``--run-model`` and no
network. The seam's own behavior (validation rules, adapters, taxonomy) is covered in
``test_service.py``; here we only pin what the HTTP layer adds on top.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from openmed_studio import HIDDEN_POLICIES, __version__
from openmed_studio.main import API_KEY_ENV, COMPAT_ENV, app, create_app, get_engine

if TYPE_CHECKING:
    from openmed import ModelLoader

# Policies whose profile keeps a re-identification mapping (openmed ORs the profile's own
# keep_mapping) — the two POLICY_MODELS entries with keep_mapping=True. The stub mirrors
# that so the anonymize-policy tests can assert both branches.
_REVERSIBLE_POLICIES = {"gdpr_art9_health", "china_pipl"}


class _StubEngine:
    """A model-free stand-in for PIIEngine covering every service call surface."""

    model_name = "stub-model"
    backend: str | None = None
    is_loaded = True

    def extract(self, _text, **_):
        return [
            SimpleNamespace(
                label="first_name", text="John", start=0, end=4, confidence=0.99
            )
        ]

    def analyze(self, _text, **_):
        return [
            SimpleNamespace(
                label="DISEASE", text="diabetes", start=0, end=8, confidence=0.97
            )
        ]

    def extract_zero_shot(self, _text, **_):
        # openmed's zero-shot Entity exposes .score, not .confidence — the adapter maps it.
        return [
            SimpleNamespace(
                label="Problem", text="diabetes", start=0, end=8, score=0.88
            )
        ]

    def deidentify(self, text, *, keep_mapping=False, policy=None, **_):
        if text == "BAD":  # lets the batch test exercise per-note isolation
            raise ValueError("bad note content")
        reversible = keep_mapping or policy in _REVERSIBLE_POLICIES
        mapping = {"[first_name]": "John"} if reversible else None
        return SimpleNamespace(
            deidentified_text="[first_name] A. Doe",
            pii_entities=[
                SimpleNamespace(
                    label="first_name", text="John", start=0, end=4, confidence=0.99
                )
            ],
            mapping=mapping,
        )

    def reidentify(self, deidentified_text, mapping):
        for key, value in mapping.items():
            deidentified_text = deidentified_text.replace(key, value)
        return deidentified_text


class _CodedRuntimeError(RuntimeError):
    """Stand-in for an openmed taxonomy error that is a RuntimeError carrying ``.code``."""

    def __init__(self, code: str, message: str = "openmed internal failure") -> None:
        super().__init__(message)
        self.code = code


class _RaisingEngine(_StubEngine):
    """Stub whose model call raises, to exercise the API error paths.

    Only ``extract`` is overridden — every error-path test drives the taxonomy through
    ``POST /pii/extract`` (all seven routes share the same ``service._run`` translation, so
    one endpoint covers the mapping). Add more overrides here only if a test needs them.
    """

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def extract(self, _text, **_):
        raise self._exc


@contextlib.contextmanager
def _client(engine: object | None = None, target=app):
    target.dependency_overrides[get_engine] = lambda: engine or _StubEngine()
    try:
        with TestClient(target) as test_client:
            yield test_client
    finally:
        target.dependency_overrides.clear()


@pytest.fixture
def client():
    with _client() as test_client:
        yield test_client


# --- /health -----------------------------------------------------------------


def test_health_ok(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "openmed-studio"
    assert body["version"] == __version__
    assert body["model"] == "stub-model"
    assert body["backend"] == "auto"  # stub engine has backend=None -> "auto"
    assert body["max_text_chars"] == 50_000  # OPENMED_STUDIO_MAX_TEXT_LENGTH default
    assert body["model_loaded"] is True
    assert body["auth_required"] is False  # no OPENMED_STUDIO_API_KEY set in tests
    assert body["working_directory_clean"] is True  # tests run from the repo root


def test_health_reports_configured_backend() -> None:
    engine = SimpleNamespace(model_name=None, backend="mlx", is_loaded=False)
    with _client(engine) as override:
        assert override.get("/health").json()["backend"] == "mlx"


def test_health_flags_a_poisoned_working_directory_without_naming_it(
    monkeypatch, tmp_path
) -> None:
    # The guard 503s every model call while a CWD entry shadows a model; /health says so
    # (checked per request) as a bare boolean — the entry and the directory stay in the
    # server log. It stays a 200 "ok": the process is up, and a restart from the same
    # directory wouldn't help.
    monkeypatch.chdir(tmp_path)
    with _client() as override:
        assert override.get("/health").json()["working_directory_clean"] is True
        (tmp_path / "OpenMed").mkdir()
        resp = override.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["working_directory_clean"] is False
    assert "OpenMed" not in resp.text
    assert tmp_path.name not in resp.text


def test_create_app_warns_once_about_a_poisoned_working_directory(
    monkeypatch, tmp_path, caplog
) -> None:
    # Logged at startup, before any request — without OPENMED_STUDIO_PRELOAD the guard
    # would otherwise stay silent until the first model call 503s.
    import logging

    monkeypatch.chdir(tmp_path)
    (tmp_path / "openai").mkdir()
    with caplog.at_level(logging.WARNING, logger="openmed_studio"):
        create_app()
    warnings = [r for r in caplog.records if "working directory" in r.getMessage()]
    assert len(warnings) == 1
    assert "'openai'" in warnings[0].getMessage()


def test_create_app_is_quiet_about_a_clean_working_directory(
    monkeypatch, tmp_path, caplog
) -> None:
    import logging

    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING, logger="openmed_studio"):
        create_app()
    assert not [r for r in caplog.records if "working directory" in r.getMessage()]


# --- success paths: each route reaches the right service function ------------


def test_extract_ok(client) -> None:
    resp = client.post("/pii/extract", json={"text": "Patient John."})
    assert resp.status_code == 200
    assert resp.json()["entities"][0]["label"] == "first_name"


def test_ner_ok(client) -> None:
    resp = client.post(
        "/ner",
        json={"text": "diabetes", "model_name": "disease_detection_superclinical_141m"},
    )
    assert resp.status_code == 200
    assert resp.json()["entities"][0]["label"] == "DISEASE"  # NER labels UPPERCASE


def test_zero_shot_maps_score_to_confidence(client) -> None:
    resp = client.post(
        "/zero-shot",
        json={
            "text": "diabetes",
            "model_name": "zeroshot_disease_small_166m",
            "labels": ["Problem"],
        },
    )
    assert resp.status_code == 200
    entity = resp.json()["entities"][0]
    assert entity["label"] == "Problem"  # arbitrary user label passes through
    assert entity["confidence"] == 0.88  # openmed's .score surfaced as confidence


def test_deidentify_with_mapping(client) -> None:
    resp = client.post(
        "/pii/deidentify", json={"text": "John Doe", "keep_mapping": True}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["deidentified_text"] == "[first_name] A. Doe"
    assert body["method"] == "mask"
    assert body["mapping"] == {"[first_name]": "John"}


def test_deidentify_without_mapping_is_null(client) -> None:
    resp = client.post("/pii/deidentify", json={"text": "John Doe"})
    assert resp.json()["mapping"] is None


def test_anonymize_policy_masking_has_no_mapping(client) -> None:
    resp = client.post(
        "/pii/anonymize-policy",
        json={"text": "John Doe", "policy": "hipaa_safe_harbor"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "hipaa_safe_harbor"  # policy name occupies the method slot
    assert body["mapping"] is None  # a masking policy is irreversible


def test_anonymize_policy_reversible_keeps_mapping(client) -> None:
    resp = client.post(
        "/pii/anonymize-policy",
        json={"text": "John Doe", "policy": "gdpr_art9_health"},
    )
    body = resp.json()
    assert body["method"] == "gdpr_art9_health"
    assert body["mapping"] == {"[first_name]": "John"}  # surrogate policy keeps a key


@pytest.mark.parametrize("policy", sorted(HIDDEN_POLICIES))
def test_anonymize_policy_rejects_hidden_policy_phi_safely(client, policy) -> None:
    # A profile openmed ships but the app hides (it keeps some identifiers verbatim) is
    # outside the Policy literal, so the route 422s before the engine, without echoing
    # the note.
    secret = "SENSITIVE-PATIENT-NAME-98765"
    resp = client.post("/pii/anonymize-policy", json={"text": secret, "policy": policy})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert secret not in resp.text


def test_anonymize_policy_rejects_method_field(client) -> None:
    # `method` is a forbidden extra on AnonymizePolicyRequest (the policy overrides it).
    resp = client.post(
        "/pii/anonymize-policy",
        json={"text": "x", "policy": "hipaa_safe_harbor", "method": "mask"},
    )
    assert resp.status_code == 422


def test_reidentify_ok(client) -> None:
    resp = client.post(
        "/pii/reidentify",
        json={
            "deidentified_text": "[first_name] A. Doe",
            "mapping": {"[first_name]": "John"},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "John A. Doe"


def test_batch_isolates_a_bad_note(client) -> None:
    resp = client.post("/pii/deidentify/batch", json={"items": ["good", "BAD"]})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[0]["ok"] is True
    assert results[0]["deidentified_text"] == "[first_name] A. Doe"
    assert results[1]["ok"] is False
    assert "bad note" in results[1]["error"]


# --- error taxonomy: ServiceError.kind -> HTTP status + envelope -------------


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (ValueError("bad option"), 400),  # kind="bad_options"
        (RuntimeError("model down"), 503),  # kind="unavailable"
        (OSError("io error"), 503),  # kind="unavailable"
        (ImportError("run uv sync --extra gliner"), 503),  # kind="dependency"
        (KeyError("leak-me"), 500),  # kind="internal" (catch-all)
        (_CodedRuntimeError("internal_error"), 500),  # openmed InternalError
    ],
)
def test_engine_failure_maps_to_status_and_envelope(exc, status) -> None:
    with _client(_RaisingEngine(exc)) as override:
        resp = override.post("/pii/extract", json={"text": "x"})
    assert resp.status_code == status
    error = resp.json()["error"]
    assert set(error) == {"code", "message", "details"}


@pytest.mark.parametrize(
    "exc", [KeyError("leak-me"), _CodedRuntimeError("internal_error", "leak-me")]
)
def test_internal_error_does_not_leak_raw_message(exc) -> None:
    with _client(_RaisingEngine(exc)) as override:
        resp = override.post("/pii/extract", json={"text": "x"})
    assert resp.status_code == 500
    assert (
        "leak-me" not in resp.text
    )  # the raw exception detail never reaches the client


def test_validation_error_is_phi_safe(client) -> None:
    # An out-of-range option triggers a 422; the request text (possible PHI) must not be
    # echoed back, and the envelope carries only type/loc/msg field errors.
    secret = "SENSITIVE-PATIENT-NAME-98765"
    resp = client.post(
        "/pii/deidentify", json={"text": secret, "confidence_threshold": 5.0}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert secret not in resp.text


def test_unknown_field_is_rejected(client) -> None:
    resp = client.post("/pii/extract", json={"text": "x", "bogus": 1})
    assert resp.status_code == 422


# --- model_name allowlists ----------------------------------------------------

_NOTE = "SENSITIVE-PATIENT-NAME-98765"

# One body per route family that takes a model_name (all but the model_name itself).
_MODEL_ROUTES = [
    ("/pii/extract", {"text": _NOTE}),
    ("/pii/deidentify", {"text": _NOTE}),
    ("/pii/deidentify/batch", {"items": [_NOTE]}),
    ("/pii/anonymize-policy", {"text": _NOTE, "policy": "hipaa_safe_harbor"}),
    ("/ner", {"text": _NOTE}),
    ("/zero-shot", {"text": _NOTE, "labels": ["Problem"]}),
]

# Names no route admits: two that openmed would load with trust_remote_code=True, an
# arbitrary Hub id, and a pasted "MRN" that must not come back in the response.
_DISALLOWED_MODELS = [
    "openai/privacy-filter",
    "OpenMed/Privacy-Filter-Multilingual",
    "attacker/model",
    "org/SECRET-MRN-4471",
]


def _assert_phi_safe_model_name_422(resp) -> None:
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert any(
        d["loc"][-1] == "model_name" and "is not an allowed" in d["msg"]
        for d in error["details"]
    )
    assert _NOTE not in resp.text
    assert "SECRET" not in resp.text


@pytest.mark.parametrize("model_name", _DISALLOWED_MODELS)
@pytest.mark.parametrize(("path", "body"), _MODEL_ROUTES)
def test_disallowed_model_name_is_a_phi_safe_422(
    client, path, body, model_name
) -> None:
    resp = client.post(path, json={**body, "model_name": model_name})
    _assert_phi_safe_model_name_422(resp)


@pytest.mark.parametrize("model_name", _DISALLOWED_MODELS)
@pytest.mark.parametrize("path", ["/compat/pii/extract", "/compat/pii/deidentify"])
def test_compat_disallowed_model_name_is_a_phi_safe_422(
    monkeypatch, path, model_name
) -> None:
    # /compat relaxes unknown fields for OpenMed-REST parity, but not model_name: it
    # calls the engine directly, so it takes the same PII allowlist.
    with _client(target=_compat_app(monkeypatch)) as override:
        resp = override.post(path, json={"text": _NOTE, "model_name": model_name})
    _assert_phi_safe_model_name_422(resp)


def test_default_pii_mlx_build_is_accepted(client) -> None:
    from openmed_studio.engine import DEFAULT_PII_MLX_MODEL

    resp = client.post(
        "/pii/extract", json={"text": "x", "model_name": DEFAULT_PII_MLX_MODEL}
    )
    assert resp.status_code == 200


@pytest.mark.parametrize("path", ["/pii/extract", "/compat/pii/extract"])
def test_local_model_path_is_a_503_that_names_neither_path_nor_model(
    monkeypatch, tmp_path, path
) -> None:
    # The engine refuses a model name that exists under the working directory; over HTTP
    # that is the generic 503 envelope — the path, the working directory and the model
    # name stay in the server log.
    from typing import cast

    import openmed

    from openmed_studio import PIIEngine

    def fail(*_args, **_kwargs):
        raise AssertionError("openmed was called")

    monkeypatch.setattr(openmed, "extract_pii", fail)
    engine = PIIEngine(loader=cast("ModelLoader", object()))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "OpenMed" / "OpenMed-PII-SuperClinical-Small-44M-v1").mkdir(
        parents=True
    )
    with _client(engine, target=_compat_app(monkeypatch)) as override:
        resp = override.post(path, json={"text": "x"})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "service_unavailable"
    for leak in ("OpenMed", str(tmp_path.name), "working directory"):
        assert leak not in resp.text


def test_ner_rejects_an_aliass_repo_id(client) -> None:
    # NER admits the curated aliases only, not their HF repo ids.
    resp = client.post(
        "/ner",
        json={
            "text": "x",
            "model_name": "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-141M",
        },
    )
    assert resp.status_code == 422


# --- auth (X-API-Key) --------------------------------------------------------


def test_missing_key_is_401_when_auth_enabled(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret")
    with _client() as override:
        resp = override.post("/pii/extract", json={"text": "x"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_correct_key_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret")
    with _client() as override:
        resp = override.post(
            "/pii/extract", json={"text": "x"}, headers={"X-API-Key": "secret"}
        )
    assert resp.status_code == 200


def test_wrong_key_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret")
    with _client() as override:
        resp = override.post(
            "/pii/extract", json={"text": "x"}, headers={"X-API-Key": "nope"}
        )
    assert resp.status_code == 401


def test_health_is_open_even_with_auth(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret")
    with _client() as override:
        resp = override.get("/health")
    assert resp.status_code == 200
    assert resp.json()["auth_required"] is True


# --- /compat (opt-in OpenMed-REST parity) ------------------------------------


def test_compat_absent_by_default(client) -> None:
    # The module-level app is built with compat off, so the routes 404.
    assert client.post("/compat/pii/extract", json={"text": "x"}).status_code == 404


def _compat_app(monkeypatch):
    monkeypatch.setenv(COMPAT_ENV, "1")
    return create_app()


def test_compat_extract_uses_openmed_shape(monkeypatch) -> None:
    with _client(target=_compat_app(monkeypatch)) as override:
        resp = override.post(
            "/compat/pii/extract", json={"text": "John", "keep_alive": "5m"}
        )
    assert resp.status_code == 200  # unknown `keep_alive` accepted (extra=ignore)
    entity = resp.json()["entities"][0]
    assert (
        entity["entity_type"] == "first_name"
    )  # openmed carries label AND entity_type
    assert "metadata" in entity


def test_compat_deidentify_echoes_original_and_counts(monkeypatch) -> None:
    with _client(target=_compat_app(monkeypatch)) as override:
        resp = override.post(
            "/compat/pii/deidentify", json={"text": "John Doe", "keep_mapping": True}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["original_text"] == "John Doe"  # upstream parity echoes the input
    assert body["num_entities_redacted"] == 1
    assert "timestamp" in body
    assert body["mapping"] == {"[first_name]": "John"}
    assert "redacted_text" in body["pii_entities"][0]


def test_compat_requires_auth(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret")
    with _client(target=_compat_app(monkeypatch)) as override:
        resp = override.post("/compat/pii/extract", json={"text": "x"})
    assert resp.status_code == 401


_COMPAT_PATHS = ["/compat/pii/extract", "/compat/pii/deidentify"]


@pytest.mark.parametrize(
    "lang",
    [
        # openmed defaults these to its privacy-filter model, which it swaps in for the
        # default English model and loads with trust_remote_code=True
        "fa",
        "sv",
        "ru",
        "zu",
        "EN",  # the Literal is exact, as on the primary routes
        "zz",
        "SECRET-4471",
    ],
)
@pytest.mark.parametrize("path", _COMPAT_PATHS)
def test_compat_rejects_a_lang_outside_the_apps_list(monkeypatch, path, lang) -> None:
    # /compat relaxes unknown fields for parity, but its lang is the same Lang Literal as
    # the primary routes': anything else is a PHI-safe 422 before the engine.
    with _client(target=_compat_app(monkeypatch)) as override:
        resp = override.post(path, json={"text": _NOTE, "lang": lang})
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert [d["loc"][-1] for d in error["details"]] == ["lang"]
    assert _NOTE not in resp.text
    assert "SECRET" not in resp.text


@pytest.mark.parametrize("path", _COMPAT_PATHS)
def test_compat_accepts_a_supported_lang(monkeypatch, path) -> None:
    captured: dict[str, object] = {}

    class _Recording(_StubEngine):
        def extract(self, _text, **kwargs):
            captured.update(kwargs)
            return super().extract(_text, **kwargs)

        def deidentify(self, text, **kwargs):
            captured.update(kwargs)
            return super().deidentify(text, **kwargs)

    with _client(_Recording(), target=_compat_app(monkeypatch)) as override:
        resp = override.post(path, json={"text": "Jean Dupont", "lang": "fr"})
    assert resp.status_code == 200
    assert captured["lang"] == "fr"


# --- tooling: the TestClient's HTTP backend ----------------------------------


def test_testclient_is_backed_by_httpx2() -> None:
    # starlette's TestClient subclasses httpx2.Client when httpx2 is installed, and
    # otherwise falls back to plain httpx with only a StarletteDeprecationWarning. That
    # fallback is quiet in the worst way: starlette types TestClient against httpx2
    # alone, so without it ty sees every client call here as Unknown and still passes.
    import httpx2

    assert issubclass(TestClient, httpx2.Client)
