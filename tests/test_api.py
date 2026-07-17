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

import pytest
from fastapi.testclient import TestClient

from openmed_studio import __version__
from openmed_studio.main import API_KEY_ENV, COMPAT_ENV, app, create_app, get_engine

# Policies whose profile keeps a re-identification mapping (openmed ORs the profile's own
# keep_mapping). The stub mirrors that so the anonymize-policy tests can assert both branches.
_REVERSIBLE_POLICIES = {
    "gdpr_pseudonymization",
    "gdpr_art9_health",
    "canada_pipeda",
    "uk_ico_anonymisation",
}


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


def test_health_reports_configured_backend() -> None:
    engine = SimpleNamespace(model_name=None, backend="mlx", is_loaded=False)
    with _client(engine) as override:
        assert override.get("/health").json()["backend"] == "mlx"


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
        json={"text": "John Doe", "policy": "gdpr_pseudonymization"},
    )
    body = resp.json()
    assert body["method"] == "gdpr_pseudonymization"
    assert body["mapping"] == {"[first_name]": "John"}  # surrogate policy keeps a key


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
    ],
)
def test_engine_failure_maps_to_status_and_envelope(exc, status) -> None:
    with _client(_RaisingEngine(exc)) as override:
        resp = override.post("/pii/extract", json={"text": "x"})
    assert resp.status_code == status
    error = resp.json()["error"]
    assert set(error) == {"code", "message", "details"}


def test_internal_error_does_not_leak_raw_message() -> None:
    with _client(_RaisingEngine(KeyError("leak-me"))) as override:
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
