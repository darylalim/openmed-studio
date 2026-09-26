"""Shared pytest configuration, fixtures, and the opt-in marker for model tests.

Tests that actually load the OpenMed PII model are marked ``@pytest.mark.model``.
They are skipped by default (so the suite stays fast) and enabled with::

    uv run pytest --run-model

This uses the canonical pytest pattern for optional slow tests: a custom CLI
option (``pytest_addoption``) plus a collection hook that skips the marked tests
unless the option is passed.
"""

from __future__ import annotations

import os

import pytest

# Operator knobs the fast suite pins at their defaults, scrubbed so a value exported in the
# developer's shell (to run the app) can't leak in. Two are read ONCE, at import, by
# openmed_studio.validation — a non-default allowlist or text cap means spurious failures,
# and a malformed extra a collection error — which is why this runs here: pytest imports
# this file before any test module, so before anything imports openmed_studio. The other
# three bite the API tests: main.py mounts /compat when its module-level `app =
# create_app()` runs, checks the key on every request, and runs the preload in the lifespan
# every TestClient enters — outside dependency_overrides, so it would load (and, uncached,
# download) the real model, which a fast test must never do. OPENMED_STUDIO_BACKEND stays:
# no fast test reads it unpatched, and it is a legitimate choice for --run-model.
# Tests that exercise a knob set it themselves (monkeypatch, or an explicit subprocess env
# whose {**os.environ, ...} copies this scrubbed environment);
# tests/test_validation.py::test_conftest_scrubs_operator_knobs_before_import pins it.
SCRUBBED_KNOBS = (
    "OPENMED_STUDIO_EXTRA_MODELS",
    "OPENMED_STUDIO_MAX_TEXT_LENGTH",
    "OPENMED_STUDIO_API_KEY",
    "OPENMED_STUDIO_COMPAT",
    "OPENMED_STUDIO_PRELOAD",
)
for _knob in SCRUBBED_KNOBS:
    os.environ.pop(_knob, None)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-model",
        action="store_true",
        default=False,
        help="run tests that load the OpenMed PII model (slow; downloads on first run)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-model"):
        return
    skip_model = pytest.mark.skip(
        reason="needs --run-model (loads the OpenMed PII model)"
    )
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip_model)


@pytest.fixture(scope="session")
def loader():
    """A single shared ``ModelLoader`` reused across all model tests.

    Session-scoped so the ~44M-parameter PII model is initialized at most once
    for the whole test run (the documented pipeline-reuse best practice).

    Built via ``PIIEngine().loader`` (not a bare ``ModelLoader()``) so the model
    tests exercise the app's real loader construction, including the
    ``torch_attention_backend="eager"`` pin. On openmed 2.x that pin is
    belt-and-braces: openmed's ``"auto"`` requests no attention implementation, so
    a bare loader lands on eager too and loads fine — but the pin is what the app
    ships, so it is what these tests load under (see ``PIIEngine.loader`` / "Known
    gotchas").

    It calls ``PIIEngine()`` directly, not ``service.build_engine()``, so it does
    not read ``OPENMED_STUDIO_BACKEND``: openmed picks the backend itself (MLX on
    Apple Silicon when the ``mlx`` extra is installed, else Hugging Face).
    """
    from openmed_studio import PIIEngine

    return PIIEngine().loader


@pytest.fixture
def note() -> str:
    """A synthetic clinical note. Every identifier is fabricated."""
    return (
        "Patient: John A. Doe (MRN: 1234567). DOB: 01/15/1970. "
        "Seen on 03/22/2024 by Dr. Emily Carter at Springfield General Hospital. "
        "Contact: john.doe@example.com, phone (415) 555-0137. "
        "SSN: 123-45-6789. Address: 742 Evergreen Terrace, Springfield, IL 62704."
    )
