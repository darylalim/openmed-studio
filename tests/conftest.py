"""Shared pytest configuration, fixtures, and the opt-in marker for model tests.

Tests that actually load the OpenMed PII model are marked ``@pytest.mark.model``.
They are skipped by default (so the suite stays fast) and enabled with::

    uv run pytest --run-model

This uses the canonical pytest pattern for optional slow tests: a custom CLI
option (``pytest_addoption``) plus a collection hook that skips the marked tests
unless the option is passed.
"""

from __future__ import annotations

import pytest


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
