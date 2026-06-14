# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`openmed-explore` is a collection of **runnable examples** for the [OpenMed](https://openmed.life/docs/)
clinical-NLP library (PyPI package `openmed`). It is *not* the library itself — the
library source lives at `github.com/maziyarpanahi/openmed`. Examples live under `examples/`;
each is a standalone script demonstrating one capability.

## Working with Python

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

## Commands

This is a [uv](https://docs.astral.sh/uv/) **non-package** project (`[tool.uv] package = false`
in `pyproject.toml`) — uv installs the declared dependencies into `.venv` but builds no wheel.

```bash
# Run an example. uv auto-creates .venv and installs openmed[hf] from pyproject.toml.
uv run python examples/deidentify_pii.py

# Re-run fully offline once the model is cached (skips HF Hub network checks + token warning).
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python examples/deidentify_pii.py

# Swap the portable Torch/Transformers backend for Apple's native MLX backend (Apple Silicon).
uv sync --extra mlx
```

Lint, format, and type-check with the project-pinned tools (configured under
`[tool.ruff.lint]` and `[tool.ty.environment]`):

```bash
uv run ruff check .            # lint
uv run ruff check --fix .      # lint + auto-fix
uv run ruff format .           # format
uv run ty check                # type-check (resolves openmed types from .venv)
```

Run the test suite with pytest (configured under `[tool.pytest.ini_options]`):

```bash
uv run pytest                  # fast tests only; model tests are skipped
uv run pytest --run-model      # also run the tests that load the OpenMed PII model
```

Test layout (`tests/`): fast, no-model tests live in `test_pii_pure.py`; tests that load the
model are in `test_pii_model.py`, all marked `@pytest.mark.model` and **skipped by default**.
The `--run-model` opt-in is wired via `pytest_addoption` + `pytest_collection_modifyitems` in
`conftest.py`, which also provides the session-scoped `loader` fixture (model loads once) and a
`note` fixture. The `shift_dates` upstream bug (see Known gotchas) is captured as a `strict=True`
`xfail` — if it ever xpasses, the suite fails, signalling the bug was fixed.

Note: `ty` is configured to target Python 3.10 (the minimum supported). openmed ships inline
type hints — e.g. `deidentify(method=...)` expects the `Literal` of the five method names — so
keep the example's `DeidMethod` alias in sync with those.

## How the examples work

- **Backend:** the default dependency is `openmed[hf]` (Hugging Face / PyTorch), which runs
  everywhere (CPU, CUDA, Apple MPS). The `mlx` extra is Apple-Silicon-only and expects an
  MLX-packaged model (a repo id ending in `-mlx`).
- **Model download:** the first run pulls a model from the HF Hub and caches it under
  `~/.cache/openmed`; later runs are offline. The default PII model is the small
  `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` (~44M params).
- **Model reuse:** construct one `ModelLoader()` and pass `loader=` to every `extract_pii` /
  `deidentify` call so the model loads once instead of per-call. This is the pattern in
  `examples/deidentify_pii.py` and the documented best practice.
- **Python:** `requires-python = ">=3.10"`; examples are verified on 3.11, but uv may pick a
  newer interpreter (e.g. 3.13) for `.venv`.

## OpenMed PII API (verified against installed v1.5.5)

Top-level imports: `from openmed import extract_pii, deidentify, reidentify, ModelLoader, OpenMedConfig`.

- `extract_pii(text, model_name=<default>, confidence_threshold=0.5, use_smart_merging=True, lang="en", *, loader=None)`
  returns PII entities, each with `.label`, `.text`, `.start`, `.end`, `.confidence`.
  Labels are **lowercase** (`first_name`, `last_name`, `date`, `ssn`, `phone_number`, …).
- `deidentify(text, method="mask", ..., keep_mapping=False, *, consistent=False, seed=None, locale=None, loader=None)`
  returns a `DeidentificationResult` with `.deidentified_text`, `.pii_entities`, `.mapping`.
  Methods: `mask`, `remove`, `replace` (Faker surrogates — use `consistent=True, seed=N` for
  determinism), `hash`, `shift_dates`.
- `reidentify(deidentified_text, mapping)` → original text (use with `deidentify(..., keep_mapping=True)`).

## Known gotchas

- **`shift_dates` is a no-op with the default model.** `openmed/core/pii.py:905` matches the
  literal label `"DATE"`, but the default model emits lowercase `"date"`, so dates get masked
  instead of shifted (no parameter fixes this with that model — needs a model that emits
  canonical `"DATE"` labels). `examples/deidentify_pii.py` detects this at runtime and prints
  a note rather than silently appearing to work.
- **`reidentify()` mis-restores overlapping mapping keys.** It applies `str.replace`
  per entry, so a key that is a prefix of another (e.g. `ALIAS_1` vs `ALIAS_10`)
  corrupts the longer one. `tests/test_pii_pure.py` captures this as a `strict` xfail.
- **pysbd `SyntaxWarning`s** (a transitive dependency) appear on Python ≥3.12 from its regex
  literals; they are harmless. Examples silence them with
  `warnings.filterwarnings("ignore", category=SyntaxWarning)` *before* importing `openmed`.
- The `.venv` here is ~600 MB (Torch + Transformers) and is gitignored.
