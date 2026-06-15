# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`openmed-studio` is a **clinical-NLP application** built on the
[OpenMed](https://openmed.life/docs/) clinical-NLP library (PyPI package `openmed`). It is
*not* the library itself — the library source lives at `github.com/maziyarpanahi/openmed`.
The aim is an app that surfaces OpenMed's full capability set (clinical NER, PII/PHI
de-identification, anonymization, zero-shot extraction). **Today it implements PII/PHI
de-identification only**; the other capabilities are the roadmap. The project is a
[FastAPI](https://fastapi.tiangolo.com/) service in `openmed_studio/` (a reusable
framework-free `PIIEngine` + thin HTTP endpoints); `examples/deidentify_pii.py` remains as a
library-level demo of the same OpenMed calls.

## Working with Python

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

## Commands

This is a [uv](https://docs.astral.sh/uv/) **non-package** project (`[tool.uv] package = false`
in `pyproject.toml`) — uv installs the declared dependencies into `.venv` but builds no wheel.

```bash
# Run the de-identification demo. uv auto-creates .venv and installs deps from pyproject.toml.
uv run python examples/deidentify_pii.py

# Serve the de-identification API (interactive docs at http://127.0.0.1:8080/docs).
# Set OPENMED_STUDIO_API_KEY to require an X-API-Key header on /pii/* (unset = open, local-only).
uv run uvicorn openmed_studio.main:app --port 8080   # or: uv run python -m openmed_studio

# Re-run fully offline once the model is cached (skips HF Hub network checks + token warning).
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python examples/deidentify_pii.py

# Swap the portable Torch/Transformers backend for Apple's native MLX backend (Apple Silicon).
uv sync --extra mlx
# Force a backend for the service (default unset = openmed auto-detects: MLX on Apple Silicon
# when the mlx extra is installed, else HuggingFace). "mlx" fails loudly if MLX is unavailable.
OPENMED_STUDIO_BACKEND=mlx uv run uvicorn openmed_studio.main:app --port 8080
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

Test layout (`tests/`): fast, no-model tests live in `test_pii_pure.py`, `test_api.py`, and
`test_engine.py` (the API tests inject a stub engine via FastAPI `dependency_overrides` and cover
`_resolve_backend` + the `/health` backend report; the engine tests cover `PIIEngine`'s
lazy-loading contract, backend selection (bare `ModelLoader` vs `OpenMedConfig(backend=...)`), and
the engine-side `shift_dates` — the `shift_date_text`/`_is_date_label` helpers plus the
`_shift_dates` orchestration, which stubs `extract` via `monkeypatch` so no model loads). Model
tests are
in `test_pii_model.py` plus the `@pytest.mark.model` tests in `test_api.py` and `test_engine.py`
(which drive the real engine via the shared `loader` fixture — including one asserting the
engine-side `shift_dates` really shifts dates with the default model), all **skipped by default**.
The `--run-model` opt-in is wired via `pytest_addoption` + `pytest_collection_modifyitems` in
`conftest.py`, which also provides the session-scoped `loader` fixture (model loads once) and a
`note` fixture. The `shift_dates` upstream bug (see Known gotchas) is captured as a `strict=True`
`xfail` — if it ever xpasses, the suite fails, signalling the bug was fixed.

Note: `ty` is configured to target Python 3.10 (the minimum supported). openmed ships inline
type hints — e.g. `deidentify(method=...)` expects the `Literal` of the five method names — so
keep the `DeidMethod` aliases (in `examples/deidentify_pii.py` and `openmed_studio/engine.py`) in
sync with those; `test_pii_pure.py` and `test_api.py` enforce each.

## How it works

- **Backend:** the default dependency is `openmed[hf]` (Hugging Face / PyTorch), which runs
  everywhere (CPU, CUDA, Apple MPS). The `mlx` extra adds Apple's native MLX backend
  (Apple-Silicon-only). openmed auto-detects the backend (`openmed/core/backends.py`
  `get_backend`): it prefers MLX on Apple Silicon when `mlx` imports, else HuggingFace.
  `PIIEngine(backend=...)` / the `OPENMED_STUDIO_BACKEND` env var pin it explicitly (`"mlx"` raises
  off-Apple; `None`/unset = auto). The default English model is **not** in openmed's
  `_MLX_MODEL_MAP`, so on MLX it converts on-the-fly on first run (cached under
  `~/.cache/openmed/mlx/`); pre-converted `-mlx` repos exist but must be passed as a local dir
  to skip conversion.
- **Model download:** the first run pulls a model from the HF Hub and caches it under
  `~/.cache/openmed`; later runs are offline. The default PII model is the small
  `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` (~44M params).
- **Model reuse:** construct one `ModelLoader()` and pass `loader=` to every `extract_pii` /
  `deidentify` call so the model loads once instead of per-call. This is the pattern in
  `examples/deidentify_pii.py` and the documented best practice.
- **Python:** `requires-python = ">=3.10"`; the demo is verified on 3.11, but uv may pick a
  newer interpreter (e.g. 3.13) for `.venv`.
- **App structure:** `openmed_studio/` is the FastAPI app — `engine.py` (framework-free
  `PIIEngine`: one shared `ModelLoader` (built with an optional `backend` →
  `OpenMedConfig(backend=...)`, else bare so openmed auto-detects), lazy model load, wrappers over `extract_pii`/
  `deidentify`/`reidentify` with per-call `lang`/`model_name`/`date_shift_days`/`keep_year`;
  `method="shift_dates"` is handled in-engine by `_shift_dates` — `shift_date_text` +
  `_is_date_label` — rather than delegated to openmed (whose shift path is a no-op; see Known
  gotchas), returning a `_ShiftDatesResult` that duck-types openmed's `DeidentificationResult`),
  `schemas.py` (Pydantic models, `extra="forbid"`, `text` capped at 50k chars), `main.py`
  (`create_app()` + the module-level `app`, `get_engine` (overridable dependency) wiring
  `OPENMED_STUDIO_BACKEND` via `_resolve_backend` into `PIIEngine(backend=...)`, `/health`
  reporting the configured backend, and `_run` translating backend failures to `400`/`503`),
  and `__main__.py` (`python -m
  openmed_studio`). Routes: `GET /health` and `POST /pii/{extract,deidentify,deidentify/batch,
  reidentify}`; the `/pii/*` routes depend on `require_api_key`, which enforces an `X-API-Key`
  header only when `OPENMED_STUDIO_API_KEY` is set (otherwise open, with a startup warning). It
  stays a uv **non-package** project, so pytest and uvicorn import `openmed_studio` via the repo
  root on `sys.path` (`pythonpath = ["."]` for pytest; uvicorn adds its CWD). The `DeidMethod`
  `Literal` lives in `engine.py` and is re-exported by `schemas.py`;
  `tests/test_api.py::test_schema_deidmethod_matches_openmed` keeps it in sync with openmed's
  canonical method set.

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

- **OpenMed's `shift_dates` is a no-op; the engine works around it.** `openmed/core/pii.py:905`
  shifts only entities whose label is the exact string `"DATE"`, but the default model emits
  lowercase `"date"`, so openmed masks dates instead of shifting them. Model choice does **not**
  fix this (a tempting but wrong assumption): smart-merging (`openmed/core/pii_entity_merger.py`)
  relabels regex-detected dates to lowercase `"date"`, overriding whatever the model emits — and
  `date_of_birth`/`DATEOFBIRTH` aren't in openmed's shiftable set anyway. So the service handles
  `method="shift_dates"` itself in `PIIEngine._shift_dates` (`engine.py`): it normalizes date
  labels via `_is_date_label`, shifts each with `shift_date_text` (format-preserving, `keep_year`
  honored), masks all non-date PII, and applies one consistent offset — so the HTTP `shift_dates`
  path produces real shifted dates with the default model. The raw openmed no-op stays documented
  two ways: the `strict` xfail `tests/test_pii_model.py::test_shift_dates_actually_shifts_dates`,
  and the runtime note `examples/deidentify_pii.py` prints (it still calls openmed directly).
- **`reidentify()` mis-restores overlapping mapping keys.** It applies `str.replace`
  per entry, so a key that is a prefix of another (e.g. `ALIAS_1` vs `ALIAS_10`)
  corrupts the longer one. `tests/test_pii_pure.py` captures this as a `strict` xfail.
- **pysbd `SyntaxWarning`s** (a transitive dependency) appear on Python ≥3.12 from its regex
  literals; they are harmless. The demo and `openmed_studio/engine.py` silence them with
  `warnings.filterwarnings("ignore", category=SyntaxWarning)` *before* importing `openmed`.
- The `.venv` here is ~600 MB (Torch + Transformers) and is gitignored.
