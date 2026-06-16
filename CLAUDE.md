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
library-level demo of the same OpenMed calls. `streamlit_app.py` (the `ui` extra) is an
optional [Streamlit](https://streamlit.io/) front-end — a thin HTTP client over the `/pii/*`
API (no model in-process), so the service still enforces auth/validation.

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

# Run the optional Streamlit UI (thin HTTP client over /pii/*; needs the service running above).
uv sync --extra ui
uv run streamlit run streamlit_app.py                # opens http://localhost:8501

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
uv run --extra ui ty check     # include the UI (streamlit_app.py imports streamlit at top level)
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
that `deidentify` forwards every method — including `shift_dates` with its `date_shift_days`/
`keep_year` controls — straight to openmed, monkeypatching `openmed.deidentify` so no model
loads). Model tests are
in `test_pii_model.py` plus the `@pytest.mark.model` tests in `test_api.py` and `test_engine.py`
(which drive the real engine via the shared `loader` fixture), all **skipped by default**.
The `--run-model` opt-in is wired via `pytest_addoption` + `pytest_collection_modifyitems` in
`conftest.py`, which also provides the session-scoped `loader` fixture (model loads once) and a
`note` fixture. The `shift_dates` upstream no-op (see Known gotchas) is captured as a `strict=True`
`xfail` — if it ever xpasses, the suite fails, signalling a model swap made the canonical `"DATE"`
labels shift for real.

UI tests cover the Streamlit front-end: `test_ui_helpers.py` unit-tests the pure helpers in
`ui_helpers.py` (`render_highlighted` escaping/overlap handling, `build_base_opts` payload logic)
and runs in the default suite (no extra needed — `ui_helpers` imports only the stdlib).
`test_ui_app.py` drives `streamlit_app.py` via `streamlit.testing.v1.AppTest` (render path) and
calls `api`/`fetch_health` directly, mocking `requests.Session.request` (no service, no model); it
opens with `pytest.importorskip("streamlit")` so it **skips unless the `ui` extra is installed**.
Because `streamlit_app.py` imports `streamlit` at module top, type-checking and exercising the UI
need the extra: `uv sync --extra ui` then `uv run --extra ui pytest` / `uv run --extra ui ty check`
(mirroring how the `mlx` extra gates that backend).

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
  `OpenMedConfig(backend=...)`, else bare so openmed auto-detects), lazy model load, thin wrappers over `extract_pii`/
  `deidentify`/`reidentify` with per-call `lang`/`model_name`/`date_shift_days`/`keep_year`; every
  method — including `method="shift_dates"` — is delegated straight to openmed (with the default
  model `shift_dates` masks dates rather than shifting them; see Known gotchas)),
  `schemas.py` (Pydantic models, `extra="forbid"`, `text` capped via
  `OPENMED_STUDIO_MAX_TEXT_LENGTH` (default 50k, read at import by `_max_text_chars`), plus the
  `ErrorResponse`/`ErrorDetail` envelope), `main.py`
  (`create_app()` + the module-level `app`, `get_engine` (overridable dependency) wiring
  `OPENMED_STUDIO_BACKEND` via `_resolve_backend` into `PIIEngine(backend=...)`, an optional
  startup model-preload lifespan gated by `OPENMED_STUDIO_PRELOAD` (`_lifespan` warms the model
  off the event loop via `run_in_threadpool`; a failure degrades to lazy load), `/health`
  reporting version/model/backend/`max_text_chars` and load+auth state, `_run` translating
  backend failures to `400`/`503`, and global exception handlers wrapping every non-2xx in the
  `{"error": {code, message, details}}` envelope via `_error_response`/`_ERROR_RESPONSES` — the
  `validation_error` handler strips `input` so request PHI isn't echoed back),
  and `__main__.py` (`python -m
  openmed_studio`). Routes: `GET /health` and `POST /pii/{extract,deidentify,deidentify/batch,
  reidentify}`; the `/pii/*` routes depend on `require_api_key`, which enforces an `X-API-Key`
  header only when `OPENMED_STUDIO_API_KEY` is set (otherwise open, with a startup warning). An
  opt-in `/compat` router (mounted by `create_app` only when `OPENMED_STUDIO_COMPAT` is set via
  `_compat_enabled`/`_build_compat_router`) mirrors OpenMed's own REST surface —
  `POST /compat/pii/{extract,deidentify}` with lenient (`extra="ignore"`) request models that
  accept+ignore `keep_alive`, and openmed-shaped responses (`pii_entities`,
  `num_entities_redacted`, `timestamp`, echoed `original_text` — the input-echo is why it's off
  by default). It
  stays a uv **non-package** project, so pytest and uvicorn import `openmed_studio` via the repo
  root on `sys.path` (`pythonpath = ["."]` for pytest; uvicorn adds its CWD). The `DeidMethod`
  `Literal` lives in `engine.py` and is re-exported by `schemas.py`;
  `tests/test_api.py::test_schema_deidmethod_matches_openmed` keeps it in sync with openmed's
  canonical method set.
- **UI structure:** the optional Streamlit front-end lives at the repo root (not in
  `openmed_studio/`): `streamlit_app.py` (the app — `get_session`/`fetch_health`/`api` as an HTTP
  client over `/pii/*`, `_call` wrapping `api` in a spinner, the sidebar + four tabs
  (`Detect` → `/pii/extract`, `Single note`/`Batch` → `/pii/deidentify[/batch]`, `Re-identify`) in
  `main()`, guarded by `if __name__ == "__main__"` so importing the module for tests has no side
  effects) and `ui_helpers.py` (pure, Streamlit-free `render_highlighted`/`render_legend`/
  `render_plain`/`build_base_opts`, kept separate so they unit-test without a browser). Gated by the
  `ui` extra (`requests`, `streamlit>=1.58` — the UI uses 1.58 horizontal/`stretch` flex layout).
  The confidence slider defaults to `0.5` (the de-identify API default is `0.7`).

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

- **OpenMed's `shift_dates` is a no-op with the default model.** `openmed/core/pii.py:905`
  shifts only entities whose label is the exact string `"DATE"`, but the default
  `OpenMed-PII-SuperClinical-Small-44M-v1` model emits lowercase `"date"`, so openmed masks dates
  instead of shifting them. Smart-merging (`openmed/core/pii_entity_merger.py`) compounds it:
  it relabels regex-detected dates to lowercase `"date"`, overriding whatever the model emits —
  and `date_of_birth`/`DATEOFBIRTH` aren't in openmed's shiftable set anyway. The service does
  **not** work around this — `PIIEngine.deidentify` delegates `shift_dates` to openmed like every
  other method — so on the default model the `shift_dates` path masks dates. Switching to a model
  that emits canonical `"DATE"` labels makes native shifting work; the Privacy Filter family
  (`OpenMed/privacy-filter-*`) is a candidate because it decodes with Viterbi-constrained BIOES and
  *bypasses* regex smart-merging, so its labels aren't relabeled to lowercase `"date"` (verify
  before relying on it). The no-op is pinned two ways: the `strict` xfail
  `tests/test_pii_model.py::test_shift_dates_actually_shifts_dates` (an XPASS means a model swap
  fixed it — delete the xfail), and the runtime note `examples/deidentify_pii.py` prints.
- **`reidentify()` mis-restores overlapping mapping keys.** It applies `str.replace`
  per entry, so a key that is a prefix of another (e.g. `ALIAS_1` vs `ALIAS_10`)
  corrupts the longer one. `tests/test_pii_pure.py` captures this as a `strict` xfail.
- **pysbd `SyntaxWarning`s** (a transitive dependency) appear on Python ≥3.12 from its regex
  literals; they are harmless. The demo and `openmed_studio/engine.py` silence them with
  `warnings.filterwarnings("ignore", category=SyntaxWarning)` *before* importing `openmed`.
- The `.venv` here is ~600 MB (Torch + Transformers) and is gitignored.
