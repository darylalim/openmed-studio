# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`openmed-studio` is a **clinical-NLP application** built on the
[OpenMed](https://openmed.life/docs/) clinical-NLP library (PyPI package `openmed`). It is
*not* the library itself — the library source lives at `github.com/maziyarpanahi/openmed`.
The aim is an app that surfaces OpenMed's full capability set (clinical NER, PII/PHI
de-identification, anonymization, zero-shot extraction). **Today it implements PII/PHI
de-identification only**; the other capabilities are the roadmap. The project is a
[Streamlit](https://streamlit.io/) app (`streamlit_app.py`) that runs the model **in-process**
through a reusable, framework-free `PIIEngine` and a thin in-process service seam
(`openmed_studio/service.py`). There is no separate web service — the app *is* the delivery surface
(it was a FastAPI service + thin HTTP client; that boundary was removed, see "What was dropped").

## Working with Python

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

## Commands

This is a [uv](https://docs.astral.sh/uv/) **non-package** project (`[tool.uv] package = false`
in `pyproject.toml`) — uv installs the declared dependencies into `.venv` but builds no wheel.

```bash
# Run the Streamlit app (opens http://localhost:8501). uv auto-creates .venv and installs deps.
uv run streamlit run streamlit_app.py

# Re-run fully offline once the model is cached (skips HF Hub network checks + token warning).
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run streamlit run streamlit_app.py

# Swap the portable Torch/Transformers backend for Apple's native MLX backend (Apple Silicon).
uv sync --extra mlx
# Force a backend (default unset = openmed auto-detects: MLX on Apple Silicon when the mlx extra
# is installed, else HuggingFace). "mlx" fails loudly if MLX is unavailable.
OPENMED_STUDIO_BACKEND=mlx uv run streamlit run streamlit_app.py
```

Lint, format, and type-check with the project-pinned tools (configured under
`[tool.ruff.lint]` and `[tool.ty.environment]`):

```bash
uv run ruff check .            # lint
uv run ruff check --fix .      # lint + auto-fix
uv run ruff format .           # format
uv run ty check                # type-check (resolves openmed/streamlit types from .venv)
```

Run the test suite with pytest (configured under `[tool.pytest.ini_options]`):

```bash
uv run pytest                  # fast tests only; model tests are skipped
uv run pytest --run-model      # also run the tests that load the OpenMed PII model
```

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, `ty check`, and `pytest`
on every push and PR across Python 3.10 and 3.13 (model tests stay skipped, so CI needs no model
download).

Test layout (`tests/`): fast, no-model tests live in `test_pii_pure.py`, `test_service.py`,
`test_validation.py`, `test_engine.py`, `test_ui_helpers.py`, and `test_ui_app.py`.
`test_service.py` covers the in-process seam (`PIIEngine`-stub) — `resolve_backend`/`build_engine`
backend wiring, the dict adapters (`_entity_dict`, the deidentify shaping), the success paths, that
the `use_safety_sweep` flag is forwarded to the engine (default on, overridable), and the
`ValueError`→message / `RuntimeError`+`OSError`→"unavailable" error taxonomy (`ServiceError`).
`test_validation.py` pins the input guarantees enforced before the engine is reached: the text
(50k) / batch (≤100) / mapping (≤5,000) caps, the `Lang`/`DeidMethod` enums, the confidence range,
`model_name` format, the `OPENMED_STUDIO_MAX_TEXT_LENGTH` knob, the `DeidMethod`↔openmed and
`Lang`⊆openmed (`SUPPORTED_LANGUAGES`) sync, and that a rejection message never echoes the offending
input (PHI). `test_engine.py` covers
`PIIEngine`'s lazy-loading contract, backend selection (bare `ModelLoader` vs
`OpenMedConfig(backend=...)`), and that `deidentify` forwards every method — including `shift_dates`
with its `date_shift_days`/`keep_year` controls and `use_safety_sweep` (while never forwarding
`audit`) — straight to openmed (monkeypatching `openmed.deidentify` so no model loads). It also pins
that `PIIEngine.reidentify` restores a kept mapping in one regex pass so overlapping/substring
keys can't corrupt each other (no model). Model tests are in `test_pii_model.py` plus the
`@pytest.mark.model` tests in `test_engine.py` (which drive the real engine via the shared `loader`
fixture), all **skipped by default**. The `--run-model` opt-in is wired via `pytest_addoption` +
`pytest_collection_modifyitems` in `conftest.py`, which also provides the session-scoped `loader`
fixture (model loads once) and a `note` fixture.
`test_pii_model.py::test_shift_dates_actually_shifts_dates` asserts the date-shifting fix holds:
openmed >=1.6.0 shifts the default model's lowercase-`"date"` entities (via canonical-label
normalization) instead of masking them. (`test_pii_model.py` narrows openmed 1.6.0's
`DeidentificationResult | AuditReport` return back to `DeidentificationResult` via a `_deidentify`
cast helper, since it never passes `audit=True`.)

UI tests: `test_ui_helpers.py` unit-tests the pure helpers in `ui_helpers.py`
(`render_highlighted` escaping/overlap handling, the theme-agnostic marks — a translucent `color_for`
tint plus `color: inherit` — and `build_base_opts` payload logic).
`test_ui_app.py` drives `streamlit_app.py` via `streamlit.testing.v1.AppTest`, stubbing the engine
**in-process** by patching `service.build_engine` (the shared module the running app imports) — no
model, no network; sentinel values a real model would never produce (`[[STUB-DEID-OUTPUT]]`,
`STUB/sentinel-model`) prove the rendered data came from the stub. It also covers the
`Single`→`Re-identify` session-state handoff across the `@st.fragment` boundary, that the rendered
marks are theme-agnostic (`color: inherit` + an `rgba` tint), and a widget-key
collision guard across all tabs. It opens
with `pytest.importorskip("streamlit")`, but streamlit is a **core** dependency, so the default suite
runs both UI test files and `ty check` sees `streamlit_app.py` — no extra needed.

Note: `ty` is configured to target Python 3.10 (the minimum supported). openmed ships inline
type hints — e.g. `deidentify(method=...)` expects the `Literal` of the five method names — so
keep the `DeidMethod` alias (in `openmed_studio/engine.py`, re-exported by `validation.py`) in
sync with those; `test_validation.py::test_validation_deidmethod_matches_openmed` enforces it. Tests
pass the `PIIEngine`-typed seam a structural stub via `typing.cast` (the repo convention, also in
`test_engine.py`).

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
- **Model reuse:** the app builds one `PIIEngine` (one shared `ModelLoader`) cached via Streamlit's
  `st.cache_resource`, so the model loads at most once per process and is reused across every tab
  and request. The engine pattern (construct one `ModelLoader`, pass `loader=` to every call) is the
  documented best practice.
- **Python:** `requires-python = ">=3.10"`; verified on 3.11, but uv may pick a
  newer interpreter (e.g. 3.13) for `.venv`.
- **App structure:** `openmed_studio/` is the framework-free core (no Streamlit, no HTTP):
  - `engine.py` — the `PIIEngine`: one shared `ModelLoader` (built with an optional `backend` →
    `OpenMedConfig(backend=...)`, else bare so openmed auto-detects), lazy model load, thin
    wrappers over `extract_pii`/`deidentify`/`reidentify` with per-call
    `lang`/`model_name`/`date_shift_days`/`keep_year`/`use_safety_sweep`; every method —
    including `method="shift_dates"` — is delegated straight to openmed (openmed >=1.6.0
    shifts dates correctly on the default model). `use_safety_sweep` defaults to `True`
    (openmed's default) — a deterministic structured-identifier sweep run after model
    detection, exposed as a sidebar toggle — so de-identification can redact identifiers the
    `Detect` tab's `extract_pii` (which has no sweep) does not; the `Detect` caption flags
    this. Defines the `DeidMethod` and
    `Backend` `Literal`s and `DEFAULT_PII_MODEL`.
  - `validation.py` — the Pydantic request models (`ExtractRequest`, `DeidentifyRequest`,
    `DeidentifyBatchRequest`, `ReidentifyRequest`, `extra="forbid"`) and the bound primitives
    (`ClinicalText`/`MAX_TEXT_CHARS` via `OPENMED_STUDIO_MAX_TEXT_LENGTH`, read at import by
    `_max_text_chars`; `MAX_BATCH_ITEMS`, `MAX_MAPPING_ENTRIES`, `Lang`, `_check_model_name`).
    These import only `pydantic`/`os`/`re` — no web framework — so they are reused as the
    in-process validation layer. Re-exports `DeidMethod`.
  - `service.py` — the single in-process chokepoint (framework-free): `resolve_backend()` (reads
    `OPENMED_STUDIO_BACKEND`), `build_engine()` (the `PIIEngine` factory the UI caches),
    `_validate()` (calls `model_validate()` and raises a **PHI-safe** `ServiceError` built from
    only `loc`/`msg` — never Pydantic's `input`), `_run()` (translates `ValueError`→bad-options and
    `RuntimeError`/`OSError`→backend-unavailable into `ServiceError`, mirroring the old 400/503
    split), the dict adapters (`_entity_dict`, `_deidentify_dict`), and
    `extract`/`deidentify`/`deidentify_batch`/`reidentify(engine, …, **opts)` that validate →
    call the engine → adapt to plain dicts. Every UI engine call funnels through here, so nothing
    bypasses validation.
  - `__init__.py` — re-exports `DEFAULT_PII_MODEL`, `PIIEngine`, and `__version__`.

  It stays a uv **non-package** project, so pytest imports `openmed_studio` via the repo root on
  `sys.path` (`pythonpath = ["."]` for pytest; Streamlit adds the app's directory). The `DeidMethod`
  `Literal` lives in `engine.py`, is re-exported by `validation.py`, and
  `tests/test_validation.py::test_validation_deidmethod_matches_openmed` keeps it in sync with openmed's
  canonical method set.
- **UI structure:** the Streamlit app lives at the repo root: `streamlit_app.py` (the app —
  `get_engine` is `service.build_engine` wrapped in `st.cache_resource`; `_call` runs a `service.*`
  function in a spinner and renders any `ServiceError`; the sidebar reads engine
  state (model/backend/`is_loaded`) directly, and the four tabs (`Detect` → `service.extract`,
  `Single note`/`Batch` → `service.deidentify[_batch]`, `Re-identify` → `service.reidentify`) live in
  `main()`, guarded by `if __name__ == "__main__"` so importing for tests has no side effects). The
  `Detect`/`Batch`/`Re-identify` tab renderers are `@st.fragment` so an in-tab interaction reruns
  only that tab; `Single note` is **intentionally not** a fragment, because its form submit must
  trigger a full rerun to hand `last_deidentified`/`last_mapping` (via `st.session_state`, not widget
  keys) to the `Re-identify` tab. `_render_highlight(text, entities)` (shared by Detect and Single)
  renders the highlighted text plus its legend. The pure helpers live in
  `ui_helpers.py` (Streamlit-free `render_highlighted`/`render_legend` — both **theme-agnostic**: a
  translucent per-label tint from `PALETTE`/`color_for` plus `color: inherit`, so the marks read on
  light or dark with no runtime theme detection — `render_plain`/`build_base_opts`/
  `build_batch_table`, kept separate so they unit-test without a browser). The de-identified output
  offers a `Download` button (no copy-to-clipboard button — the in-process tool deliberately avoids
  sending PHI to a browser-side clipboard component). The UI
  consumes the plain dicts `service` produces (`result["entities"]`, `result["deidentified_text"]`,
  `result.get("mapping")`). `streamlit>=1.58` (1.58 horizontal/`height="stretch"` flex layout) is a
  core dependency. The confidence slider defaults to `0.5` (the de-identify default is `0.7`). App
  config lives in `.streamlit/config.toml` (both `[theme.light]` and `[theme.dark]` are defined so
  the app honors the user's mode; the theme-agnostic marks read correctly in either;
  `gatherUsageStats = false` — a
  clinical-text tool shouldn't phone home); any local secrets go in the gitignored
  `.streamlit/secrets.toml`.
- **What was dropped (vs the old FastAPI service):** the HTTP boundary and everything that only
  existed because of it — API-key auth (`OPENMED_STUDIO_API_KEY` / `X-API-Key`), the `{"error":
  {...}}` JSON envelope, the `/compat` OpenMed-REST surface (`OPENMED_STUDIO_COMPAT`), the startup
  preload (`OPENMED_STUDIO_PRELOAD`), and `main.py`/`__main__.py`/uvicorn/fastapi/httpx/requests.
  This is now a **local, single-user** tool — put it behind your own auth/proxy before exposing it.
  The guarantees that protect the *model* regardless of transport are **kept**, enforced in-process
  by `service.py`: the text/batch/mapping caps, the value/enum/format checks, backend pinning, and
  not echoing input on a validation error. Only `OPENMED_STUDIO_BACKEND` and
  `OPENMED_STUDIO_MAX_TEXT_LENGTH` remain as env knobs.

## OpenMed PII API (verified against installed v1.6.0)

Top-level imports: `from openmed import extract_pii, deidentify, reidentify, ModelLoader, OpenMedConfig`.

- `extract_pii(text, model_name=<default>, confidence_threshold=0.5, use_smart_merging=True, lang="en", *, loader=None)`
  returns PII entities, each with `.label`, `.text`, `.start`, `.end`, `.confidence`.
  Labels are **lowercase** (`first_name`, `last_name`, `date`, `ssn`, `phone_number`, …).
- `deidentify(text, method="mask", ..., keep_mapping=False, *, consistent=False, seed=None, locale=None, use_safety_sweep=True, audit=False, loader=None)`
  returns a `DeidentificationResult` with `.deidentified_text`, `.pii_entities`, `.mapping`
  (or an `AuditReport` when `audit=True` — 1.6.0 types the return as
  `DeidentificationResult | AuditReport`; the app's engine returns it as `Any`, never sets
  `audit`, and `tests/test_pii_model.py` casts it back to `DeidentificationResult`).
  `use_safety_sweep=True` (the app's default, exposed as a sidebar toggle) runs a deterministic
  structured-identifier sweep after detection; `extract_pii` has no such parameter.
  Methods: `mask`, `remove`, `replace` (Faker surrogates — use `consistent=True, seed=N` for
  determinism), `hash`, `shift_dates`.
- `reidentify(deidentified_text, mapping)` → original text (use with `deidentify(..., keep_mapping=True)`).

## Known gotchas

- **`shift_dates` was fixed in openmed 1.6.0.** Earlier versions shifted only entities labelled
  exactly `"DATE"`, but the default `OpenMed-PII-SuperClinical-Small-44M-v1` model emits lowercase
  `"date"`, so they masked dates instead. openmed 1.6.0 matches dates by canonical label
  (`openmed/core/pii.py:_is_date_entity` normalizes the model's `"date"`), so `shift_dates` now
  shifts dates on the default model. `tests/test_pii_model.py::test_shift_dates_actually_shifts_dates`
  asserts this (it was a `strict` xfail before the upgrade).
- **openmed's `reidentify()` mis-restores overlapping mapping keys; the app fixes it.**
  openmed applies `str.replace` per entry, so a key that is a prefix/substring of another
  (e.g. `ALIAS_1` vs `ALIAS_10`, or unbracketed `hash`/`replace` surrogates) corrupts the
  longer one, and a replacement value that contains another key gets re-substituted.
  `PIIEngine.reidentify` instead restores in a single regex pass (longest key first), so no
  replacement is re-scanned and both failure modes are eliminated; `tests/test_engine.py` pins
  the prefix and value-contains-key cases. The raw-openmed limitation is still captured as a
  `strict` xfail in `tests/test_pii_pure.py`.
- **pysbd `SyntaxWarning`s** (a transitive dependency) appear on Python ≥3.12 from its regex
  literals; they are harmless. `openmed_studio/engine.py` silences them with
  `warnings.filterwarnings("ignore", category=SyntaxWarning)` *before* importing `openmed`.
- The `.venv` here is ~600 MB (Torch + Transformers) and is gitignored.
