# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`openmed-studio` is a **clinical-NLP application** built on the
[OpenMed](https://openmed.life/docs/) clinical-NLP library (PyPI package `openmed`) — *not* the
library itself (that lives at `github.com/maziyarpanahi/openmed`). The aim is to surface OpenMed's
full capability set (clinical NER, PII/PHI de-identification, anonymization, zero-shot extraction).
**Today it implements PII/PHI de-identification — including surrogate anonymization (the `Anonymize`
tab over `deidentify(method="replace")`) — plus clinical NER (token-classification).** Deeper
anonymization (OpenMed's `Anonymizer`/`policy` machinery) and zero-shot extraction are the roadmap.
It's a [Streamlit](https://streamlit.io/) app (`streamlit_app.py`) running the model **in-process**
through a framework-free `PIIEngine` behind a thin service seam (`openmed_studio/service.py`) — no
separate web service. (It was a FastAPI service + HTTP client; that boundary was removed — see "What
was dropped".)

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
on pushes to `main` and every PR, across Python 3.10 and 3.13 (model tests stay skipped, so CI needs
no model download).

Test layout (`tests/`) — fast no-model tests by file (model tests are a separate opt-in, below):

| File | Pins |
|------|------|
| `test_pii_pure.py` | pure-Python behavior; the raw-openmed `reidentify` overlap bug as a `strict` xfail (see "Known gotchas") |
| `test_service.py` | the in-process seam (a `PIIEngine` stub): backend wiring, the dict adapters, success paths, engine-option forwarding, the `analyze` path, the `ServiceError` taxonomy (`ValueError`→message / `RuntimeError`+`OSError`→"unavailable"), and batch per-note isolation |
| `test_validation.py` | pre-engine input guards: the text (50k) / batch (≤100) / mapping (≤5,000) caps, the enums/ranges/formats, the `OPENMED_STUDIO_MAX_TEXT_LENGTH` knob, that a rejection never echoes the input (PHI), and the openmed-sync guards |
| `test_engine.py` | `PIIEngine` lazy-load + backend selection (bare `ModelLoader` vs `OpenMedConfig(backend=...)`), that `deidentify`/`analyze` forward to openmed (monkeypatched, no model), and the one-pass `reidentify` (see "Known gotchas") |
| `test_ui_helpers.py` | the pure `ui_helpers.py` helpers — `render_highlighted` escaping/overlap, the theme-agnostic marks, `build_base_opts` payload |
| `test_ui_app.py` | drives the app via `streamlit.testing.v1.AppTest` (engine stubbed in-process; sentinels like `[[STUB-DEID-OUTPUT]]` prove output came from the stub) |

Named guards worth knowing — each **fails CI when openmed drifts**:
`test_validation_deidmethod_matches_openmed` (`DeidMethod`↔openmed),
`test_validation_lang_subset_of_openmed` (`Lang`⊆`SUPPORTED_LANGUAGES`),
`test_validation_ner_models_resolve_in_openmed` (`NER_MODELS`↔registry, incl. baked
`recommended_confidence`/`entity_types`),
`test_deidentify_forwards_every_openmed_param_or_allowlists_it` (introspects
`inspect.signature(openmed.deidentify)`, pinning the forwarded-vs-excluded split from "OpenMed API"),
and `test_shift_dates_actually_shifts_dates` (see "Known gotchas").

Model tests (`test_pii_model.py` + the `@pytest.mark.model` tests in `test_engine.py`) are
**skipped by default** and drive the real engine via the shared `loader` fixture; `--run-model` opts
in, wired in `tests/conftest.py` (`pytest_addoption` + `pytest_collection_modifyitems`, plus the
session-scoped `loader` and a `note` fixture).

Note: `ty` targets Python 3.10 (the minimum). openmed ships inline type hints — `deidentify(method=…)`
expects the `Literal` of the five method names — so the `DeidMethod` alias (in `engine.py`,
re-exported by `validation.py`) must stay in sync; the guard above enforces it. Tests pass the
`PIIEngine` seam a structural stub via `typing.cast` (the repo convention).

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
  `st.cache_resource`, so the PII model loads at most once per process and is reused across every tab
  and request. The engine pattern (construct one `ModelLoader`, pass `loader=` to every call) is the
  documented best practice. The shared loader dispatches/caches by `model_name`, so the `Clinical
  NER` tab loads a per-domain NER model (~141M each, `Medical` 434M) into the *same* loader on first
  use of that domain — switching domains loads another model rather than rebuilding the loader.
- **Python:** `requires-python = ">=3.10"`; verified on 3.11, but uv may pick a
  newer interpreter (e.g. 3.13) for `.venv`.
- **App structure:** `openmed_studio/` is the framework-free core (no Streamlit, no HTTP):
  - `engine.py` — the `PIIEngine` (one shared `ModelLoader`, lazy load) plus the model registry:
    - *Loader + wrappers:* the `ModelLoader` is built with an optional `backend` →
      `OpenMedConfig(backend=...)`, else bare so openmed auto-detects; thin wrappers cover
      `extract_pii`/`deidentify`/`reidentify`.
    - *De-identify options* (per-call, surfaced in each de-identifying tab's `Advanced` expander,
      conditioned on the method): `consistent`/`seed`/`locale` are the `replace` determinism knobs
      (`locale` e.g. `pt_BR` overrides the locale openmed derives from `lang`);
      `date_shift_days`/`keep_year` drive `shift_dates`; `use_safety_sweep` (default `True`) is a
      deterministic structured-identifier sweep run after detection that redacts identifiers
      `extract_pii` (no sweep) misses — the `Detect` caption flags this. `use_smart_merging`
      (default on) is forwarded too. Every method delegates straight to openmed (see "Known gotchas"
      for the `shift_dates` fix).
    - *Clinical NER:* `analyze(text, *, model_name, confidence_threshold=0.0, aggregation_strategy,
      group_entities)` delegates to `analyze_text`. `model_name` is **required** (NER is one model
      per domain; an absent one silently falls back to openmed's disease-only default). `analyze_text`
      returns a `PredictionResult` *object* (its `output_format="dict"` is a misnomer), so `_entities`
      unwraps `.entities`; no `lang` (analyze_text has none).
    - *Registry:* defines the `DeidMethod`/`Backend` `Literal`s, `DEFAULT_PII_MODEL`,
      `DEFAULT_NER_MODEL`, the `NerModel` `NamedTuple`, and `NER_MODELS` — a curated
      `dict[domain → NerModel]` of one ~141M "superclinical" model per category (`Medical` = the
      broader 434M `clinicalner`). Each `NerModel` bakes registry metadata (`alias`, `display_name`,
      `recommended_confidence`, `entity_types`, `params`) so the UI needs **no runtime openmed
      import**; the drift guard pins it to the live registry.
  - `validation.py` — the Pydantic request models (`ExtractRequest`, `NerRequest`,
    `DeidentifyRequest`, `DeidentifyBatchRequest`, `ReidentifyRequest`, all `extra="forbid"`) plus
    the bound primitives: `ClinicalText`/`MAX_TEXT_CHARS` (from `OPENMED_STUDIO_MAX_TEXT_LENGTH` via
    `_max_text_chars` at import), `MAX_BATCH_ITEMS`, `MAX_MAPPING_ENTRIES`, `Lang`,
    `_check_model_name`, `RequiredModelName` (the non-optional model id `NerRequest` requires), and
    `_check_locale` (a format guard on the optional `replace` `locale`). Imports only
    `pydantic`/`os`/`re` — no web framework — so it doubles as the in-process validation layer.
    Re-exports `DeidMethod`.
  - `service.py` — the single in-process chokepoint (framework-free); every UI engine call funnels
    through it, so nothing bypasses validation:
    - `resolve_backend()` (reads `OPENMED_STUDIO_BACKEND`) and `build_engine()` (the `PIIEngine`
      factory the UI caches).
    - `_validate()` — `model_validate()`, raising a **PHI-safe** `ServiceError` from only `loc`/`msg`
      (never Pydantic's `input`).
    - `_run()` — translates `ValueError`→bad-options and `RuntimeError`/`OSError`→backend-unavailable
      into `ServiceError` (the old 400/503 split, now capability-neutral since NER flows through it).
    - the dict adapters (`_entity_dict`, `_deidentify_dict`) and the entry points
      `extract`/`analyze`/`deidentify`/`deidentify_batch`/`reidentify`, which validate → call the
      engine → adapt to plain dicts. `analyze` reuses `_validate`/`_run`/`_entity_dict` verbatim
      (NER's `EntityPrediction` exposes the same fields the adapter reads).
    - `deidentify_batch` isolates each note: a per-note `ValueError` becomes an `{"ok": False}` row
      so one bad note doesn't abort the batch, while a backend `RuntimeError`/`OSError` propagates
      through `_run` and aborts the whole batch.
  - `__init__.py` — re-exports `DEFAULT_PII_MODEL`, `DEFAULT_NER_MODEL`, `NER_MODELS`,
    `PIIEngine`, and `__version__`.

  It stays a uv **non-package** project, so pytest imports `openmed_studio` via the repo root on
  `sys.path` (`pythonpath = ["."]`; Streamlit adds the app's directory).
- **UI structure:** the Streamlit app lives at the repo root in `streamlit_app.py`; the pure,
  Streamlit-free render helpers live in `ui_helpers.py` so they unit-test without a browser.
  - *App + tabs:* `get_engine` is `service.build_engine` wrapped in `st.cache_resource`; `_call`
    runs a `service.*` function in a spinner and renders any `ServiceError`. `main()` titles the
    page/heading "OpenMed Studio" and lays out the six tabs (`Detect`→`service.extract`,
    `Clinical NER`→`service.analyze`, `Single note`/`Batch`→`service.deidentify[_batch]`,
    `Anonymize`→`service.deidentify` (`method=replace`), `Re-identify`→`service.reidentify`), guarded
    by `if __name__ == "__main__"` so importing for tests has no side effects.
  - *Fragments + handoff:* `Detect`/`Clinical NER`/`Batch`/`Re-identify` renderers are `@st.fragment`
    so an in-tab interaction reruns only that tab; `Single note` and `Anonymize` are **intentionally
    not**, because their form submit must trigger a full rerun to hand `last_deidentified`/
    `last_mapping` (via `st.session_state`, not widget keys) to `Re-identify`. `_set_handoff` sets the
    two together once per submit — the single security-relevant copy of "so a stale mapping can't
    linger" — *not* on re-render.
  - *Result persistence:* all four de-identifying surfaces persist their latest result in
    `st.session_state` (`single_result`/`anon_result` via the shared `_submit_deidentify` helper,
    which centralizes submit→call→persist→handoff so Single/Anonymize can't drift; plus
    `batch_result` and `reid_result`) and render from there, so post-submit reruns (a Download, a
    control tweak, the "Show re-identification key" click) don't blank the panel and a failed/empty
    re-submit warns without losing the last good result (a snapshot caption flags this). The mapping
    is revealed in an `@st.dialog` (`_show_mapping_dialog`) behind a button, not an always-open
    expander.
  - *De-identification controls:* `Method` plus the method-conditional `Advanced` knobs
    (the surrogate methods `replace`/`format_preserve`→consistent/seed/locale,
    `shift_dates`→date_shift_days/keep_year, plus the safety
    sweep) live in `Single note` + `Batch` via a shared `_render_deid_controls(key_prefix=…, lang=…)`
    (above each tab's form, widget keys `key_prefix`-scoped so the tabs don't collide). `Detect` has
    its own confidence slider + smart-merge toggle; `Anonymize` reads the sidebar `Language`. Only
    `Method`/`Advanced` are per-tab — the sidebar holds just the engine readout
    (model/backend/`is_loaded`, read directly) and the lone global `Language` filter
    (`_render_sidebar` returns the chosen `lang`).
  - *Clinical NER controls* (`_render_ner`, independent of the de-id controls): a domain picker
    (`st.selectbox` over `NER_MODELS`, default `Disease`) sits **outside** the form so selecting a
    domain reruns the fragment and refreshes both a reactive preview (the model's `display_name`,
    size, `entity_types` — flagging `Medical` as the broad 434M model) and the confidence slider's
    default (seeded from the model's `recommended_confidence`, per-domain keyed). `model_name`
    resolves via `NER_MODELS[domain].alias`. Because `engine.is_loaded` only tracks whether *a* model
    has loaded, the per-domain download wait-hint is driven by a `st.session_state` set of analyzed
    domains (passed to `_call(..., needs_load=...)`), so switching to a not-yet-downloaded domain
    still warns.
  - *Rendering:* `_render_highlight(text, entities)` (shared by `Detect`, `Single`, `Anonymize`, and
    `Clinical NER`) renders the highlighted text plus its legend, label-agnostic so it handles NER's
    UPPERCASE labels unchanged. `ui_helpers.py`'s `render_highlighted`/`render_legend` are
    **theme-agnostic**: a translucent per-label tint from `PALETTE`/`color_for` plus `color: inherit`,
    so the marks read on light or dark with no runtime theme detection (`render_plain`/
    `build_base_opts`/`build_batch_table` are kept separate for browserless unit tests). The
    de-identified output offers a `Download` button (**no** copy-to-clipboard — the in-process tool
    deliberately avoids sending PHI to a browser-side clipboard component); entity tables render
    confidence as a `ProgressColumn`, count/method metrics are bordered cards, and
    `Download`/`Re-identify` confirm with an `st.toast`. The UI consumes the plain dicts `service`
    produces (`result["entities"]`, `result["deidentified_text"]`, `result.get("mapping")`). The
    confidence slider defaults to `0.5` (the de-identify default is `0.7`).
  - *Config:* `streamlit>=1.58` (1.58 horizontal/`height="stretch"` flex layout) is a core
    dependency. `.streamlit/config.toml` defines both `[theme.light]` and `[theme.dark]` (so the app
    honors the user's mode; the theme-agnostic marks read correctly in either) plus a shared `[theme]`
    with `baseRadius` and a semantic `red`/`green`/`orange` palette brightened per mode, so status
    accents feel intentional; `gatherUsageStats = false` (a clinical-text tool shouldn't phone home).
    Local secrets go in the gitignored `.streamlit/secrets.toml`, and the download outputs
    (`deidentified.txt`/`anonymized.txt`/`reidentified.txt`/`deidentified_batch.json`) are gitignored
    too, since they can carry PHI or its surrogates.
- **What was dropped (vs the old FastAPI service):** the HTTP boundary and everything that only
  existed because of it — API-key auth (`OPENMED_STUDIO_API_KEY` / `X-API-Key`), the `{"error":
  {...}}` JSON envelope, the `/compat` OpenMed-REST surface (`OPENMED_STUDIO_COMPAT`), the startup
  preload (`OPENMED_STUDIO_PRELOAD`), and `main.py`/`__main__.py`/uvicorn/fastapi/httpx/requests.
  This is now a **local, single-user** tool — put it behind your own auth/proxy before exposing it.
  The guarantees that protect the *model* regardless of transport are **kept**, enforced in-process
  by `service.py`: the text/batch/mapping caps, the value/enum/format checks, backend pinning, and
  not echoing input on a validation error. Only `OPENMED_STUDIO_BACKEND` and
  `OPENMED_STUDIO_MAX_TEXT_LENGTH` remain as env knobs.

## OpenMed API (verified against installed v1.7.0)

Top-level imports: `from openmed import extract_pii, deidentify, reidentify, analyze_text, ModelLoader, OpenMedConfig`.
Registry helpers used by the NER picker / drift guard: `get_all_models()` (dict alias→ModelInfo),
`list_model_categories()`.

- `extract_pii(text, model_name=<default>, confidence_threshold=0.5, config=None, use_smart_merging=True, lang="en", normalize_accents=None, *, loader=None)`
  returns a `PredictionResult` object (like `analyze_text`, below) whose `.entities` are PII
  predictions with `.label`/`.text`/`.start`/`.end`/`.confidence` — the engine's `_entities` unwraps
  it. Labels are **lowercase** (`first_name`, `last_name`, `date`, `ssn`, `phone_number`, …). The
  engine forwards `confidence_threshold`/`use_smart_merging`/`lang`/`model_name`/`loader` only (it
  owns loading, so `config`/`normalize_accents` are not threaded).
- `deidentify(text, method="mask", model_name=<default>, confidence_threshold=0.7,
  use_smart_merging=True, keep_mapping=False, consistent=False, seed=None, locale=None,
  date_shift_days=None, keep_year=False, lang="en", use_safety_sweep=True, audit=False,
  loader=None, …)` — the installed v1.7.0 signature **also** accepts `shift_dates` (a bool,
  distinct from `method="shift_dates"`), `normalize_accents`, `config`, `policy`,
  `calibration_thresholds_path`, and 1.7.0's `patient_key`/`date_shift_max_days`/
  `date_shift_secret`/`surrogate_vault`/`custom_recognizer`/`cache_results`/`max_cache_entries`.
  (Note: `keep_year` now defaults to `False` upstream, but the app always passes its own value —
  default `True` in `_DeidentifyOptions`/`PIIEngine.deidentify` — so the flip is inert.)
  It returns a `DeidentificationResult` with
  `.deidentified_text`, `.pii_entities`, `.mapping` (or an `AuditReport` when `audit=True` —
  1.7.0 types the return as `DeidentificationResult | AuditReport`; the app's engine returns it
  as `Any`, never sets `audit`, and `tests/test_pii_model.py` casts it back to
  `DeidentificationResult`).
  The app's engine forwards `method`/`confidence_threshold`/`use_smart_merging`/`keep_mapping`/
  `consistent`/`seed`/`locale`/`date_shift_days`/`keep_year`/`use_safety_sweep` plus
  `lang`/`model_name`/`loader`; it deliberately does **not** forward `audit` (would flip the
  return type), `config` (the engine owns loading via `loader=`), or the advanced
  `shift_dates`/`normalize_accents`/`policy`/`calibration_thresholds_path` and the 1.7.0
  `patient_key`/`date_shift_max_days`/`date_shift_secret`/`surrogate_vault`/`custom_recognizer`/
  `cache_results`/`max_cache_entries` knobs.
  `tests/test_engine.py::test_deidentify_forwards_every_openmed_param_or_allowlists_it`
  introspects this real signature and pins the forwarded-vs-excluded split so it can't drift.
  `use_safety_sweep=True` runs a post-detection structured-identifier sweep `extract_pii` has no
  equivalent of (see the `engine.py` notes for how the UI surfaces it).
  Methods: `mask`, `remove`, `replace` (Faker surrogates — `consistent=True, seed=N` for
  determinism, `locale="pt_BR"` etc. for a specific surrogate locale, exposed in the de-identifying
  tabs' `Advanced` expander), `hash`, `shift_dates`, and `format_preserve` (a `replace` sibling
  added in 1.7.0 — synthetic *format-preserving* surrogates for structured identifiers, masking
  free-text entities like names it can't shape-preserve; shares `replace`'s consistent/seed/locale
  knobs).
- `reidentify(deidentified_text, mapping)` → original text (use with `deidentify(..., keep_mapping=True)`).
- `analyze_text(text, model_name="disease_detection_superclinical", *, loader=None,
  confidence_threshold=0.0, aggregation_strategy="simple", output_format="dict",
  group_entities=False, …)` — the general **clinical NER** (token-classification) entry point. With
  the default `output_format="dict"` it returns a `PredictionResult` **object** (a misnomer — *not*
  a plain dict) whose `.entities` is a `list[EntityPrediction]`, each with `.text`/`.label`/
  `.confidence`/`.start`/`.end`. Labels are **UPPERCASE** (`DISEASE`, `CHEM`, `GENE`, …), unlike
  `extract_pii`'s lowercase. Clinical NER is **one model per domain** (no universal model), selected
  by registry alias via `model_name`; the app curates one per domain in `engine.NER_MODELS` and
  pins them with `tests/test_validation.py::test_validation_ner_models_resolve_in_openmed`. The
  app's engine forwards `model_name`/`confidence_threshold`/`aggregation_strategy`/`group_entities`/
  `output_format="dict"`/`loader` (no `lang` — `analyze_text` has none). It excludes the alternate
  construction / tuning knobs (`model_id`/`config`/`include_confidence`/`formatter_kwargs`/
  `metadata`/`use_fast_tokenizer`/`sentence_*`) plus 1.7.0's `cache_results`/`max_cache_entries`;
  `tests/test_engine.py::test_analyze_forwards_every_openmed_param_or_allowlists_it` pins that split
  (it matters more here than for `deidentify`: `analyze_text` declares `**pipeline_kwargs`, so a
  drifted forwarded param would be silently swallowed rather than raising).

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
- The `.venv` here is ~1.1 GB (Torch + Transformers) and is gitignored.
