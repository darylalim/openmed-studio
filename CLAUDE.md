# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`openmed-studio` is a **clinical-NLP application** built on the
[OpenMed](https://openmed.life/docs/) clinical-NLP library (PyPI package `openmed`) — *not* the
library itself (that lives at `github.com/maziyarpanahi/openmed`). The aim is to surface OpenMed's
full capability set (clinical NER, PII/PHI de-identification, anonymization, zero-shot extraction).
**Today it implements PII/PHI de-identification — including surrogate anonymization (the `Anonymize`
tab over `deidentify(method="replace")`) and **policy-driven anonymization** (the `Policy de-ID` tab
over `deidentify(policy=…)` — OpenMed's regulatory compliance profiles: HIPAA Safe Harbor, GDPR
pseudonymization, etc.) — clinical NER (token-classification), and zero-shot (GLiNER) extraction (the
`Zero-shot` tab over `openmed.ner.infer`, behind the optional `gliner` extra).** Deeper policy tooling
(user-authored custom policies, cross-document `SurrogateVault` consistency) is the roadmap.
It has **two delivery surfaces over one shared in-process seam** (`openmed_studio/service.py`): a
[Streamlit](https://streamlit.io/) app (`streamlit_app.py`) and a [FastAPI](https://fastapi.tiangolo.com/)
service (`openmed_studio/main.py`). Both run the model **in-process** through a framework-free
`PIIEngine` — the FastAPI service is a *thin HTTP layer over the same seam the UI uses*, not a
separate service the UI calls (see "The two surfaces").

## Working with Python

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

## Commands

This is a [uv](https://docs.astral.sh/uv/) **non-package** project (`[tool.uv] package = false`
in `pyproject.toml`) — uv installs the declared dependencies into `.venv` but builds no wheel.

### Running it

```bash
# Run the Streamlit app (opens http://localhost:8501). uv auto-creates .venv and installs deps.
uv run streamlit run streamlit_app.py

# Re-run fully offline once the model is cached (skips HF Hub network checks + token warning).
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run streamlit run streamlit_app.py

# Add Apple's native MLX backend (Apple Silicon); openmed then prefers it over Torch, which stays.
uv sync --extra mlx

# Enable the Zero-shot (GLiNER) tab. gliner caps transformers<5.17, so this extra CONFLICTS with
# the marker `hf-latest` extra (declared in [tool.uv] conflicts) — uv forks the lock so the
# default install/CI stay on the latest transformers and only this opt-in downgrades. Combines
# with --extra mlx. Until installed, the Zero-shot tab shows install instructions, not the form.
uv sync --extra gliner
# Force a backend (default unset = openmed auto-detects: MLX on Apple Silicon when the mlx extra
# is installed, else HuggingFace). "mlx" fails loudly if MLX is unavailable.
OPENMED_STUDIO_BACKEND=mlx uv run streamlit run streamlit_app.py

# Run the FastAPI service (the second delivery surface; open http://127.0.0.1:8080/docs). fastapi
# and uvicorn are CORE deps, so no extra is needed. Host/port via OPENMED_STUDIO_HOST/PORT.
uv run python -m openmed_studio
# or, equivalently, with uvicorn directly (e.g. to add --reload):
uv run uvicorn openmed_studio.main:app --port 8080
# Require an API key on every model route (unset = runs unauthenticated + a startup warning).
OPENMED_STUDIO_API_KEY=secret uv run python -m openmed_studio
# Warm the model at startup so the first request isn't slow; mount the opt-in /compat surface.
OPENMED_STUDIO_PRELOAD=1 OPENMED_STUDIO_COMPAT=1 uv run python -m openmed_studio
```

| Env var | Effect |
|---|---|
| `OPENMED_STUDIO_BACKEND` | Pin `hf`/`mlx` (`service.resolve_backend`); unset = openmed auto-detects. `mlx` raises off-Apple. |
| `OPENMED_STUDIO_MAX_TEXT_LENGTH` | Per-request text cap; read **at import** by `validation._max_text_chars` (default 50,000). |
| `OPENMED_STUDIO_API_KEY` | Require `X-API-Key` on every model route; unset = unauthenticated + a startup warning. |
| `OPENMED_STUDIO_PRELOAD` | Truthy = warm the model in a threadpool at FastAPI startup. |
| `OPENMED_STUDIO_COMPAT` | Truthy = mount the two-route `/compat` OpenMed-REST surface. |
| `OPENMED_STUDIO_HOST` / `_PORT` | `python -m openmed_studio` bind address (defaults `127.0.0.1:8080`). |
| `OPENMED_TORCH_ATTENTION_BACKEND` | openmed's own knob; overrides the engine's `eager` pin (see "Known gotchas"). |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | Skip HF Hub network checks once the model is cached. |

### Static checks and tests

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

### CI

CI (`.github/workflows/ci.yml`) runs on pushes to `main` and every PR: `pytest` across Python 3.10
and 3.13, and `ruff check` / `ruff format --check` / `ty check` **once** on the 3.10 leg (ruff never
reads `.venv` and ty targets 3.10 via `[tool.ty.environment]` whatever interpreter runs it, so a
second leg would duplicate the work and the failure annotations). Model tests stay skipped, so CI
needs no model download. CI installs with `uv sync --locked --group dev`, so the committed `uv.lock`
is **part of the contract**: a `pyproject.toml` dependency edit must be followed by `uv lock` or CI
fails before running a single test. `astral-sh/setup-uv` is pinned to an **exact** version — it stopped
publishing floating major tags at v8 — so it must be bumped by hand.

### Releases

Releases (`.github/workflows/release.yml`) are cut from `pyproject.toml`'s `version`. A push to
`main` touching that file runs a `guard` job that reads the version and checks whether `v<version>`
is already a tag; if it is, the run is a no-op (so a `pyproject.toml` edit that doesn't bump the
version releases nothing — the `paths` filter is only a cheap pre-filter, the tag check is the real
decision). If the tag is new, `verify` re-runs `ci.yml` through `workflow_call` — CI is **reusable**
for exactly this reason, so "green" has one definition and a tag can never point at a red commit —
and then `gh release create --target "$GITHUB_SHA"` publishes the tag and the release in one API
call (no `git push`, no third-party action in the trust chain). A pure-numeric version gets
`--latest`; anything with an a/b/rc/dev marker gets `--prerelease`. `workflow_dispatch` is the
manual retry after a failed run. **No wheel is built and nothing goes to PyPI**
(`[tool.uv] package = false`) — the release carries only GitHub's automatic source archives.
Caveat: two version bumps merged within one CI cycle can leave the intermediate version untagged,
because the called `ci.yml` brings its own `cancel-in-progress` concurrency group.

The release body comes from GitHub's `releases/generate-notes` API, **with a commit-log fallback**:
that API builds its body from merged **pull requests** only, and this repo has never had one (every
commit pushed straight to `main`), so it returns nothing but a `**Full Changelog**` link —
78 bytes, verified by dry-running the endpoint. The job therefore checks whether the generated body
contains any `* ` entry and, when it doesn't, substitutes `git log --no-merges --reverse
--format='* %s' "$prev..HEAD"` under a `## Commits` heading, keeping the generator's compare link
(with no `v*` tag yet — the state today — the range is a bare `HEAD`, so the first release's notes
list the entire history).
Two consequences: the release job needs `fetch-depth: 0` (it runs `git log`), and **commit subjects
are release notes now** — write them accordingly. If the repo ever moves to a PR workflow the
fallback goes quiet on its own, since the generated body will have `* ` entries again.

### Test layout

Test layout (`tests/`) — fast no-model tests by file (model tests are a separate opt-in, below):

| File | Pins |
|------|------|
| `test_pii_pure.py` | pure-Python behavior; the raw-openmed `reidentify` overlap bug as a `strict` xfail (see "Known gotchas") |
| `test_service.py` | the in-process seam (a `PIIEngine` stub): backend wiring, the dict adapters, success paths, engine-option forwarding, the `analyze` + `anonymize_policy` paths (policy forwarding, no forced `keep_mapping`, policy-decided mapping surfaced), the `ServiceError` taxonomy — both its message (`ValueError`→message / `RuntimeError`+`OSError`→"unavailable") **and its transport-neutral `.kind`** (`validation`/`bad_options`/`unavailable`/`dependency`/`internal`, the classification the FastAPI layer maps to a status), and batch per-note isolation — plus two `--run-model` tests that drive the real engine |
| `test_validation.py` | pre-engine input guards: the text (50k) / batch (≤100) / mapping (≤5,000) caps, the enums/ranges/formats, the `OPENMED_STUDIO_MAX_TEXT_LENGTH` knob, that a rejection never echoes the input (PHI), and the openmed-sync guards |
| `test_engine.py` | `PIIEngine` lazy-load + backend selection (the loader is **always** `ModelLoader(OpenMedConfig(backend=…, torch_attention_backend="eager"))` — one test pins `backend=None`, one pins `backend="mlx"`, and both pin the eager kwarg), that `deidentify`/`analyze`/`extract_zero_shot` forward to openmed (monkeypatched, no model — incl. `policy` forwarding, and the zero-shot test pins the in-memory index with `family="gliner"` and `is_loaded` False), the one-pass `reidentify` (see "Known gotchas"), that the model methods run their openmed call **under `self._lock`** while `reidentify` stays lock-free, and `--run-model` policy tests: masking vs reversible-surrogate, plus pins on the description prose the fast guard can't check — the sweep masking only what its patterns match under the four keep-dates profiles (and nothing with Clinical Minimal Redaction's optional sweep off), the four mask-everything profiles matching Strict No-Leak with clinical text untouched, and China PIPL / NG NDPA / ZA POPIA's surrogate-vs-mask split |
| `test_ui_helpers.py` | the pure `ui_helpers.py` helpers — `render_highlighted` escaping/overlap, the theme-agnostic marks, `build_base_opts` payload |
| `test_ui_app.py` | drives the app via `streamlit.testing.v1.AppTest` (engine stubbed in-process; sentinels like `[[STUB-DEID-OUTPUT]]` prove output came from the stub) |
| `test_api.py` | drives the FastAPI service via `fastapi.testclient.TestClient` (engine stubbed via `dependency_overrides`; needs the `httpx` dev dep, **no** `--run-model`): routing to each of the 7 seam functions, the `ServiceError.kind`→HTTP-status mapping + the `{"error":{code,message,details}}` envelope, PHI-safe 422s, `X-API-Key` auth (401/accept/reject + open `/health`), and the opt-in `/compat` surface (openmed-shaped payloads, echoed `original_text`, auth-gated) |
| `test_hooks.py` | the repo's own Claude Code hooks (no openmed, no model): **executes** `.claude/hooks/block-phi-paths.sh` rather than parsing its shell text, pinning that every `Download` filename in `streamlit_app.py` and every local-secret path is denied (exit 2), that ordinary source files and non-file tool calls are allowed (exit 0), that unparseable input **fails closed**, that the `case` arms carry no dead entries, that those same names are in `.gitignore` (the other half of the invariant), and that `.claude/settings.json` still registers the hook under `PreToolUse`. The whole file `skipif`s when `sh` or `python3` is missing |

Named guards worth knowing — each **fails CI when openmed drifts**:
`test_validation_deidmethod_matches_openmed` (`DeidMethod`↔openmed),
`test_validation_lang_subset_of_openmed` (`Lang`⊆`SUPPORTED_LANGUAGES`),
`test_validation_ner_models_resolve_in_openmed` (`NER_MODELS`↔registry, incl. baked
`recommended_confidence`/`entity_types`),
`test_zero_shot_models_resolve_in_openmed` (`ZERO_SHOT_MODELS`↔registry, incl. baked
`recommended_confidence`/`entity_types` and each `label_domain`⊆`openmed.ner.available_domains()`;
registry/label metadata only — no download, so it runs in CI without the `gliner` extra),
`test_validation_policy_matches_openmed` (`Policy`↔`openmed.core.policy.PolicyName`),
`test_policy_models_resolve_in_openmed` (`POLICY_MODELS`↔`list_policies()`/`load_policy()`, incl. baked
`default_action`/`keep_mapping`/`safety_sweep_mandatory`; profile metadata only — no
download), `test_deidentify_forwards_every_openmed_param_or_allowlists_it` (introspects
`inspect.signature(openmed.deidentify)`, pinning the forwarded-vs-excluded split from "OpenMed API" —
`policy` is now **forwarded**, not excluded), and — **the one exception** —
`test_shift_dates_actually_shifts_dates`, which is `@pytest.mark.model` in `test_pii_model.py`, so it
catches that drift only under `--run-model`, never in CI (see "Known gotchas").
Two more openmed-drift guards live outside this list:
`test_analyze_forwards_every_openmed_param_or_allowlists_it` (see "OpenMed API" — it matters *more*
than its `deidentify` twin) and `test_occurrence_prefix_matches_openmed` (see "Known gotchas").

One guard tracks the **repo's own tooling** instead of openmed. `.claude/hooks/block-phi-paths.sh`
is a PreToolUse hook that denies reads/writes of the gitignored files which can carry PHI (the app's
`Download` outputs) or local secrets, so that data never enters a context window;
`test_phi_hook_blocks_every_app_download_output` executes it against every `Download` filename in
`streamlit_app.py`, so **a new de-identifying tab fails CI until the hook grows a matching `case`
arm**. It exists because that sync was missed once: `policy_anonymized.txt` shipped with the
`Policy de-ID` tab, was gitignored the same day, and stayed readable through the plain `Read` tool
for a month. The hook matches on basename and does not cover `Bash(cat ...)`, so treat it as a guard
against accidental reads, not as containment.

Model tests (`test_pii_model.py` + the `@pytest.mark.model` tests in `test_engine.py` and
`test_service.py`) are **skipped by default** and drive the real engine via the shared `loader` fixture; `--run-model` opts
in, wired in `tests/conftest.py` (`pytest_addoption` + `pytest_collection_modifyitems`, plus the
session-scoped `loader` and a `note` fixture). The zero-shot model test
(`test_engine_extract_zero_shot_detects_user_labels`) is **doubly gated** — `@pytest.mark.model`
*and* `pytest.importorskip("gliner")` — so CI (neither flag nor extra) never downloads it; it also
skips the `loader` fixture, since the GLiNER path bypasses the shared loader.

Note: `ty` targets Python 3.10 (the minimum). openmed ships inline type hints — `deidentify(method=…)`
expects the `Literal` of the seven method names — so the `DeidMethod` alias (in `engine.py`,
re-exported by `validation.py`) must stay in sync; the guard above enforces it. Tests pass the
`PIIEngine` seam a structural stub via `typing.cast` (the repo convention).

## Conventions when editing

- **Never bypass `service.py`.** Both surfaces call `service.*`, never `PIIEngine` directly — the
  sole exception is `main.py`'s `/compat`, which needs raw entity objects.
- **The UI layer never imports `openmed`.** `streamlit_app.py`/`ui_helpers.py` import only
  `openmed_studio` + `streamlit`; registry metadata is baked into `engine.py` for exactly this
  reason, and the drift guards keep the baked copy honest.
- **Tests stub the seam structurally:** `PIIEngine(loader=cast("ModelLoader", object()))` — no model,
  no `openmed` import. A fast test must never download anything.
- **Adding a de-identifying tab, in order** (this ritual drifted once and left a PHI-carrying download
  readable for a month — see the PHI-hook note under "Test layout"):
  1. `validation.py` — the request model (`extra="forbid"`), plus any new bound primitive.
  2. `service.py` — an entry point that validates → `_run`s the engine → adapts to plain dicts.
  3. `engine.py` — only if a new openmed call or registry entry is needed; keep the `_lock`.
  4. `main.py` — a thin route + typed response model, if the capability should be served.
  5. `streamlit_app.py` — the tab; route its submit through `_submit_deidentify` so the Re-identify
     handoff can't drift.
  6. **`.gitignore` *and* `.claude/hooks/block-phi-paths.sh`** — add the new `Download` filename to
     both. `test_phi_hook_blocks_every_app_download_output` derives its set from `streamlit_app.py`,
     so CI fails until you do.
- **After any openmed/torch/transformers bump:** run `uv run pytest --run-model` (fast tests stub the
  model and cannot catch a load failure) and re-verify the gliner fork with
  `uv export --extra gliner | grep transformers`.

## How it works

- **Backend:** the default dependency is `openmed[hf]` (Hugging Face / PyTorch), which runs
  everywhere (CPU, CUDA, Apple MPS). The `mlx` extra adds Apple's native MLX backend
  (Apple-Silicon-only). openmed auto-detects the backend (`openmed/core/backends.py`
  `get_backend`): it prefers MLX on Apple Silicon when `mlx` imports, else HuggingFace.
  `PIIEngine(backend=...)` / the `OPENMED_STUDIO_BACKEND` env var pin it explicitly (`"mlx"` raises
  off-Apple; `None`/unset = auto). The default English model is **not** in openmed's
  `_MLX_MODEL_MAP`, so on MLX it converts on-the-fly on first run into `<cache_dir>/<org>_<repo>/`
  — `~/.cache/openmed/OpenMed_OpenMed-PII-SuperClinical-Small-44M-v1/`, beside the HF snapshots.
  (`openmed/mlx/inference.py::_resolve_mlx_model` falls back to `~/.cache/openmed/mlx` only for a
  falsy `cache_dir`, and `OpenMedConfig.__post_init__` fills a `None` one with `~/.cache/openmed`,
  so the app's loader never reaches that path.) To skip conversion, pass a pre-converted **Hub
  repo id** ending in `-mlx` (e.g. `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx`) as
  `model_name`; `_resolve_mlx_model` downloads it as-is. Only the HTTP API can (the UI has no
  PII-model picker), and only as a Hub id: `validation._check_model_name` rejects absolute paths,
  so a local conversion directory is not a supported route.
- **Model download:** the first run pulls a model from the HF Hub and caches it under
  `~/.cache/openmed`; later runs are offline. The default PII model is the small
  `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` (~44M params).
- **Model reuse:** the app builds one `PIIEngine` (one shared `ModelLoader`) cached via Streamlit's
  `st.cache_resource`, so the PII model loads at most once per process and is reused across every tab
  and request. The engine pattern (construct one `ModelLoader`, pass `loader=` to every call) is the
  documented best practice. The shared loader dispatches/caches by `model_name`, so the `Clinical
  NER` tab loads a per-domain NER model (~141M each) into the *same* loader on first use of that
  domain — switching domains loads another model rather than rebuilding the loader.
- **Python:** `requires-python = ">=3.10"`; verified on 3.11, but uv may pick a
  newer interpreter (e.g. 3.13) for `.venv`.
### Core modules (`openmed_studio/`)

- **App structure:** `engine.py`/`service.py`/`validation.py` are the **framework-free core** (no
  Streamlit, no HTTP); `main.py`/`__main__.py` are the FastAPI/uvicorn HTTP layer *over* that core
  (the only files that import a web framework), mirroring how `streamlit_app.py`/`ui_helpers.py` are
  the UI layer over it:
  - `engine.py` — the `PIIEngine` (one shared `ModelLoader`, lazy load) plus the model registry:
    - *Loader + wrappers:* the `ModelLoader` is always built with
      `OpenMedConfig(backend=self.backend, torch_attention_backend="eager")` — `backend` stays
      `None` unless pinned, so openmed still auto-detects it, while `torch_attention_backend="eager"`
      is pinned deliberately (the OpenMed DeBERTa-v2 models have no SDPA kernel — see "Known
      gotchas"); thin wrappers cover `extract_pii`/`deidentify`/`reidentify`. A `threading.Lock`
      (`self._lock`) serializes the four model-calling methods
      (`extract`/`analyze`/`extract_zero_shot`/`deidentify`) so concurrent inference (FastAPI
      requests, cached Streamlit sessions sharing the one engine) runs one call at a time and the
      first-call model download can't race; `reidentify` is exempt (pure regex, a `@staticmethod`).
    - *De-identify options* (per-call, surfaced in each de-identifying tab's `Advanced` expander,
      conditioned on the method): `consistent`/`seed`/`locale` are the surrogate-method
      (`replace`/`format_preserve`) determinism knobs
      (`locale` e.g. `pt_BR` overrides the locale openmed derives from `lang`);
      `date_shift_days`/`keep_year` drive `shift_dates`; `use_safety_sweep` (default `True`) is a
      deterministic structured-identifier sweep run after detection that redacts identifiers
      `extract_pii` (no sweep) misses — the `Detect` caption flags this. `use_smart_merging`
      (default on) is forwarded too. Every method delegates straight to openmed (see "Known gotchas"
      for the `shift_dates` fix).
    - *Policy anonymization:* `deidentify` also forwards a `policy` param (a compliance-profile name,
      e.g. `"hipaa_safe_harbor"` — a `Policy` `Literal` value; `None` by default = no policy). When
      set it **overrides `method`** (openmed assigns a per-label action from the profile, so the
      `Policy de-ID` tab sends **no** method) and the profile — not the caller — decides
      reversibility: the engine passes `keep_mapping=False`. See "Known gotchas" → *A `policy`
      overrides `method`* for the OR semantics and which five profiles keep a key.
    - *Clinical NER:* `analyze(text, *, model_name, confidence_threshold=0.0, aggregation_strategy,
      group_entities)` delegates to `analyze_text`. `model_name` is **required** (NER is one model
      per domain; an absent one silently falls back to openmed's disease-only default). `analyze_text`
      returns an `AnalyzeResult` *object* (its `output_format="dict"` is a misnomer; it was
      `PredictionResult` before openmed 2.0), so `_entities`
      unwraps `.entities`; no `lang` (analyze_text has none).
    - *Zero-shot (GLiNER):* `extract_zero_shot(text, *, model_name, labels, confidence_threshold=0.6)`
      delegates to `openmed.ner.infer` (NOT `analyze_text`). It resolves the registry alias to the HF
      repo id (`get_all_models()[model_name].model_id`), fabricates a **one-entry in-memory**
      `ModelIndex(ModelRecord(id=repo_id, family="gliner"), generated_at=_ZERO_SHOT_INDEX_EPOCH,
      source_dir=Path())` — because `infer`'s default on-disk index isn't shipped — and returns
      `NerResponse.entities` (unwrapped by `_entities`). This path **deliberately bypasses the shared
      loader** (openmed's GLiNER inference has its own cache and is torch-only; it doesn't need the
      DeBERTa-v2 eager pin because neither gliner nor openmed's GLiNER path ever requests SDPA —
      see "Known gotchas"), so `is_loaded` stays False after a zero-shot call and the UI tracks
      loaded domains itself. Two static helpers back the
      tab without a UI-side openmed import: `zero_shot_available()` (→ `is_gliner_available()`, so the
      tab can show install instructions instead of failing) and `default_labels(label_domain)` (→
      `get_default_labels`, seeding the label picker live). See "OpenMed API" → *Zero-shot (GLiNER)*
      for the `.score` field (openmed's zero-shot `Entity` has no `.confidence`).
    - *Registry:* defines the `DeidMethod`/`Backend` `Literal`s, `DEFAULT_PII_MODEL`,
      `DEFAULT_NER_MODEL`, the `NerModel` `NamedTuple`, and `NER_MODELS` — a curated
      `dict[domain → NerModel]` of one ~141M "superclinical" model per category. (There is no
      `Medical` domain: openmed 2.0 dropped the broad 434M `clinicalner` alias it used, leaving that
      category with no plain-PyTorch token-classification model — only an `_onnx_android` build —
      so the 10 remaining domains now mirror `ZERO_SHOT_MODELS` exactly.) Each `NerModel` bakes
      registry metadata (`alias`, `display_name`, `recommended_confidence`, `entity_types`,
      `params`) so the UI needs **no runtime openmed
      import**; the drift guard pins it to the live registry. Zero-shot has its own parallel
      `ZeroShotModel` `NamedTuple` + `ZERO_SHOT_MODELS` (10 domains, one Small/166M GLiNER checkpoint
      each, mirroring the NER domain names one-for-one) + `DEFAULT_ZERO_SHOT_MODEL`. `ZeroShotModel`
      adds a `label_domain` field (an `openmed.ner.available_domains()` key used to seed the label
      picker) and, unlike `NerModel`, its `entity_types` are the checkpoint's *training focus* (a
      "tuned for" hint), **not** the output vocabulary — zero-shot's output labels are whatever the
      user types. Its drift guard pins alias/`recommended_confidence`/`entity_types`/`label_domain` but
      **not** `info.category` (zero-shot models bucket into only a few broad categories, not per-domain).
      Policy anonymization has its own parallel `Policy` `Literal` (the 20 canonical policy names,
      mirroring `openmed.core.policy.PolicyName`; 2.5 added `clinical_preserve`) + `PolicyModel`
      `NamedTuple` + `POLICY_MODELS` (`dict[friendly display name → PolicyModel]`, 20 entries) +
      `DEFAULT_POLICY_MODEL`. Unlike `NerModel`/`ZeroShotModel` a policy loads **no model of its
      own** — it reuses the shared PII model and only changes the per-label action — so
      `PolicyModel` bakes the *behavioral* flags the preview surfaces (`default_action`/
      `keep_mapping`/`safety_sweep_mandatory`, pinned against the live `PolicyProfile` by the drift
      guard) plus a hand-authored `description` (openmed ships none — 2.5's `clinical_preserve`
      carries a `metadata["purpose"]` line, but it over-promises "treatment dates", so it is
      deliberately not reused), not model-identity fields. **Write a `description` from
      `load_policy(name).actions` / `.policy_label_actions`, never from `default_action`** — openmed
      never reaches a profile's fallback (`_canonical_actions` forces `actions` to cover all 139
      canonical labels — 2.2 added `ALLERGEN`/`ALLERGY_CRITICALITY`/`REACTION_MANIFESTATION`/
      `REACTION_SEVERITY` — and `normalize_label` funnels unknowns to `OTHER`), so a profile can
      declare `replace` while masking 123 of 139 labels (South Africa POPIA does; Nigeria NDPA
      masks 130). The `Policy de-ID` preview therefore does **not** render `default_action`; the
      field stays baked only so the guard keeps pinning it. **And whenever the sweep runs — forced
      by `safety_sweep_mandatory`, or by the tab's default-on toggle — confirm every "keeps X"
      claim with a real run:** openmed's pipeline drops `keep` spans *before* the stage-9 safety
      sweep (`core/pipeline.py`), a span the sweep re-detects under a kept label falls back to the
      flat method (`mask` — `core/pii.py::_entity_redaction_method`), and `use_safety_sweep=False`
      cannot switch a mandatory sweep off. So the four profiles whose rules keep dates and facility
      names — Research Limited Dataset, India Health ID and Clinical Preserve (sweep forced) and
      Clinical Minimal Redaction (sweep on by default) — still mask whatever the sweep's English
      patterns match (listed at `POLICY_MODELS["Research Limited Dataset"]`): `03/14/2024`-, ISO-,
      `March 14, 2024`- and `14 March 2024`-style dates, names ending in Clinic / Health Center /
      Dispensary / District|Referral|Mission Hospital, ZIPs within 100 characters of "zip" or
      "postal". Other full-date forms (`March 14th, 2024`, `2024/03/14`, `14.03.2024`), month-year,
      yearless and relative dates, ordinary `… General Hospital`/`… Medical Center` names and bare
      ZIPs survive — so "the sweep masks full dates" over-promises. Those descriptions are worded to
      that boundary in both directions (over-promising masking is the dangerous error in a PHI
      tool); `test_engine_sweep_overrides_keep_rules_only_where_its_patterns_match` (`--run-model`)
      pins it. **Nor may a description say a profile masks clinical terms:** 10 of the 20 map
      `DISEASE`/`DRUG`/… to `mask`, but the shared PII model has no clinical labels (its 54 entity
      types are identifiers and personal attributes) and the sweep no clinical patterns, so
      diagnoses pass through every profile — hence "every detected span". Two more traps the
      2.x catalog added: "surrogate policy" ≠ "reversible policy" (**four of the nine**
      `replace`-declaring profiles keep no key — Australia Privacy Act, India DPDP, ZA POPIA, NG
      NDPA; `test_policy_models_resolve_in_openmed` now fails any description that promises
      "reversible" against `keep_mapping=False`), and four of the nine APAC/African 2.x profiles
      (Malabo, Kenya DPA, Egypt PDPL, Morocco 09-08) are `mask`-everything and behaviorally
      identical to `strict_no_leak` — their descriptions say so, and
      `test_engine_mask_all_profiles_match_strict_no_leak` pins it.
  - `validation.py` — the Pydantic request models (`ExtractRequest`, `NerRequest`, `ZeroShotRequest`,
    `AnonymizePolicyRequest`, `DeidentifyRequest`, `DeidentifyBatchRequest`, `ReidentifyRequest`, all
    `extra="forbid"`) plus
    the bound primitives: `ClinicalText`/`MAX_TEXT_CHARS` (from `OPENMED_STUDIO_MAX_TEXT_LENGTH` via
    `_max_text_chars` at import), `MAX_BATCH_ITEMS`, `MAX_MAPPING_ENTRIES`, `Lang`,
    `_check_model_name`, `RequiredModelName` (the non-optional model id `NerRequest`/`ZeroShotRequest`
    require), and `_check_locale` (a format guard on the optional `replace` `locale`). `ZeroShotRequest`
    adds `labels` — a `ZeroShotLabels` type whose `_check_zero_shot_labels` `AfterValidator` strips,
    drops blanks, bounds each label to `MAX_ZERO_SHOT_LABEL_CHARS` (80), dedups case-insensitively
    (harmless duplicates collapse; unknown *fields* still fail via `extra="forbid"`), and caps the set
    at `MAX_ZERO_SHOT_LABELS` (30) — all with errors that name the cap, never a label value (PHI-safe).
    `AnonymizePolicyRequest` requires a `policy` field (the closed `Policy` `Literal`, imported from
    `engine.py`, so Pydantic rejects a typo/unknown policy PHI-safely *before* the engine) and, unlike
    `DeidentifyRequest`, carries **no `method`** (the policy overrides it) and **no `keep_mapping`** (the
    policy decides reversibility) — passing either is a forbidden extra field. Imports only
    `pydantic`/`os`/`re` — no web framework — so it doubles as the in-process validation layer.
    Re-exports `DeidMethod` and `Policy`.
  - `service.py` — the single in-process chokepoint (framework-free); **both** surfaces (the Streamlit
    UI and the FastAPI service) funnel every engine call through it — the two opt-in `/compat` routes
    are the one exception (they call the engine directly for the raw entity objects, but still validate
    via the `Compat*` models and reuse `service._run`) — so nothing bypasses validation:
    - `resolve_backend()` (reads `OPENMED_STUDIO_BACKEND`) and `build_engine()` (the `PIIEngine`
      factory both the UI's `st.cache_resource` and the API's `get_engine` wrap).
    - `ServiceError` carries a transport-neutral `.kind`
      (`validation`/`bad_options`/`unavailable`/`dependency`/`internal`, `ServiceErrorKind`): the
      Streamlit UI ignores it (renders only the message), while `main.py` maps it to an HTTP status —
      so the seam stays framework-free (no status codes) yet a served caller gets the right response.
    - `_validate()` — `model_validate()`, raising a **PHI-safe** `ServiceError` (`kind="validation"`)
      from only `loc`/`msg` (never Pydantic's `input`).
    - `_run()` — translates `ValueError`→`kind="bad_options"`, `RuntimeError`/`OSError`→
      `kind="unavailable"`, `ImportError`→`kind="dependency"`+pass-the-message (openmed's
      `MissingDependencyError` subclasses `ImportError`; the zero-shot tab/route surfaces its "run
      `uv sync --extra gliner`" hint through this branch), and any other exception→`kind="internal"`
      (generic message, detail to the log) into `ServiceError` (the old 400/503 split, now carried by
      `.kind` and capability-neutral since NER/zero-shot flow through it).
    - the dict adapters (`_entity_dict`, `_deidentify_dict`) and the entry points
      `extract`/`analyze`/`extract_zero_shot`/`deidentify`/`anonymize_policy`/`deidentify_batch`/
      `reidentify`, which validate → call the engine → adapt to plain dicts. `analyze` and
      `extract_zero_shot` reuse `_validate`/`_run`/`_entity_dict` verbatim; `_entity_dict` falls back to
      `.score` when there's no `.confidence`, so it handles openmed's zero-shot `Entity` (which exposes
      `.score`) unchanged. `anonymize_policy` reuses `deidentify`'s path (Option A): it validates an
      `AnonymizePolicyRequest`, calls `engine.deidentify(policy=…, keep_mapping=False)` (**no** method —
      the policy overrides it; **not** forced-`keep_mapping` — the policy decides), and reuses
      `_deidentify_dict` verbatim (asked to *surface* whatever mapping the policy produced, with the
      policy name in the `method` slot). Because it routes through `engine.deidentify`, **no test stub
      needs a new method**.
    - `deidentify_batch` isolates each note: a per-note `ValueError` becomes an `{"ok": False}` row
      so one bad note doesn't abort the batch, while a backend `RuntimeError`/`OSError` propagates
      through `_run` and aborts the whole batch.
  - `__init__.py` — re-exports `DEFAULT_PII_MODEL`, `DEFAULT_NER_MODEL`, `DEFAULT_ZERO_SHOT_MODEL`,
    `DEFAULT_POLICY_MODEL`, `NER_MODELS`, `ZERO_SHOT_MODELS`, `POLICY_MODELS`, `PIIEngine`, and
    `__version__`.
  - `main.py` — the FastAPI service (the only core module that imports a web framework): `create_app()`
    (and the module-level `app`). Every route is a **thin wrapper over `service.*`** — it declares a
    `validation.py` request model as the body (free OpenAPI + auto-422) and returns the seam's dict,
    coerced into a typed response model (`Entity`/`EntitiesResponse`/`DeidentifyResponse`/
    `BatchItemResult`+`DeidentifyBatchResponse`/`ReidentifyResponse`, all defined here). Seven model
    routes + `GET /health` (8 mounted; 10 with `/compat`):
    `POST /pii/extract`→`extract`, `POST /ner`→`analyze`, `POST /zero-shot`→`extract_zero_shot`,
    `POST /pii/deidentify`→`deidentify`, `POST /pii/deidentify/batch`→`deidentify_batch`,
    `POST /pii/anonymize-policy`→`anonymize_policy`, `POST /pii/reidentify`→`reidentify`, `GET /health`
    (unauthenticated, for probes), and the opt-in two-route `/compat` router. A single `@app.exception_handler(
    ServiceError)` maps `.kind`→status and builds the `{"error":{code,message,details}}` envelope (so
    routes need no try/except); handlers for `StarletteHTTPException` (401 etc.) and
    `RequestValidationError` (**PHI-safe** — drops Pydantic's `input`) reuse the same envelope. Auth is
    `require_api_key` (a no-op unless `OPENMED_STUDIO_API_KEY` is set; `create_app` warns at startup when
    it's unset); an opt-in `OPENMED_STUDIO_PRELOAD` lifespan warms the model in a threadpool. The
    `/compat` surface (mounted only when `OPENMED_STUDIO_COMPAT` is truthy) mirrors OpenMed's own REST
    shape (`pii_entities`/`num_entities_redacted`/`timestamp`/echoed `original_text`, `keep_alive`
    ignored) for `/compat/pii/{extract,deidentify}` only; it calls the engine directly for the raw
    entity objects upstream's shape needs (the seam's adapter drops `redacted_text`/`metadata`) but
    reuses `service._run` for the same error translation. Caveat: this compat shape is **hand-authored
    against OpenMed's REST spec** and — unlike everything in "OpenMed API (verified against installed
    v…)" — cannot be pinned by a drift guard (those fields don't exist in the installed `openmed`
    package), so treat it as best-effort parity, not a verified contract. **FastAPI 0.139 includes
    routers lazily**, so
    to introspect routes use `app.openapi()["paths"]`, not `app.routes` (which holds an `_IncludedRouter`
    wrapper for a mounted router — see "Known gotchas").
  - `__main__.py` — `python -m openmed_studio` → `uvicorn.run("openmed_studio.main:app", …)` with
    `OPENMED_STUDIO_HOST`/`OPENMED_STUDIO_PORT` (defaults `127.0.0.1:8080`).

  It stays a uv **non-package** project, so pytest imports `openmed_studio` via the repo root on
  `sys.path` (`pythonpath = ["."]`; Streamlit adds the app's directory).
### UI (`streamlit_app.py`, `ui_helpers.py`)

- **UI structure:** the Streamlit app lives at the repo root in `streamlit_app.py`; the pure,
  Streamlit-free render helpers live in `ui_helpers.py` so they unit-test without a browser.
  - *App + tabs:* `get_engine` is `service.build_engine` wrapped in `st.cache_resource`; `_call`
    runs a `service.*` function in a spinner and renders any `ServiceError`. `main()` titles the
    page/heading "OpenMed Studio" and lays out the eight tabs (`Detect`→`service.extract`,
    `Clinical NER`→`service.analyze`, `Zero-shot`→`service.extract_zero_shot`,
    `Single note`/`Batch`→`service.deidentify[_batch]`,
    `Anonymize`→`service.deidentify` (`method=replace`), `Policy de-ID`→`service.anonymize_policy`
    (`deidentify(policy=…)`), `Re-identify`→`service.reidentify`), guarded
    by `if __name__ == "__main__"` so importing for tests has no side effects.
  - *Fragments + handoff:* `Detect`/`Clinical NER`/`Zero-shot`/`Batch`/`Re-identify` renderers are
    `@st.fragment` so an in-tab interaction reruns only that tab; `Single note`, `Anonymize`, and
    `Policy de-ID` are **intentionally not**, because their form submit must trigger a full rerun to
    hand `last_deidentified`/`last_mapping` (via `st.session_state`, not widget keys) to `Re-identify`
    (a reversible policy — one of the five `keep_mapping=True` profiles — round-trips this way).
    `_set_handoff` sets the two
    together once per submit — the single security-relevant copy of "so a stale mapping can't
    linger" — *not* on re-render. The shared `_submit_deidentify` helper takes an optional `call=`
    (defaulting to `service.deidentify`; `Policy de-ID` passes `service.anonymize_policy`) so all three
    surfaces share the one submit→call→persist→handoff sequence.
  - *Result persistence:* all five de-identifying surfaces persist their latest result in
    `st.session_state` (`single_result`/`anon_result`/`policy_result` via the shared
    `_submit_deidentify` helper, which centralizes submit→call→persist→handoff so
    Single/Anonymize/Policy can't drift; plus `batch_result` and `reid_result`) and render from there,
    so post-submit reruns (a Download, a
    control tweak, the "Show re-identification key" click) don't blank the panel and a failed/empty
    re-submit warns without losing the last good result (a snapshot caption flags this). The mapping
    is revealed in an `@st.dialog` (`_show_mapping_dialog`) behind a button, not an always-open
    expander.
  - *De-identification controls:* `Method` plus the method-conditional `Advanced` knobs
    (the surrogate methods `replace`/`format_preserve`→consistent/seed/locale,
    `shift_dates`→date_shift_days/keep_year, plus the safety
    sweep) live in `Single note` + `Batch` via a shared `_render_deid_controls(key_prefix=…, lang=…)`
    (above each tab's form, widget keys `key_prefix`-scoped so the tabs don't collide). `Detect` has
    its own confidence slider + smart-merge toggle; `Anonymize` reads the sidebar `Language` and
    carries its own in-form controls (confidence + `Deterministic` in a `[3, 2]` column row, then
    seed/locale in an `Advanced` expander — the same shape as `Policy de-ID`, so all three
    de-identifying tabs read alike). Only `Method`/`Advanced` are per-tab — the sidebar holds just the engine readout
    (model/backend/`is_loaded`, read directly) and the lone global `Language` filter
    (`_render_sidebar` returns the chosen `lang`).
  - *Clinical NER controls* (`_render_ner`, independent of the de-id controls): a domain picker
    (`st.selectbox` over `NER_MODELS`, default `Disease`) sits **outside** the form so selecting a
    domain reruns the fragment and refreshes both a reactive preview (the model's `display_name`,
    size, `entity_types`) and the confidence slider's default (seeded from the model's
    `recommended_confidence`, per-domain keyed). `model_name`
    resolves via `NER_MODELS[domain].alias`. Because `engine.is_loaded` only tracks whether *a* model
    has loaded, the per-domain download wait-hint is driven by a `st.session_state` set of analyzed
    domains (passed to `_call(..., needs_load=...)`), so switching to a not-yet-downloaded domain
    still warns.
  - *Zero-shot controls* (`_render_zero_shot`, `@st.fragment`): first gate on
    `engine.zero_shot_available()` — when the `gliner` extra isn't installed, render the
    `uv sync --extra gliner` install hint (an `st.code`) and **return before the form**, so the tab
    degrades instead of failing. Otherwise a domain picker (`st.selectbox` over `ZERO_SHOT_MODELS`,
    default `Disease`) sits **outside** the form (reactive preview + per-domain confidence default like
    NER); inside the form, an `st.multiselect(accept_new_options=True, max_selections=…)` **seeded from
    `engine.default_labels(model.label_domain)`** lets the user edit the suggested labels or add their
    own free-text ones. `model_name` resolves via `ZERO_SHOT_MODELS[domain].alias`; because the
    zero-shot path bypasses the shared loader entirely (so `is_loaded` is *always* blind to it), a
    per-domain `st.session_state` set (`zs_analyzed_domains`) drives the wait-hint. The keys are
    `zs_`-scoped so they don't collide with the NER tab's identically-labelled widgets.
  - *Policy anonymization controls* (`_render_policy_anon`, **not** a fragment — it feeds the
    Re-identify handoff like `Anonymize`): a policy picker (`st.selectbox` over `POLICY_MODELS`, default
    `HIPAA Safe Harbor`) sits **outside** the form (reactive preview: display name + canonical `name`
    / reversible? / safety-sweep, then the hand-authored `description` — deliberately **not**
    `default_action`, which openmed never reaches (see the *Registry* note), so it misleads as a
    headline; it appears only in the `Advanced` caption. Refreshed on pick — a full rerun like
    `Single note`'s method picker). There is **no Method control** (the policy selects the action). Inside the form:
    text area, confidence slider, and an `Advanced` expander with the surrogate knobs
    (consistent/seed/locale — they apply to the `replace`-based policies) + the safety-sweep toggle.
    `build_policy_opts` shapes the payload (no `method`, no `keep_mapping`); the tab submits via
    `_submit_deidentify(call=service.anonymize_policy)` and renders through the shared
    `_render_deid_result`. All widget keys are `policy_`-scoped; the text-area label is distinct
    ("Clinical note to anonymize under a policy") so it doesn't collide with `Anonymize`'s.
  - *Rendering:* `_render_highlight(text, entities)` (shared by `Detect`, `Single`, `Anonymize`,
    `Policy de-ID`, `Clinical NER`, and `Zero-shot`) renders the highlighted text plus its legend, label-agnostic so it
    handles NER's UPPERCASE labels and zero-shot's arbitrary user-typed labels unchanged (every label
    is HTML-escaped, so a user-supplied label can't inject markup). `ui_helpers.py`'s `render_highlighted`/`render_legend` are
    **theme-agnostic**: a translucent per-label tint from `PALETTE`/`color_for` plus `color: inherit`,
    so the marks read on light or dark with no runtime theme detection (`render_plain`/
    `build_base_opts`/`build_batch_table` are kept separate for browserless unit tests). `PALETTE` is
    the **nine** Nord accents (five Aurora, four Frost), mirroring `.streamlit/config.toml`; nine and
    not ten is load-bearing, because `color_for` hashes with `sum(ord(c)) % len(PALETTE)` and at ten
    `first_name` and `date` — the most common pair in a clinical note — collide. Per-hue alphas keep
    every tint visible on both Nord's `#2e3440` and white while holding text above WCAG AA. The
    de-identified output offers a `Download` button (**no** copy-to-clipboard — the in-process tool
    deliberately avoids sending PHI to a browser-side clipboard component); every entity table goes
    through the shared `_render_entity_table` (confidence as a `ProgressColumn` plus a `placeholder`,
    because zero rows here means "nothing cleared the confidence threshold", not "nothing ran");
    metric cards sit in an `st.container(horizontal=True)` at `width="content"` carrying their own
    tab's Material Symbol (a bare `st.metric` defaults to `width="stretch"`, so under `layout="wide"`
    a lone bordered card spanned the whole page to show one number), and `Detect`/`Clinical NER`/
    `Zero-shot` add a `Distinct types` card whose bar sparkline is the per-label counts in first-seen
    order — the order `render_legend` also uses, so the bars line up with the legend pills below;
    the `Batch` table sets `row_height` because its two wide columns hold multi-line notes that
    otherwise truncate to one line; and `Download`/`Re-identify` confirm with an `st.toast`. The UI consumes the plain dicts `service`
    produces (`result["entities"]`, `result["deidentified_text"]`, `result.get("mapping")`). The
    confidence slider defaults to `0.5` (the de-identify default is `0.7`).
  - *Config:* `streamlit>=1.61` is a core dependency — the floor is set by `st.metric(icon=…)` on
    every KPI card, which 1.60 and earlier reject with a `TypeError` (the horizontal/
    `height="stretch"` flex containers are older). `.streamlit/config.toml` is **Nord, dark only**: one flat `[theme]` (plus
    `[theme.sidebar]`) and *no* `[theme.light]`/`[theme.dark]`, which is precisely what removes the
    light/dark selector — Streamlit offers it only when both mode sections exist
    (`runtime/app_session.py` populates `custom_theme.light`/`.dark` only from those sections), so
    re-adding either brings the toggle back. It sets the Polar Night backgrounds / Snow Storm text /
    Frost accents, `baseRadius`/`buttonRadius` `4px` with widget+sidebar borders on, and an Aurora
    `red`/`orange`/`yellow`/`green`/`violet` + Frost `blue` semantic palette that `ui_helpers.PALETTE`
    mirrors, so status accents and entity marks come from one set of nine colors. It deliberately
    omits the upstream Nord template's `font`/`codeFont` (Inter + JetBrains Mono via
    fonts.googleapis.com) — a tool that already sets `gatherUsageStats = false` shouldn't make an
    outbound CDN call per page load, and the fonts would silently fall back in an air-gapped deploy;
    the font *metrics* (`baseFontSize`/`headingFontSizes`/…) are family-independent and stay.
    `gatherUsageStats = false` (a clinical-text tool shouldn't phone home);
    and `[client] showErrorDetails = "none"`, so a traceback (which can quote note text) never reaches
    the browser — full detail goes to the server console, meaning **debug from the terminal, not the
    page**.
    Local secrets go in the gitignored `.streamlit/secrets.toml`, and the **five** download outputs
    (`deidentified.txt`/`deidentified_batch.json`/`anonymized.txt`/`policy_anonymized.txt`/
    `reidentified.txt` — same order as `.gitignore` and the PHI hook's `case` arm, so the three lists
    diff cleanly) are gitignored too, since they can carry PHI or its surrogates.
### The two surfaces

- **They are parallel, not layered:** both import `service.py`; neither calls the other, and the
  Streamlit app runs the model **in-process** — there is **no HTTP client and no `requests` dependency
  in the UI, and re-adding one is a regression.** `main.py`/`__main__.py` + `fastapi`/`uvicorn` (core)
  and `httpx` (dev, for `TestClient`) exist for what a *served* surface needs and the UI doesn't:
  API-key auth (`OPENMED_STUDIO_API_KEY` / `X-API-Key`), the `{"error":{code,message,details}}` JSON
  envelope + PHI-safe 422, `/health`, the opt-in `/compat` OpenMed-REST surface
  (`OPENMED_STUDIO_COMPAT`), and the startup preload (`OPENMED_STUDIO_PRELOAD`) — env knobs are
  tabled under "Commands".
  Security posture: still a **local / small-scale** tool — an unset API key runs
  the service **unauthenticated** (with a loud startup warning), so set the key (and use TLS or your own
  reverse proxy) before exposing it or processing real PHI. The guarantees that protect the *model*
  regardless of surface are enforced in-process by `service.py` (text/batch/mapping caps,
  value/enum/format checks, backend pinning, no input echo on a validation error) plus the engine's
  concurrency lock — so both the UI and the API inherit them.

## OpenMed API (verified against installed v2.5.0)

Top-level imports: `from openmed import extract_pii, deidentify, reidentify, analyze_text, ModelLoader, OpenMedConfig`.
Registry helpers used by the NER picker / drift guard: `get_all_models()` (dict alias→ModelInfo),
`list_model_categories()`.

- `extract_pii(text, model_name=<default>, confidence_threshold=0.5, config=None, use_smart_merging=True, lang="en", cache_results=False, max_cache_entries=128, normalize_accents=None, *, preserve_whitespace=False, locale=None, loader=None, batch_size=None, num_workers=None, custom_recognizer=None, abdm=None, code_mixed=False, token_language_tags=None, lid_model=None, transliterated_name_config=None, budget=None)`
  returns a `PredictionResult` object (an object with `.entities`, like `analyze_text`'s
  `AnalyzeResult` below — but a plain dataclass, not a `Mapping`) whose `.entities` are PII
  predictions with `.label`/`.text`/`.start`/`.end`/`.confidence` — the engine's `_entities` unwraps
  it. Labels are **lowercase** (`first_name`, `last_name`, `date`, `ssn`, `phone_number`, …). The
  engine forwards `confidence_threshold`/`use_smart_merging`/`lang`/`model_name`/`loader` only (it
  owns loading, so `config`/`normalize_accents`, 1.8.0's `locale`/`cache_results`/
  `max_cache_entries`/`custom_recognizer`, and 2.x's `preserve_whitespace`/`batch_size`/`num_workers`/
  `abdm`/`code_mixed`/`token_language_tags`/`lid_model`/`transliterated_name_config`/`budget` are not
  threaded). No drift guard pins this split (unlike
  `deidentify`/`analyze_text`), since the unforwarded params are all optional with safe defaults.
- `deidentify(text, method="mask", model_name=<default>, confidence_threshold=0.7,
  use_smart_merging=True, keep_mapping=False, consistent=False, seed=None, locale=None,
  date_shift_days=None, keep_year=False, lang="en", use_safety_sweep=True, audit=False,
  loader=None, …)` — the installed v2.5.0 signature (unchanged from 2.1.0) **also** accepts `shift_dates` (a bool,
  distinct from `method="shift_dates"`), `normalize_accents`, `config`, `policy`,
  `calibration_thresholds_path`, 1.7.0's `patient_key`/`date_shift_max_days`/
  `date_shift_secret`/`surrogate_vault`/`custom_recognizer`/`cache_results`/`max_cache_entries`,
  and 2.x's India/code-mixed i18n plumbing `abdm`/`code_mixed`/`token_language_tags`/`lid_model`/
  `transliterated_name_config` plus the `budget` (`RequestBudget`) accounting object. All are
  excluded — but note **`abdm=None` means AUTO, not off**:
  `openmed/core/custom_recognizer.py::abdm_mode_enabled` switches the India ABDM recognizers on
  whenever `policy == "india_dpdp_act"`, `locale` ends in `_in`, or `lang` is `hi`/`te` — all three of
  which the app forwards — so leaving it unset is a deliberate "let openmed decide per context"
  (forcing `False` would gut the India DPDP policy the `Policy de-ID` tab now offers), not an inert
  omission. The other four and `budget` do default to off/`None`.
  (Note: `keep_year` now defaults to `False` upstream, but the app always passes its own value —
  default `True` in `_DeidentifyOptions`/`PIIEngine.deidentify` — so the flip is inert.)
  `date_shift_days` (and 1.7.0's `patient_key`/`date_shift_max_days`/`date_shift_secret`) are
  **validated, not ignored**: `core/pii.py::_resolve_deidentification_method` raises
  `InputError("date_shift_days requires method='shift_dates'. …")` when paired with any other method
  (2.3's `openmed.core.errors.InputError` subclasses `ValueError`, as the plain `ValueError` it
  replaced did), so the seam surfaces a `bad_options` 400. The UI only renders them under `shift_dates`, but the HTTP
  request model accepts them with any method.
  It returns a `DeidentificationResult` with
  `.deidentified_text`, `.pii_entities`, `.mapping` (or an `AuditReport` when `audit=True` —
  1.7.0 types the return as `DeidentificationResult | AuditReport`; the app's engine returns it
  as `Any`, never sets `audit`, and `tests/test_pii_model.py` casts it back to
  `DeidentificationResult`).
  The app's engine forwards `method`/`confidence_threshold`/`use_smart_merging`/`keep_mapping`/
  `consistent`/`seed`/`locale`/`date_shift_days`/`keep_year`/`use_safety_sweep`/`policy` plus
  `lang`/`model_name`/`loader`; it deliberately does **not** forward `audit` (would flip the
  return type), `config` (the engine owns loading via `loader=`), or the advanced
  `shift_dates`/`normalize_accents`/`calibration_thresholds_path`, the 1.7.0
  `patient_key`/`date_shift_max_days`/`date_shift_secret`/`surrogate_vault`/`custom_recognizer`/
  `cache_results`/`max_cache_entries` knobs, or the 2.x
  `abdm`/`code_mixed`/`token_language_tags`/`lid_model`/`transliterated_name_config`/`budget` ones.
  `policy` (an `Optional[str]` — a canonical name from `openmed.core.policy.list_policies()`, default
  `None`) selects a **regulatory compliance profile** that assigns a per-label action; the `Policy
  de-ID` tab drives it. The policy machinery lives in `openmed.core.policy` (**not** top-level
  exported — `from openmed.core.policy import PolicyName, list_policies, load_policy`), which ships 20
  built-ins (`hipaa_safe_harbor`, `hipaa_expert_review_assist`, `gdpr_pseudonymization`,
  `gdpr_art9_health`, `research_limited_dataset`, `strict_no_leak`, `clinical_minimal_redaction`,
  2.5's `clinical_preserve`, `canada_pipeda`, `uk_ico_anonymisation`, `australia_privacy_act`, plus 2.x's `china_pipl`,
  `india_dpdp_act`, `africa_malabo_baseline`, `za_popia`, `ng_ndpa`, `ke_dpa`, `india_health_id`,
  `eg_pdpl`, `ma_law_09_08`) + 8 aliases (`POLICY_ALIASES`) — the `policies/fhir_*`/`omop_*` JSONs
  2.5 also ships belong to `openmed.structured.schema_policy` and are *not* `deidentify(policy=)`
  profiles (`load_policy` rejects them); `load_policy(name)`
  returns a frozen `PolicyProfile` (`default_action`/`keep_mapping`/`reversible_id`/
  `safety_sweep_mandatory`/…) the app bakes into `POLICY_MODELS`. (Custom policies can't ride the
  public `deidentify(policy=str)` API — the name must be canonical — so they're out of scope; the
  `Anonymizer`/`AnonymizerConfig` classes are the low-level Faker surrogate generator `method=replace`
  already uses internally, **not** a policy engine.)
  `tests/test_engine.py::test_deidentify_forwards_every_openmed_param_or_allowlists_it`
  introspects this real signature and pins the forwarded-vs-excluded split so it can't drift.
  `use_safety_sweep=True` runs a post-detection structured-identifier sweep `extract_pii` has no
  equivalent of (see the `engine.py` notes for how the UI surfaces it).
  Methods: `mask`, `remove`, `replace` (Faker surrogates — `consistent=True, seed=N` for
  determinism, `locale="pt_BR"` etc. for a specific surrogate locale, exposed in the de-identifying
  tabs' `Advanced` expander), `hash`, `shift_dates`, and `format_preserve` (a `replace` sibling
  added in 1.7.0 — synthetic *format-preserving* surrogates for structured identifiers, masking
  free-text entities like names it can't shape-preserve; shares `replace`'s consistent/seed/locale
  knobs). `aadhaar_mask` (added in 2.0) is India-specific: a value passing openmed's Aadhaar checksum
  (`pii_i18n.validate_aadhaar`) renders as `XXXX XXXX NNNN`, and everything else falls through to the
  ordinary mask placeholder — so **with `keep_mapping=False`** it is byte-identical to `mask` on notes
  with no Aadhaar number (verified end-to-end on the app's own `EXAMPLE_NOTE`). With
  `keep_mapping=True` — **the UI default** — it diverges: `aadhaar_mask` is outside openmed's
  unique-placeholder set, so repeated labels collapse onto one placeholder instead of `_2`/`_3` (see
  the occurrence-mapping gotcha). It is the **only method that leaves part of
  an identifier in the output** (the UIDAI display form keeps the last four digits), making it
  strictly weaker than `mask`, so `_render_deid_controls` renders an `st.warning` when it is picked —
  a real callout, not a caption prefixed with `:material/warning:`, which read as ordinary grey
  metadata (pinned by `tests/test_ui_app.py::test_aadhaar_mask_warns_that_it_retains_digits`, which
  also asserts the text stays *out* of the caption stream so it can't regress) and the
  `DeidMethod` literal lists it **last** — the guard compares sets, so order is the app's to choose,
  and `streamlit_app.py`'s `METHODS = list(get_args(DeidMethod))` feeds that order into the picker.
- `reidentify(deidentified_text, mapping)` → original text (use with `deidentify(..., keep_mapping=True)`).
- `analyze_text(text, model_name="disease_detection_superclinical", *, loader=None,
  confidence_threshold=0.0, aggregation_strategy="simple", output_format="dict",
  group_entities=False, …)` — the general **clinical NER** (token-classification) entry point. With
  the default `output_format="dict"` it returns an `AnalyzeResult` **object** (a misnomer only in
  part — it *is* a `Mapping[str, Any]`, so `result["entities"]` works, but `__getitem__` delegates to
  `to_dict()` and hands back plain **dicts**, not `EntityPrediction`s; hence `_entities` reads
  `.entities`. It was `PredictionResult` before 2.0) whose `.entities` is a
  `list[EntityPrediction]`, each with
  `.text`/`.label`/`.confidence`/`.start`/`.end`. Labels are **UPPERCASE** (`DISEASE`, `CHEM`, `GENE`, …), unlike
  `extract_pii`'s lowercase. Clinical NER is **one model per domain** (no universal model), selected
  by registry alias via `model_name`; the app curates one per domain in `engine.NER_MODELS` and
  pins them with `tests/test_validation.py::test_validation_ner_models_resolve_in_openmed`. The
  app's engine forwards `model_name`/`confidence_threshold`/`aggregation_strategy`/`group_entities`/
  `output_format="dict"`/`loader` (no `lang` — `analyze_text` has none). It excludes the alternate
  construction / tuning knobs (`model_id`/`config`/`include_confidence`/`formatter_kwargs`/
  `metadata`/`use_fast_tokenizer`/`sentence_*` — including 2.x's `sentence_backend`) plus 1.7.0's
  `cache_results`/`max_cache_entries` and 2.x's `assert_context` (assertion-status detection);
  `tests/test_engine.py::test_analyze_forwards_every_openmed_param_or_allowlists_it` pins that split
  (it matters more here than for `deidentify`: `analyze_text` declares `**pipeline_kwargs`, so a
  drifted forwarded param would be silently swallowed rather than raising).
- **Zero-shot (GLiNER)** lives in the `openmed.ner` subpackage, **not** the top-level API, and behind
  the optional `gliner` extra: `from openmed.ner import infer, NerRequest, ModelIndex, ModelRecord,
  Entity, is_gliner_available, get_default_labels, available_domains`.
  `infer(NerRequest(model_id=<HF repo id>, text, labels=[...], threshold=0.5), *, index=<ModelIndex>,
  index_path=None, config=None, loader=None)` → `NerResponse(entities=[Entity(text, start, end, label, **score**, group,
  extras)], meta={...})`. Four things the app works around: (1) `Entity` exposes `.score`, **not**
  `.confidence` — the service adapter's `.score` fallback handles it. (2) `infer`'s default index is a
  file (`<site-packages>/models/index.json`) openmed **doesn't ship**, so a caller must pass an
  in-memory `ModelIndex` (the engine fabricates a one-entry one; `ModelRecord.id` must be the **HF repo
  id**, resolved from the registry alias via `get_all_models()[alias].model_id`) — rather than pointing
  `index_path=` at an on-disk index the app has none of. (3) the GLiNER branch
  **ignores `loader=`** and caches models itself (`lru_cache` in `openmed.ner.families.gliner`), and is
  torch-only (no MLX) — `load_gliner_handle` is not publicly re-exported, so `infer` is the only clean
  route. (4) it raises `MissingDependencyError` (an `ImportError` subclass) when `gliner` isn't
  installed, which `service._run` maps to a pass-through `ServiceError` install hint.

## Known gotchas

- **`shift_dates` was fixed in openmed 1.6.0.** Earlier versions shifted only entities labelled
  exactly `"DATE"`, but the default `OpenMed-PII-SuperClinical-Small-44M-v1` model emits lowercase
  `"date"`, so they masked dates instead. openmed 1.6.0 matches dates by canonical label
  (`openmed/core/pii.py:_is_date_entity` normalizes the model's `"date"`), so `shift_dates` now
  shifts dates on the default model. `tests/test_pii_model.py::test_shift_dates_actually_shifts_dates`
  asserts this (it was a `strict` xfail before the upgrade).
- **A `policy` overrides `method`, and `keep_mapping` is ORed with the policy's own flag.** When
  `deidentify(policy=…)` is set, openmed's pipeline assigns each span's action from the profile and
  **ignores the flat `method`** (verified: `method="replace"` + `policy="hipaa_safe_harbor"` still
  masks). And the effective mapping is `explicit_keep_mapping OR profile.keep_mapping` — so passing
  `keep_mapping=True` alongside a *masking* policy (HIPAA Safe Harbor) wrongly makes it **reversible**
  (openmed returns a mask-token→original mapping), contradicting the policy's irreversible posture.
  `service.anonymize_policy` therefore passes `keep_mapping=False` and lets the profile decide: the
  **five** `keep_mapping=True` profiles (GDPR pseudonymization, GDPR Art. 9 health, Canada PIPEDA,
  UK ICO, China PIPL) return a re-identification key; everything else doesn't — including the four
  *surrogate* profiles that keep no key (Australia Privacy Act, India DPDP, ZA POPIA, NG NDPA), so
  "surrogate" does **not** select reversibility. This only surfaces under `--run-model` (a stub can't model openmed's OR), so
  `tests/test_engine.py::test_engine_deidentify_policy_masks_and_pseudonymizes` pins both branches.
- **Re-identification has TWO independent hazards, and each side owns one of them.**
  *(a) openmed mis-restores overlapping plain keys.* It applies `str.replace` per entry, so a key
  that is a prefix/substring of another (e.g. `ALIAS_1` vs `ALIAS_10`, or unbracketed
  `hash`/`replace` surrogates) corrupts the longer one, and a replacement value that contains
  another key gets re-substituted. `PIIEngine.reidentify` restores in a single regex pass (longest
  key first), so no replacement is re-scanned and both those modes are eliminated;
  `tests/test_engine.py` pins the prefix and value-contains-key cases, and the raw-openmed
  limitation stays a `strict` xfail in `tests/test_pii_pure.py`.
  *(b) openmed 2.x added an occurrence-mapping protocol the app must speak.* When one redacted
  surface stands for several distinct originals, `_build_reidentification_mapping` emits
  `__openmed_occurrence_v1__:<8-digit ordinal>:<surface>` keys instead of one plain key. Those are
  **protocol, not literal text** — matching them verbatim finds nothing, so a restorer that doesn't
  parse them silently leaves every affected placeholder in the "re-identified" output. This is
  reachable in two default clicks: `method="aadhaar_mask"` is not in openmed's unique-placeholder
  set (`core/pii.py:2244-2262` suffixes `_2`/`_3` only for `mask`/`remove`, plus `shift_dates` on a
  non-date span and `format_preserve` when it falls back to mask), so every repeated label collapses
  onto one placeholder and the entire mapping comes back occurrence-keyed — and `Keep mapping`
  defaults on. `PIIEngine.reidentify` therefore groups occurrence keys by surface and hands out
  their originals in ordinal order as the single pass walks the document (ordinals are assigned in
  entity order upstream, so ordinal order *is* document order), leaving a surface untouched once its
  group is exhausted. The prefix is **baked** in `engine.py` (`_OCCURRENCE_MAPPING_PREFIX`) rather
  than imported, because `reidentify` is a pure lock-free `@staticmethod` with no openmed import;
  `tests/test_pii_pure.py::test_occurrence_prefix_matches_openmed` pins the copy against openmed's
  private constant so a rename fails CI. The two hazards are disjoint — both must be handled.
- **`locale` widens DETECTION in openmed 2.x, not just surrogate generation — in the UI through
  regional overlays, not the sweep's 28→35.** The `Advanced` → "Surrogate locale" box reads like
  a Faker knob, and through 1.x it was one. In 2.x the locale reaches both pattern stages:
  *(a) the safety sweep.* `core/safety_sweep.py::_patterns_for_language` special-cases *only*
  `lang="en"` **with `locale=None`** to `PII_PATTERNS` + Aadhaar; **any** non-`None` locale escapes
  to `pii_i18n.py::get_patterns_for_language`, whose "language-agnostic" base adds seven
  (**28 → 35**): two passport MRZ (TD3/TD1), China's USCC, and the four
  `INDIA_HEALTH_ID_PII_PATTERNS` — ABHA number, ABHA address, UPI ID, ration card (only the first
  two are health IDs, whatever the name says).
  *(b) smart merging.* `core/pii.py::_apply_pii_smart_merging` *always* calls
  `get_patterns_for_language(lang, locale)`, so those seven are live at detection whenever smart
  merging is on — which it always is in the UI's de-identifying tabs (only `Detect` has a toggle).
  Measured on the default model with it on: an ABHA number, ABHA address, UPI ID, ration card and
  USCC were each detected identically with the locale blank or `en_US`; with it off (the API's
  `use_smart_merging=false`) a ration card was missed blank and caught under `en_US`. What changes
  detection in the UI is a locale with its own `LOCALE_PII_PATTERNS` overlay, which *both* stages
  append: `en_GB` (37 sweep patterns) relabelled an NHS number from `phone_number` to
  `national_id` and caught an NI number that blank missed outright; `en_CA` has 38,
  `en_IN`/`en_ZA` 43, while `en_US`/`pt_BR` have no overlay. So a regional locale changes
  *what gets found*, not only what replaces it, and blank is never the wider setting. All three
  locale inputs' help text says so (Single/Batch, `Anonymize`, `Policy de-ID` — the last also
  forwards the locale under a masking policy). Reproduce with
  `openmed.core.safety_sweep._patterns_for_language("en", None)` vs `(..., "en_US")` / `"en_GB"`.
- **The engine pins eager attention because DeBERTa-v2 has no SDPA kernel — but the pin is now
  belt-and-braces, not load-bearing.** The OpenMed models (default PII + the NER models) are
  `DebertaV2ForTokenClassification`, which has no SDPA kernel. The precise transformers rule
  (`modeling_utils.py::get_correct_attn_implementation`, verified identical in 5.13.1, 5.15.1, 5.16.1
  and 5.17.0) is:
  `_sdpa_can_dispatch` raises `DebertaV2ForTokenClassification does not support ...
  scaled_dot_product_attention` **only when the caller requested SDPA explicitly** — a caller that
  passes nothing still falls back to eager silently. So the earlier framing ("transformers ≥5.13
  hard-errors where ≤5.12 downgraded") was wrong: what mattered was that openmed's
  `torch_attention_backend="auto"` used to request SDPA explicitly. **openmed 2.x no longer does** —
  `openmed/torch/attention.py::select_attn_implementation("auto")` returns `None` (its own comment:
  "Selecting SDPA from Torch capability alone can force an unsupported implementation on models such
  as DeBERTa-v2"), and `models.py::_apply_attention_pipeline_kwargs` only sets
  `attn_implementation` when that result is not `None`. `PIIEngine.loader` still builds every
  `ModelLoader` with `OpenMedConfig(torch_attention_backend="eager")` — **keep it**: eager is the impl
  these models run under either way, and pinning it means an openmed regression here cannot silently
  break every model load. No fast test catches a load failure (they stub the model; the real load path
  is `--run-model` only), so verify model loading end-to-end after any torch/transformers/openmed bump.
  The `OPENMED_TORCH_ATTENTION_BACKEND` env var still overrides the pin.
- **The `gliner` extra forks the transformers version, on purpose — and the marker extra's floor
  must be re-derived on every gliner bump.** `gliner` caps transformers (`<5.17.0` as of 0.2.29;
  `<5.14.0` in 0.2.28, `<5.7.0` in 0.2.27, `<5.2.0` in 0.2.25–0.2.26, and *uncapped* in
  0.2.23–0.2.24), but the rest of the stack targets the latest. uv builds one universal
  lock, so **merely declaring** a bare `gliner` extra would cap transformers for *every* install —
  including CI and the PII/NER tabs. `pyproject.toml` avoids that with a `[tool.uv] conflicts` between
  the `gliner` extra and a marker `hf-latest = ["transformers>=5.17"]` extra, so uv **forks** the lock:
  the default resolution (and `--extra mlx`) stays on the latest transformers, and only
  `--extra gliner` (which combines with `--extra mlx`) downgrades. The `hf-latest` extra has no runtime
  purpose; don't "clean it up" or the fork collapses. **Keep its floor at or above gliner's cap.** uv
  forks on the *declaration* alone and never checks that a declared conflict is real, and this has
  now drifted twice: gliner 0.2.28 raised its cap from `<5.7` to `<5.14`, making the old `>=5.7`
  floor satisfiable alongside it (5.7–5.13), and 0.2.29 raised it to `<5.17`, doing the same to
  `>=5.14` (5.14–5.16) — each time the fork still worked, but the recorded reason had quietly
  become false. The conflict also depends on the **`gliner` extra's own floor** (`>=0.2.25`): the
  old `gliner>=0.2.0` admitted the uncapped 0.2.23–0.2.24, so `gliner==0.2.24` +
  `transformers>=5.17` co-resolved even after the marker floor was raised. Re-read
  `Requires-Dist: transformers` from the gliner wheel on every bump, raise the marker floor to
  match, and keep the `gliner` floor above any uncapped release — verify with
  `uv pip compile` of `gliner>=<floor>` + `transformers>=<marker floor>`, which must fail.
  The zero-shot path needs no eager pin regardless of the fork's transformers version: neither
  `gliner.GLiNER.from_pretrained` (openmed passes only `cache_dir`/`token` —
  `openmed/ner/families/gliner.py::_load_model`) nor the checkpoints themselves request SDPA, and per
  the gotcha above an unrequested SDPA degrades to eager silently. Depend on **bare `gliner`**, not
  `openmed[gliner]` (the latter's `gliner[tokenizers]` drags in mecab/stanza/spacy — 136 packages vs 75
  at openmed 2.5 —
  for tokenizers this app never uses). Verify the fork after any dependency bump:
  `uv export --extra gliner | grep transformers` should show a version below gliner's cap (`5.16.x`
  today), and `uv export` (no extras) the latest (`5.17.x` today).
- **pysbd `SyntaxWarning`s** (a transitive dependency) appear on Python ≥3.12 from its regex
  literals; they are harmless. `openmed_studio/engine.py` silences them with
  `warnings.filterwarnings("ignore", category=SyntaxWarning)` *before* importing `openmed`.
- **torch 2.14 prints `FutureWarning: torch.jit.script is deprecated` on the first DeBERTa-v2
  load.** It comes from transformers' own `models/deberta_v2/modeling_deberta_v2.py` (its
  `@torch.jit.script` helpers), not from this app or openmed; torch 2.14 promoted it from a
  `DeprecationWarning` to a `FutureWarning`, so Streamlit/uvicorn consoles now show it. It is
  harmless today, but it is torch announcing a removal — a future torch that drops `jit.script`
  breaks DeBERTa-v2 loading until transformers moves off it (another reason `--run-model` gates
  every torch bump). Under `-W error::FutureWarning` the model fails to load. On Python 3.14+ the
  text is stronger — "`torch.jit.script` is not supported in Python 3.14+ and may break" — and
  CI covers only 3.10/3.13, while a fresh uv venv here picks 3.14.
- **openmed's error taxonomy (2.3+) multiply-inherits, so `service._run`'s `except` order is
  load-bearing.** `openmed.core.errors.InputError` is a `ValueError` *and* `TypeError`, and
  `ModelLoadError` (raised when `from_pretrained` fails, e.g. an unknown model id) is an
  `ImportError` *and* a `ValueError`. `_run` catches `ValueError` first, so a load failure
  lands in `bad_options` (400) — and in `deidentify_batch`, one `{"ok": False}` row per note
  rather than an abort — as the plain `ValueError` it replaced did in 2.1, while an
  offline/integrity failure (`ModelIntegrityError`/`OfflineModeError`, both `RuntimeError`; e.g.
  an uncached model under `HF_HUB_OFFLINE=1`) is `unavailable` (503) and aborts a batch. Moving `except ImportError` above `except ValueError`
  would silently reclassify every load failure as `dependency`.
- **`st.dataframe(key=…)` is inert unless selection is activated** (re-verified on Streamlit 1.64.0). The
  performance guidance to give a dataframe a stable `key` so it doesn't remount when its data
  changes applies only to a *selectable* one: `streamlit/elements/arrow.py` sets
  `proto.id = compute_and_register_element_id(…, user_key=key, …)` **inside** its
  `if is_selection_activated:` branch, so under the default `on_select="ignore"` the key is accepted,
  never registered, and silently dropped — no error, no warning, and nothing `ruff` or `ty` can see.
  Verified on this app: the rendered proto's `id` stays `''`. Those tables are identified by delta
  path instead, which is already stable across reruns, so `_render_entity_table` deliberately passes
  no key rather than imply a guarantee it doesn't provide. `placeholder` and `row_height`, by
  contrast, *do* apply without selection.
- **All eight tab bodies run on every rerun, and that is the cheaper option here.** `st.tabs`
  defaults to `on_change="ignore"`, so every tab's body executes even while hidden. Measured on this
  app: a full rerun is ~18 ms with nothing submitted and ~24 ms once all four persisted panels hold
  a result, the eight tab bodies being the bulk of it (profiled at ~20 of ~26 ms). They are cheap
  because a tab mostly just *declares* widgets — each returns before its `service.*` call unless its
  own form was submitted, and the only openmed work in a hidden tab is noise
  (`zero_shot_available()` 0.0003 ms, cached in an openmed module global; `default_labels()`
  0.03 ms); only the persisted de-identify panels do more, re-rendering their highlight and entity
  table from `st.session_state`. Switching to `on_change="rerun"` + `if tab.open:` would shave most
  of that off a submit path already dominated by model inference while charging a full server rerun
  for every tab *click*, which costs nothing today — a net loss until some tab body starts doing
  real work at render time. Two neighbouring "optimizations" from the same guidance would be
  outright **bugs**: gating an `Advanced` expander on `.open` stops declaring the widgets its form's
  submit reads (a collapsed expander would silently send options other than the ones on screen), and
  `@st.fragment(parallel=True)` can overlap nothing behind `PIIEngine`'s single inference lock. The
  numbers are recorded at the `st.tabs` call in `streamlit_app.py`.
- **FastAPI ≥0.139 includes routers lazily (still true at 0.141.1).** `app.include_router(...)` no longer eagerly flattens the
  child routes into `app.routes`; it stores an `_IncludedRouter` wrapper (no `.path`). So the `/compat`
  routes are absent from `app.routes` even when mounted — if you need to introspect routes, use
  `app.openapi()["paths"]` instead. Requests route correctly regardless; only `.path`-based
  introspection is affected. (`test_api.py` sidesteps this entirely — it verifies the `/compat` mount
  *behaviorally*, e.g. a 404 when unmounted, rather than by listing routes.)
- **The FastAPI `TestClient` warns about `httpx` vs `httpx2`.** `starlette.testclient` emits a
  `StarletteDeprecationWarning` ("install `httpx2` instead") on import; at runtime it is harmless
  and `httpx` still works (starlette 1.7 still falls back to it). One cost since starlette 1.7: it
  dropped `TestClient`'s typed `get`/`post` overrides, and because its (unchanged) `TYPE_CHECKING`
  branch imports `httpx2`, without `httpx2` installed ty sees those methods as `Unknown` and
  `test_api.py`'s client calls go un-type-checked (ty still passes). Swapping deps is therefore a type-coverage
  decision, not just a noise fix — don't do it silently.
- The `.venv` here is ~1.2 GB (Torch + Transformers) and is gitignored.
