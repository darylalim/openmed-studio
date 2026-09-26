# OpenMed Studio

A clinical-NLP app built on [OpenMed](https://openmed.life/docs/).

It surfaces OpenMed's toolkit through two interfaces over one shared in-process core: a
[Streamlit](https://streamlit.io/) UI and a [FastAPI](https://fastapi.tiangolo.com/) HTTP API. Today it
does PII/PHI de-identification (including surrogate anonymization and **policy-driven anonymization**
under regulatory compliance profiles like HIPAA Safe Harbor and GDPR), clinical NER, and zero-shot
(GLiNER) extraction; deeper policy tooling (custom policies, cross-document consistency) is on the
roadmap.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10–3.14 (PyTorch has no 3.15 wheels yet).

```bash
uv run streamlit run streamlit_app.py
```

`uv` reads [`pyproject.toml`](pyproject.toml), creates a `.venv`, installs the dependencies, and
opens the app at `http://localhost:8501`. The first de-identification downloads a small
(~44M-parameter) clinical PII model — `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` — from the
Hugging Face Hub and caches it under `~/.cache/openmed`, so later runs are fast and offline.

## What it does

The app opens with eight tabs:

| Tab | What it does |
| --- | --- |
| **Detect** | Find and highlight PII/PHI without redacting — audit what the model sees before choosing a method. |
| **Clinical NER** | Extract clinical entities (diseases, drugs, anatomy, genes, …) with a curated token-classification model per domain. |
| **Zero-shot** | Extract *any* entity types you name, with no fine-tuned model per label, via OpenMed's [GLiNER](https://github.com/urchade/GLiNER) models. |
| **Single note** | De-identify one note — original (PII highlighted) beside the redacted text, with a download and a re-identification key. |
| **Batch** | De-identify up to 100 notes at once — a results table with per-note entity counts; a failing note is isolated as a `Failed` row instead of aborting the batch. |
| **Anonymize** | Replace *detected* PII/PHI with realistic *fake* surrogates rather than masks; round-trips through Re-identify. |
| **Policy de-ID** | Anonymize under one of 10 **regulatory policies** (HIPAA Safe Harbor, GDPR Art. 9 health data, China PIPL, South Africa POPIA, Nigeria NDPA, Kenya DPA, …) — the policy decides, per entity type, whether to mask or surrogate. Masking policies are irreversible; only the policies that keep a mapping (GDPR Art. 9, China PIPL) yield a re-identification key. |
| **Re-identify** | Restore originals from a kept mapping (auto-filled from the last Single note, Anonymize, or Policy de-ID run). Each surrogate is swapped back wherever its text appears, so if an age became `2`, other `2`s in the note (a dose, a blood pressure) change too — check the result. |

Detect, Clinical NER, Zero-shot, Single note, and Policy de-ID render matched entities as highlighted
text with a color legend, plus an entity table. A few more things worth knowing:

- Clinical NER and Zero-shot each pick a domain from the same ten (Disease, Pharmaceutical,
  Chemical, Anatomy, Genomics, Protein, Oncology, Species, Pathology, Hematology).
  A live preview shows the model's name, size, and what it detects, and the confidence slider seeds
  from that model's recommended threshold.
- Zero-shot lets you edit the suggested labels or type your own (e.g. "chemotherapy regimen",
  "biopsy site"); all labels are extracted together in one pass. It needs the optional `gliner`
  backend — see [Zero-shot (GLiNER)](#zero-shot-gliner).
- Anonymize leaves anything the model misses in place, so review the output before sharing.
- Policy de-ID picks a compliance profile instead of a method: the policy decides each entity type's
  action, so the same note anonymizes differently under each. A live preview shows the policy's
  canonical name, whether it is reversible, whether it enforces the safety sweep, and a short
  description of what it masks or substitutes. Reversibility is the profile's own call, not the
  action's: masking policies (HIPAA Safe Harbor) never keep a key, and only some surrogate policies
  do (GDPR Art. 9 Health, China PIPL) — the others, South Africa POPIA and Nigeria NDPA, replace
  identifiers *irreversibly*. When a key is kept it round-trips through Re-identify; the preview
  says which case you are in.
- OpenMed ships 20 compliance profiles, and the app offers only the 10 that leave nothing they
  detect in place. The other 10 (among them GDPR pseudonymization, Canada PIPEDA, UK ICO and the
  research and clinical profiles) *keep* whatever falls under OpenMed's catch-all `OTHER` label,
  and OpenMed files 26 of the PII model's 54 entity types there: license, tax and employee IDs,
  fax and health-plan numbers, employers, religion, political views, sexuality. Bar the few formats
  the safety sweep catches, those identifiers would pass through verbatim and stay out of the
  entity table, so the profiles stay hidden (the API rejects them with a 422) until OpenMed stops
  keeping them.

### Controls

- The sidebar reports the engine's model / backend / load state and holds the one global filter: the
  detection language (12 supported), which applies to Detect, Single note, Batch, Anonymize, and
  Policy de-ID.
- **Single note** and **Batch** each expose the de-identification method (`mask` / `remove` /
  `replace` / `hash` / `shift_dates` / `format_preserve` / `aadhaar_mask`), a confidence slider,
  `keep_mapping`, and an Advanced expander whose knobs follow the chosen method:
  - `replace` / `format_preserve` (surrogates; `format_preserve` keeps each identifier's shape, so a
    phone stays phone-shaped) — a determinism toggle, `seed`, and surrogate `locale`. Some locales
    (e.g. `en_GB`, `en_IN`) also add that region's identifier patterns to detection, so the locale
    can change what gets found, not just what replaces it.
  - `shift_dates` — `date_shift_days` and `keep_year`.
  - the safety sweep (any method).
- `aadhaar_mask` is India-specific: a number that passes the Aadhaar checksum becomes
  `XXXX XXXX NNNN` (the UIDAI masked form, which **keeps the last four digits**), and every other
  entity gets the ordinary mask placeholder. Because it leaves part of an identifier in the
  output, the method picker warns when you choose it.
- The confidence slider defaults to `0.5` for higher PHI recall (the `deidentify` default is `0.7`).
  The model loads on the first request, so that call shows a spinner and is slower than the rest.

## HTTP API (FastAPI)

The same capabilities are available over HTTP. The API is a thin layer over the *same* in-process core
the UI uses (see [How it works](#how-it-works)) — it is not a separate service the UI talks to; the two
run independently. `fastapi` and `uvicorn` are core dependencies, so nothing extra to install:

```bash
uv run python -m openmed_studio                       # serve on http://127.0.0.1:8080
# or, to pass uvicorn flags directly (e.g. --reload):
uv run uvicorn openmed_studio.main:app --port 8080
```

Open `http://127.0.0.1:8080/docs` for interactive OpenAPI docs. Endpoints:

| Method & path | Does |
| --- | --- |
| `POST /pii/extract` | Detect PII/PHI entities (no redaction). |
| `POST /ner` | Clinical NER — pass a `model_name` (a `NER_MODELS` alias). |
| `POST /zero-shot` | Zero-shot extraction — pass `model_name` + `labels` (needs the `gliner` extra). |
| `POST /pii/deidentify` | De-identify one note (`method`, `keep_mapping`, …). |
| `POST /pii/deidentify/batch` | De-identify up to 100 notes; a bad note is isolated as `{"ok": false, …}`. |
| `POST /pii/anonymize-policy` | Anonymize under a regulatory `policy` (the policy picks each action); the profiles the UI hides return 422. |
| `POST /pii/reidentify` | Restore originals from a kept `mapping`. |
| `GET /health` | Liveness + configured model/backend/limits, and `working_directory_clean` (see [Security & notes](#security--notes)); always unauthenticated. |

Every non-2xx response uses one envelope: `{"error": {"code", "message", "details"}}`. Validation
errors are PHI-safe — the offending request text is never echoed back.

`model_name` is allowlisted per capability. The PII routes (and `/compat`) accept only the default
model — omit `model_name` to get it, or OpenMed's own default for a non-English `lang` — or its
pre-converted MLX build (see [Apple Silicon (MLX)](#apple-silicon-mlx)); `/ner` and `/zero-shot`
accept only the ten curated aliases of their domain pickers (`NER_MODELS` / `ZERO_SHOT_MODELS` in
[`engine.py`](openmed_studio/engine.py)). Anything else — another Hugging Face repo, another
OpenMed registry alias, a different casing — is a 422 that doesn't echo the name. The OpenAPI
schema (`/docs`, `/openapi.json`) lists each route's accepted names in the `model_name` field's
description and examples. An operator can widen every allowlist with `OPENMED_STUDIO_EXTRA_MODELS`
(below); those additions aren't listed.

### Auth & configuration

Authentication is off by default for local use (the service logs a startup warning). Set
`OPENMED_STUDIO_API_KEY` to require an `X-API-Key` header on every model route (`/health` stays open):

```bash
OPENMED_STUDIO_API_KEY=secret uv run python -m openmed_studio
curl -H "X-API-Key: secret" -H "Content-Type: application/json" \
  -d '{"text": "Patient John Doe, MRN 1234567."}' http://127.0.0.1:8080/pii/deidentify
```

| Env var | Effect |
| --- | --- |
| `OPENMED_STUDIO_API_KEY` | Require this key via `X-API-Key`. Unset = unauthenticated (local) + a warning. |
| `OPENMED_STUDIO_HOST` / `OPENMED_STUDIO_PORT` | Bind address for `python -m openmed_studio` (default `127.0.0.1:8080`). |
| `OPENMED_STUDIO_PRELOAD` | Truthy = warm the model at startup (in a worker thread) so the first request isn't slow. |
| `OPENMED_STUDIO_COMPAT` | Truthy = mount an opt-in `/compat/pii/{extract,deidentify}` surface matching OpenMed's own REST shape (echoes the original text — off by default). It ignores unknown fields like OpenMed's does, but `lang` and `model_name` follow the primary routes' rules (the 12 supported languages, the PII allowlist). |
| `OPENMED_STUDIO_BACKEND` / `OPENMED_STUDIO_MAX_TEXT_LENGTH` | Same as for the UI — backend pin and per-request text cap. |
| `OPENMED_STUDIO_EXTRA_MODELS` | Comma-separated model ids to accept as `model_name` on every route, in addition to the curated ones (exact match, case included; read at startup, and a malformed entry stops the app). You own what these load: OpenMed downloads an unregistered repo without an integrity check, and a privacy-filter name (`openai/privacy-filter…`, `OpenMed/privacy-filter-…`) goes to a loader that runs repo code (`trust_remote_code=True`) for three first-party repos — `openai/privacy-filter`, `OpenMed/privacy-filter-multilingual` and `OpenMed/privacy-filter-nemotron` — and, when OpenMed's MLX backend is unavailable, swaps an `-mlx` name for one of them. |

> **Run it locally.** Like the UI, the API is a single-user / small-scale tool. An unset API key means
> **no auth** — put it behind your own auth, TLS, or reverse proxy before exposing it or processing real
> PHI, and start it from a clean directory (the checklist is under [Security & notes](#security--notes)).
> Concurrent requests are serialized on the shared model (one inference at a time).

## How it works

The model runs in-process — even the [HTTP API](#http-api-fastapi) loads it in-process rather than
calling out to a separate service. Both surfaces —
[`streamlit_app.py`](streamlit_app.py) and [`openmed_studio/main.py`](openmed_studio/main.py) (FastAPI) —
call a reusable, framework-free [`PIIEngine`](openmed_studio/engine.py) (one shared `ModelLoader`)
through the in-process seam in [`openmed_studio/service.py`](openmed_studio/service.py), which validates
each request and adapts OpenMed's results. Because both go through the one seam, they enforce the same
guards; the API adds only HTTP concerns (routing, auth, status codes) on top.

- **Validation.** The Pydantic models in [`openmed_studio/validation.py`](openmed_studio/validation.py)
  gate every request before it reaches the model: the per-request text cap (50k chars, override with
  `OPENMED_STUDIO_MAX_TEXT_LENGTH`), the batch (≤100) and mapping (≤5,000) bounds, the language/method
  enums, the confidence range, and a per-capability allowlist on every `model_name` (widen it with
  `OPENMED_STUDIO_EXTRA_MODELS`). On a rejection the service seam builds the error from only the
  field's location and message — never Pydantic's echoed input — so the offending text (PHI) isn't
  shown.
- **Backend.** Inference is auto-detected: MLX on Apple Silicon when the `mlx` extra is installed,
  else Hugging Face / PyTorch (CPU, CUDA, Apple MPS). Pin it with `OPENMED_STUDIO_BACKEND=hf|mlx` —
  an explicit `mlx` pin *raises* on a non-MLX host rather than falling back. See
  [Apple Silicon (MLX)](#apple-silicon-mlx).
- **Model reuse.** Streamlit caches the engine (`st.cache_resource`), so the PII model loads at most
  once per process and is reused across every tab. The shared loader dispatches by model name, so the
  Clinical NER tab loads a per-domain model into the same loader on first use of that domain.
- **Nord theme, dark only.** `.streamlit/config.toml` carries a single `[theme]` section, which is
  what locks the app to one mode — Streamlit shows the light/dark selector only when both
  `[theme.light]` and `[theme.dark]` exist. The entity highlights draw from the same nine Nord
  accents (translucent tint plus `color: inherit`, so they still need no runtime theme detection),
  and no webfont is loaded: the upstream Nord template's Google Fonts would mean an outbound CDN
  call on every page load, which a clinical-text tool shouldn't make.
- **Isolated reruns.** The Detect / Clinical NER / Zero-shot / Batch / Re-identify tabs are
  `st.fragment`s, so an interaction in one doesn't rerun the others; Single note, Anonymize, and
  Policy de-ID stay full reruns so they can hand their result to Re-identify.

## Optional extras

Both extras are opt-in via `uv sync --extra …` and combine with each other.

### Apple Silicon (MLX)

On M-series Macs, add Apple's native [MLX](https://github.com/ml-explore/mlx) backend, which
OpenMed then prefers over the portable Torch/Transformers one:

```bash
uv sync --extra mlx
```

OpenMed doesn't map the default model to a pre-converted MLX build, so it converts it on the fly on
first run and caches the result beside the downloaded models, in
`~/.cache/openmed/OpenMed_OpenMed-PII-SuperClinical-Small-44M-v1/` (each Clinical NER model gets a
sibling directory the same way). To skip the conversion, pass the default model's own
pre-converted build, `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx`, as `model_name` in an
[HTTP API](#http-api-fastapi) request; OpenMed downloads it as-is. It is the only `-mlx` repo the
API accepts unless an operator adds others with `OPENMED_STUDIO_EXTRA_MODELS`. The UI has no
PII-model picker. See the [MLX backend docs](https://openmed.life/docs/mlx-backend/).

### Zero-shot (GLiNER)

The Zero-shot tab needs OpenMed's optional GLiNER backend:

```bash
uv sync --extra gliner
```

GLiNER pins an older `transformers` than the rest of the stack, so this extra is kept separate from
the default install: `pyproject.toml` declares a conflict between `gliner` and a marker `hf-latest`
extra, which makes uv fork the lock. The default install (and CI, and the PII / Clinical NER tabs)
stay on the latest `transformers`; only `uv sync --extra gliner` resolves to the older one. Until the
extra is installed, the tab shows install instructions rather than the form, and the other tabs are
unaffected.

## Development

Lint, format, and type-check with the project-pinned tools:

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run ty check              # type-check
```

Run the tests with pytest:

```bash
uv run pytest                # fast tests only (model-loading tests are skipped)
uv run pytest --run-model    # also run tests that load the OpenMed PII model
```

Tests live in [`tests/`](tests/). The fast tests need no model: they stub the engine and cover the
in-process service seam, the input guarantees in `validation.py` (caps, enums, format checks, and the
openmed-registry sync guards), `PIIEngine`'s loading contract, the Streamlit UI (via
`streamlit.testing.v1.AppTest`), and the FastAPI service (via `fastapi.testclient.TestClient` — auth,
the error envelope, status mapping, and `/compat`). The `--run-model` tests load real models to verify
detection, masking, deterministic replacement, and round-trips; the zero-shot model test is
additionally gated on the `gliner` extra, so CI never downloads it. The suite ignores the
`OPENMED_STUDIO_EXTRA_MODELS`, `_MAX_TEXT_LENGTH`, `_API_KEY`, `_COMPAT` and `_PRELOAD` settings
in your shell (`tests/conftest.py` clears them), so exporting them to run the app doesn't skew it.

CI (`.github/workflows/ci.yml`) runs on pushes to `main` and on every pull request: the tests across
Python 3.10, 3.13 and 3.14, and the lint / format / type checks once on the 3.10 leg (they are
interpreter-independent, so another run would only duplicate the work).

Releases (`.github/workflows/release.yml`) follow `version` in `pyproject.toml`: bump it, merge to
`main`, and — once CI re-runs green — the workflow tags `v<version>` and publishes a GitHub release.
The notes are GitHub's own, generated from merged pull requests, falling back to the commit subjects
in the range when there are none. Nothing is published to PyPI; this is an application, not a
library.

## Security & notes

**Run it locally.** This is a single-user / small-scale tool. Both surfaces open a network port —
the [HTTP API](#http-api-fastapi) on `127.0.0.1` by default, the Streamlit UI on every interface
unless you bind it (item 2 below) — so put them behind your own auth, TLS, or a reverse proxy
before exposing them, and don't run them on a network with real PHI as-is. The guards that protect
the model are enforced in-process by the service seam — so **both** the UI and the API inherit them:

- The text / batch / mapping caps, the value / enum / format checks, the per-capability
  `model_name` allowlists, backend pinning, and not echoing request input on a validation error.
- A local-path guard: OpenMed resolves a model name against the filesystem *before* the Hugging
  Face Hub, relative to the working directory, so a directory named like a model would be loaded
  in its place. Before every model call the engine therefore refuses — reporting the model backend
  as unavailable (503 over HTTP) and logging the path — when a name it would load, or an `OpenMed`
  or `openai` entry, exists in the directory the app was started from. Start it from a directory
  that has neither (the repo root is fine; on macOS and Windows an `openmed` folder counts too).
  The API checks at startup and the UI when its first session loads; both log a warning naming any
  such entry, and the API's
  `/health` reports `working_directory_clean` (a yes/no, never the path).
- Concurrent API requests are serialized on the shared model (one inference at a time).

The API layer adds the HTTP-only protections back on top: **`X-API-Key` auth** (via
`OPENMED_STUDIO_API_KEY` — unset means the service runs **unauthenticated**, with a startup warning), a
uniform `{"error": {…}}` envelope, PHI-safe 422s, and the opt-in OpenMed-REST `/compat` surface
(`OPENMED_STUDIO_COMPAT`, which echoes the original text — off by default).

**Before you expose the API or the UI, or process real PHI:**

1. Set `OPENMED_STUDIO_API_KEY`. Without it every model route is open to anyone who can reach the
   port. That includes making the server download and keep in memory every model the app
   accepts — about 4.7B parameters (roughly 19 GB at 32-bit precision) of PII and NER models, by
   cycling `lang` and the `/ner` domains, plus the zero-shot models if the `gliner` extra is
   installed. The allowlists cap that; for the API, only the key stops it. **The key protects
   the API only — the Streamlit UI has no authentication of its own**, and its language and domain
   pickers reach the same model loads, so keep the UI on `127.0.0.1` (item 2) or behind a reverse
   proxy that authenticates.
2. Keep the API's default `127.0.0.1` bind, or put TLS or a reverse proxy in front of it. **The
   Streamlit UI has no such default**: Streamlit listens on all interfaces unless told otherwise,
   and the shipped `.streamlit/config.toml` doesn't tell it. Start the UI with
   `uv run streamlit run streamlit_app.py --server.address 127.0.0.1`, or add
   `address = "127.0.0.1"` under a `[server]` section of your `.streamlit/config.toml`.
3. Start the app from a clean directory that nobody else can write to. The local-path guard checks
   the working directory before each model call, but it can't close the gap between its check and
   OpenMed's own, and a directory named like a model can make OpenMed run code from it. If the
   guard ever fires (a `LocalModelPathError` in the log, or `working_directory_clean: false`),
   remove the entry (or start from a directory without it) **and restart** the app: OpenMed caches
   what it loads, so a model it already
   picked up from that directory keeps being served after the directory is gone.
4. Optionally, once the models you serve are downloaded, set `HF_HUB_OFFLINE=1` (and OpenMed's
   `OPENMED_OFFLINE=1`) so no request can trigger a download.

Anything you add to `OPENMED_STUDIO_EXTRA_MODELS` is your call: OpenMed downloads an unregistered
repo without an integrity check, and it runs the code in three first-party privacy-filter repos
(`openai/privacy-filter`, `OpenMed/privacy-filter-multilingual`, `OpenMed/privacy-filter-nemotron`),
and swaps an `-mlx` privacy-filter name for one of them when the MLX backend is unavailable.
Don't set OpenMed's own `OPENMED_TRUSTED_REMOTE_CODE_MODELS` for this app.

Other things to keep in mind:

- Treat any returned `mapping` as re-identification material — it is as sensitive as the raw PHI.
- All identifiers in the app's sample note are fabricated.
- Smart entity merging is on by default (`use_smart_merging=True`), recombining token-fragmented PII
  like dates and SSNs into whole spans.
- De-identification runs a deterministic structured-identifier safety sweep after detection
  (`use_safety_sweep=True`; Single note and Batch can switch it off, Anonymize always runs it, and
  every offered Policy de-ID profile forces it), so it may redact a few identifiers the Detect tab
  (which doesn't run the sweep) doesn't surface.
- More guides: [OpenMed docs](https://openmed.life/docs/) ·
  [PII anonymization](https://openmed.life/docs/anonymization/) ·
  [smart merging](https://openmed.life/docs/pii-smart-merging/).
