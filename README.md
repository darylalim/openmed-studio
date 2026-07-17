# OpenMed Studio

A clinical-NLP app built on [OpenMed](https://openmed.life/docs/).

It surfaces OpenMed's toolkit through two interfaces over one shared in-process core: a
[Streamlit](https://streamlit.io/) UI and a [FastAPI](https://fastapi.tiangolo.com/) HTTP API. Today it
does PII/PHI de-identification (including surrogate anonymization and **policy-driven anonymization**
under regulatory compliance profiles like HIPAA Safe Harbor and GDPR), clinical NER, and zero-shot
(GLiNER) extraction; deeper policy tooling (custom policies, cross-document consistency) is on the
roadmap.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10.

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
| **Policy de-ID** | Anonymize under a **regulatory policy** (HIPAA Safe Harbor, GDPR pseudonymization, PIPEDA, UK ICO, …) — the policy decides, per entity type, whether to mask, redact, or surrogate. Masking policies are irreversible; surrogate policies keep a re-identification key. |
| **Re-identify** | Restore originals from a kept mapping (auto-filled from the last Single note, Anonymize, or Policy de-ID run). |

Detect, Clinical NER, Zero-shot, Single note, and Policy de-ID render matched entities as highlighted
text with a color legend, plus an entity table. A few more things worth knowing:

- Clinical NER and Zero-shot each pick a domain (Disease, Pharmaceutical, Chemical, Anatomy,
  Genomics, Protein, Oncology, Species, Pathology, Hematology — plus a broad Medical model for NER).
  A live preview shows the model's name, size, and what it detects, and the confidence slider seeds
  from that model's recommended threshold.
- Zero-shot lets you edit the suggested labels or type your own (e.g. "chemotherapy regimen",
  "biopsy site"); all labels are extracted together in one pass. It needs the optional `gliner`
  backend — see [Zero-shot (GLiNER)](#zero-shot-gliner).
- Anonymize leaves anything the model misses in place, so review the output before sharing.
- Policy de-ID picks a compliance profile instead of a method: the policy decides each entity type's
  action, so the same note anonymizes differently under each. A live preview shows the policy's default
  action, whether it is reversible, and whether it enforces the safety sweep. Masking policies (HIPAA
  Safe Harbor) are irreversible; surrogate policies (GDPR pseudonymization, PIPEDA, UK ICO) keep a
  re-identification key that round-trips through Re-identify.

### Controls

- The sidebar reports the engine's model / backend / load state and holds the one global filter: the
  detection language (12 supported), which applies to Detect, Single note, Batch, Anonymize, and
  Policy de-ID.
- **Single note** and **Batch** each expose the de-identification method (`mask` / `remove` /
  `replace` / `hash` / `shift_dates` / `format_preserve`), a confidence slider, `keep_mapping`, and
  an Advanced expander whose knobs follow the chosen method:
  - `replace` / `format_preserve` (surrogates; `format_preserve` keeps each identifier's shape, so a
    phone stays phone-shaped) — a determinism toggle, `seed`, and surrogate `locale`.
  - `shift_dates` — `date_shift_days` and `keep_year`.
  - the safety sweep (any method).
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
| `POST /pii/anonymize-policy` | Anonymize under a regulatory `policy` (the policy picks each action). |
| `POST /pii/reidentify` | Restore originals from a kept `mapping`. |
| `GET /health` | Liveness + configured model/backend/limits (always unauthenticated). |

Every non-2xx response uses one envelope: `{"error": {"code", "message", "details"}}`. Validation
errors are PHI-safe — the offending request text is never echoed back.

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
| `OPENMED_STUDIO_COMPAT` | Truthy = mount an opt-in `/compat/pii/{extract,deidentify}` surface matching OpenMed's own REST shape (echoes the original text — off by default). |
| `OPENMED_STUDIO_BACKEND` / `OPENMED_STUDIO_MAX_TEXT_LENGTH` | Same as for the UI — backend pin and per-request text cap. |

> **Run it locally.** Like the UI, the API is a single-user / small-scale tool. An unset API key means
> **no auth** — put it behind your own auth, TLS, or reverse proxy before exposing it or processing real
> PHI. Concurrent requests are serialized on the shared model (one inference at a time).

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
  enums, and the confidence range. On a rejection the service seam builds the error from only the
  field's location and message — never Pydantic's echoed input — so the offending text (PHI) isn't
  shown.
- **Backend.** Inference is auto-detected: MLX on Apple Silicon when the `mlx` extra is installed,
  else Hugging Face / PyTorch (CPU, CUDA, Apple MPS). Pin it with `OPENMED_STUDIO_BACKEND=hf|mlx` —
  an explicit `mlx` pin *raises* on a non-MLX host rather than falling back. See
  [Apple Silicon (MLX)](#apple-silicon-mlx).
- **Model reuse.** Streamlit caches the engine (`st.cache_resource`), so the PII model loads at most
  once per process and is reused across every tab. The shared loader dispatches by model name, so the
  Clinical NER tab loads a per-domain model into the same loader on first use of that domain.
- **Theme-aware.** The highlights and legend adapt to light or dark mode automatically (a translucent
  tint plus `color: inherit`, no runtime theme detection; both modes live in `.streamlit/config.toml`).
- **Isolated reruns.** The Detect / Clinical NER / Zero-shot / Batch / Re-identify tabs are
  `st.fragment`s, so an interaction in one doesn't rerun the others; Single note and Anonymize stay
  full reruns so they can hand their result to Re-identify.

## Optional extras

Both extras are opt-in via `uv sync --extra …` and combine with each other.

### Apple Silicon (MLX)

On M-series Macs, swap the portable Torch/Transformers backend for Apple's native
[MLX](https://github.com/ml-explore/mlx) backend:

```bash
uv sync --extra mlx
```

The default model isn't pre-packaged for MLX, so OpenMed converts it on the fly on first run and
caches the result under `~/.cache/openmed/mlx/`. A pre-converted `-mlx` repo (e.g.
`OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx`) is an optional shortcut that skips conversion —
pass it as a local directory via `model_name=…`. See the
[MLX backend docs](https://openmed.life/docs/mlx-backend/).

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
detection,
masking, deterministic replacement, and round-trips; the zero-shot model test is additionally gated on
the `gliner` extra, so CI never downloads it.

CI (`.github/workflows/ci.yml`) runs the lint / format / type / test checks on pushes to `main` and
on every pull request, across Python 3.10 and 3.13.

## Security & notes

**Run it locally.** This is a single-user / small-scale tool. The [HTTP API](#http-api-fastapi) *does*
open a network endpoint, so put it behind your own auth, TLS, or a reverse proxy before exposing it, and
don't run it on a network with real PHI as-is. The guards that protect the model are enforced in-process
by the service seam — so **both** the UI and the API inherit them:

- The text / batch / mapping caps, the value / enum / format checks, backend pinning, and not echoing
  request input on a validation error.
- Concurrent API requests are serialized on the shared model (one inference at a time).

The API layer adds the HTTP-only protections back on top: **`X-API-Key` auth** (via
`OPENMED_STUDIO_API_KEY` — unset means the service runs **unauthenticated**, with a startup warning), a
uniform `{"error": {…}}` envelope, PHI-safe 422s, and the opt-in OpenMed-REST `/compat` surface
(`OPENMED_STUDIO_COMPAT`, which echoes the original text — off by default).

Other things to keep in mind:

- Treat any returned `mapping` as re-identification material — it is as sensitive as the raw PHI.
- All identifiers in the app's sample note are fabricated.
- Smart entity merging is on by default (`use_smart_merging=True`), recombining token-fragmented PII
  like dates and SSNs into whole spans.
- De-identification runs a deterministic structured-identifier safety sweep after detection
  (`use_safety_sweep=True`, toggleable per tab), so it may redact a few identifiers the Detect tab
  (which doesn't run the sweep) doesn't surface.
- More guides: [OpenMed docs](https://openmed.life/docs/) ·
  [PII anonymization](https://openmed.life/docs/anonymization/) ·
  [smart merging](https://openmed.life/docs/pii-smart-merging/).
