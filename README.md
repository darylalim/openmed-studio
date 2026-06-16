# openmed-studio

**A clinical-NLP application built on [OpenMed](https://openmed.life/docs/).**

The goal is an app over OpenMed's full toolkit — clinical NER, PII/PHI de-identification,
anonymization, and zero-shot extraction. **Today it implements PII/PHI de-identification**
(the rest is on the roadmap), shipped as a [FastAPI](https://fastapi.tiangolo.com/) service
plus a library-level demo of the same OpenMed calls.

## The de-identification API

A small FastAPI service in [`openmed_studio/`](openmed_studio/) wraps OpenMed's PII functions
behind HTTP. A reusable, framework-free [`PIIEngine`](openmed_studio/engine.py) holds one
shared `ModelLoader`; the endpoints are thin adapters over it.

```bash
uv run uvicorn openmed_studio.main:app --port 8080   # or: uv run python -m openmed_studio
```

Interactive docs are then at `http://127.0.0.1:8080/docs`. Endpoints:

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness; reports version, configured model, backend, text cap, lazy-load state, and whether auth is on |
| `POST /pii/extract` | Detect PII entities (label, text, char offsets, confidence) |
| `POST /pii/deidentify` | Redact via `mask` / `remove` / `replace` / `hash` / `shift_dates`; returns the surrogate→original `mapping` when `keep_mapping=true` |
| `POST /pii/deidentify/batch` | De-identify up to 100 texts in one call (`{"items": [...]}`) |
| `POST /pii/reidentify` | Restore originals from a kept mapping |

Request options (see `/docs` for the full schema): `lang` (9 supported languages),
`model_name` (override the default model), `confidence_threshold`, `consistent`/`seed`
(deterministic `replace`), and `date_shift_days`/`keep_year` (for `shift_dates`). `text` is
capped at 50k characters (override with `OPENMED_STUDIO_MAX_TEXT_LENGTH`).

Every non-2xx response uses a uniform envelope — `{"error": {"code", "message", "details"}}` —
so `validation_error` / `bad_request` / `service_unavailable` all parse the same way; for a
`validation_error` the `details` list the offending fields, with request content stripped so PHI
isn't echoed back.

```bash
curl -s localhost:8080/pii/deidentify -H 'content-type: application/json' \
  -d '{"text": "Patient John Doe, SSN 123-45-6789.", "method": "mask"}'
# → {"deidentified_text": "Patient [first_name] [last_name], SSN [ssn].", ...}
```

The model loads lazily on the first `/pii/*` request and is then reused across all
subsequent requests (one shared `ModelLoader`). Set `OPENMED_STUDIO_PRELOAD=1` to warm it at
startup instead, so the first request doesn't pay the load cost.

The inference backend is auto-detected (MLX on Apple Silicon when the `mlx` extra is
installed, else Hugging Face/PyTorch); pin it explicitly with `OPENMED_STUDIO_BACKEND=hf|mlx`,
which `/health` echoes back.

### Authentication & PHI safety

The `/pii/*` endpoints process Protected Health Information. By default (no
`OPENMED_STUDIO_API_KEY` set) the service runs **unauthenticated** for local use and logs a
startup warning. Before exposing it on a network or sending it real PHI:

- **Set `OPENMED_STUDIO_API_KEY`** — every `/pii/*` request must then send a matching
  `X-API-Key` header (else `401`); `/health` stays open.
- **Terminate TLS** in front of it (e.g. a reverse proxy) — the API itself speaks plain HTTP.
- **Treat any returned `mapping` as re-identification material** — it is as sensitive as the
  raw PHI; store it encrypted and access-controlled, never beside the redacted text.

```bash
export OPENMED_STUDIO_API_KEY=$(openssl rand -hex 16)
curl -s localhost:8080/pii/extract -H "X-API-Key: $OPENMED_STUDIO_API_KEY" \
  -H 'content-type: application/json' -d '{"text": "Patient John Doe."}'
```

## The de-identification UI

[`streamlit_app.py`](streamlit_app.py) is an optional [Streamlit](https://streamlit.io/)
front-end for the API. It is a **thin HTTP client** — it holds no model and calls the
`/pii/*` endpoints over the network, so the service still enforces auth, validation, and the
text cap. Install the `ui` extra and run it alongside the service:

```bash
uv sync --extra ui
uv run uvicorn openmed_studio.main:app --port 8080   # terminal 1: the API
uv run streamlit run streamlit_app.py                # terminal 2: the UI
```

It opens at `http://localhost:8501` with four tabs:

- **Detect** — `/pii/extract` only: highlight detected PII (with a color legend) and an entity
  table, without redacting — for auditing what the model finds before choosing a method.
- **Single note** — de-identify one note; shows the original with detected PII highlighted
  side-by-side with the redacted text, plus an entity table and (optionally) the mapping.
- **Batch** — edit a table of notes (up to 100) and de-identify them in one `/pii/deidentify/batch` call.
- **Re-identify** — restore originals from a kept mapping (auto-filled from the last single-note run).

The sidebar holds the service URL + API key (sent as `X-API-Key`), a live `/health` status, and
the shared de-identification options (method, language, confidence, `keep_mapping`, and the
deterministic-`replace` / `shift_dates` controls). The confidence slider defaults to **0.5** for
higher PHI recall — note the de-identify API's own default is `0.7`. The model loads on the first
request, so the first call shows a spinner and is slower than the rest.

## The de-identification demo

[`examples/deidentify_pii.py`](examples/deidentify_pii.py) is an end-to-end demo of
OpenMed's de-identification on a synthetic clinical note. It walks through:

1. **`extract_pii`** — detect PII spans (label, text, confidence, character offsets)
2. **`method="mask"`** — replace entities with `[LABEL]` placeholders
3. **`method="remove"`** — delete PII spans entirely
4. **`method="replace"`** — realistic, format-preserving [Faker](https://faker.readthedocs.io/) surrogates, made deterministic with `consistent=True, seed=...`
5. **`method="hash"`** — stable typed digests for linking the same entity across documents
6. **`method="shift_dates"`** — move every date by N days while preserving relative time (raw OpenMed no-op — see note below)
7. **round-trip** — keep the surrogate→original mapping, then `reidentify()` back to the original

The model is loaded once via a shared `ModelLoader` and reused across all calls
(the documented best practice), so the demo initializes the model a single time.

### Run it

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10 (this repo is verified on 3.11).

```bash
uv run python examples/deidentify_pii.py
```

`uv` reads [`pyproject.toml`](pyproject.toml), creates a `.venv`, and installs the
dependencies automatically. The first run downloads a small (~44M-parameter)
clinical PII model — `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` — from the
Hugging Face Hub and caches it under `~/.cache/openmed`, so later runs are fast
and fully offline.

### Native Apple Silicon (MLX)

On M-series Macs you can swap the portable Torch/Transformers backend for Apple's
native [MLX](https://github.com/ml-explore/mlx) backend:

```bash
uv sync --extra mlx
```

With the backend unset, openmed auto-detects MLX on Apple Silicon and falls back to
Hugging Face/PyTorch when it's unavailable. Pin it for the service with
`OPENMED_STUDIO_BACKEND=mlx` — but note an explicit `mlx` pin *raises* on a non-MLX host
rather than falling back.

The default model runs on MLX directly: it isn't pre-packaged, so openmed converts it on
the fly on first run and caches the result under `~/.cache/openmed/mlx/`. A pre-converted
`-mlx` repo (e.g. `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx`) is an optional
shortcut that skips conversion — pass it as a **local directory** via
`extract_pii(..., model_name=...)` / `deidentify(..., model_name=...)`. See the
[MLX backend docs](https://openmed.life/docs/mlx-backend/).

## Tests

```bash
uv run pytest                # fast tests only (model-loading tests are skipped)
uv run pytest --run-model    # also run tests that load the OpenMed PII model
uv run --extra ui pytest     # also run the Streamlit UI tests (needs the ui extra)
```

Tests live in [`tests/`](tests/). The fast tests need no model: `test_api.py` exercises the API
with a stubbed engine (including backend resolution and the `/health` report), `test_engine.py`
checks `PIIEngine`'s lazy-loading contract, backend selection, and that `deidentify` forwards
every method (including `shift_dates`) to OpenMed, and `test_pii_pure.py` covers OpenMed's
pure-Python surface (e.g. `reidentify` round-trips). The `--run-model` tests — `test_pii_model.py`
plus the `@pytest.mark.model` tests in the other two files — load the real model to verify
detection, masking, deterministic replacement, and round-trips. The UI tests are
`test_ui_helpers.py` (pure rendering/payload helpers — always run) and `test_ui_app.py` (drives
`streamlit_app.py` via Streamlit's `AppTest` with the network mocked; **skipped unless the `ui`
extra is installed**). Type-checking the UI also needs the extra
(`uv run --extra ui ty check`), since `streamlit_app.py` imports `streamlit` at module top.

## Notes

- All identifiers in the example note are fabricated.
- Smart entity merging is on by default (`use_smart_merging=True`) and recombines
  token-fragmented PII like dates and SSNs into whole spans.
- **`shift_dates` is a no-op with the default model.** OpenMed's shift path matches the
  literal label `"DATE"` (`openmed/core/pii.py:905`), but the default model emits lowercase
  `"date"`, so `shift_dates` *masks* dates instead of shifting them. The service delegates
  `shift_dates` to OpenMed like every other method (no workaround), so `POST /pii/deidentify`
  masks dates on the default model — switch to a model that emits canonical `"DATE"` labels for
  real shifting. The demo detects the no-op and prints a note.
- **OpenMed-REST compatibility (opt-in).** Set `OPENMED_STUDIO_COMPAT=1` to mount a `/compat`
  surface that mirrors [OpenMed's own REST service](https://openmed.life/docs/rest-service/):
  point a client's base URL at `<host>/compat`, and `POST /pii/{extract,deidentify}` accept the
  upstream body (including an ignored `keep_alive`) and return openmed's response shape
  (`pii_entities`, `num_entities_redacted`, `timestamp`, and the echoed `original_text`). It is
  off by default because echoing `original_text` returns the input (possible PHI).
- More guides: [OpenMed docs](https://openmed.life/docs/) ·
  [PII anonymization](https://openmed.life/docs/anonymization/) ·
  [smart merging](https://openmed.life/docs/pii-smart-merging/).
