# openmed-deid

**PII / PHI de-identification for clinical text**, built on
[OpenMed](https://openmed.life/docs/).

This project is deliberately narrow in scope: detect and de-identify protected health
information in clinical notes — and nothing else. It ships a
[FastAPI](https://fastapi.tiangolo.com/) service (the primary deliverable) plus a
library-level demo of the same OpenMed calls.

## The de-identification API

A small FastAPI service in [`openmed_deid/`](openmed_deid/) wraps OpenMed's PII functions
behind HTTP. A reusable, framework-free [`PIIEngine`](openmed_deid/engine.py) holds one
shared `ModelLoader`; the endpoints are thin adapters over it.

```bash
uv run uvicorn openmed_deid.main:app --port 8080   # or: uv run python -m openmed_deid
```

Interactive docs are then at `http://127.0.0.1:8080/docs`. Endpoints:

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness; reports the configured model, backend, lazy-load state, and whether auth is on |
| `POST /pii/extract` | Detect PII entities (label, text, char offsets, confidence) |
| `POST /pii/deidentify` | Redact via `mask` / `remove` / `replace` / `hash` / `shift_dates`; returns the surrogate→original `mapping` when `keep_mapping=true` |
| `POST /pii/deidentify/batch` | De-identify up to 100 texts in one call (`{"items": [...]}`) |
| `POST /pii/reidentify` | Restore originals from a kept mapping |

Request options (see `/docs` for the full schema): `lang` (9 supported languages),
`model_name` (override the default model), `confidence_threshold`, `consistent`/`seed`
(deterministic `replace`), and `date_shift_days`/`keep_year` (for `shift_dates`). `text` is
capped at 50k characters.

```bash
curl -s localhost:8080/pii/deidentify -H 'content-type: application/json' \
  -d '{"text": "Patient John Doe, SSN 123-45-6789.", "method": "mask"}'
# → {"deidentified_text": "Patient [first_name] [last_name], SSN [ssn].", ...}
```

The model loads lazily on the first `/pii/*` request and is then reused across all
subsequent requests (one shared `ModelLoader`).

The inference backend is auto-detected (MLX on Apple Silicon when the `mlx` extra is
installed, else Hugging Face/PyTorch); pin it explicitly with `OPENMED_DEID_BACKEND=hf|mlx`,
which `/health` echoes back.

### Authentication & PHI safety

The `/pii/*` endpoints process Protected Health Information. By default (no
`OPENMED_DEID_API_KEY` set) the service runs **unauthenticated** for local use and logs a
startup warning. Before exposing it on a network or sending it real PHI:

- **Set `OPENMED_DEID_API_KEY`** — every `/pii/*` request must then send a matching
  `X-API-Key` header (else `401`); `/health` stays open.
- **Terminate TLS** in front of it (e.g. a reverse proxy) — the API itself speaks plain HTTP.
- **Treat any returned `mapping` as re-identification material** — it is as sensitive as the
  raw PHI; store it encrypted and access-controlled, never beside the redacted text.

```bash
export OPENMED_DEID_API_KEY=$(openssl rand -hex 16)
curl -s localhost:8080/pii/extract -H "X-API-Key: $OPENMED_DEID_API_KEY" \
  -H 'content-type: application/json' -d '{"text": "Patient John Doe."}'
```

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
`OPENMED_DEID_BACKEND=mlx` — but note an explicit `mlx` pin *raises* on a non-MLX host
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
```

Tests live in [`tests/`](tests/). The fast tests need no model: `test_api.py` exercises the API
with a stubbed engine (including backend resolution and the `/health` report), `test_engine.py`
checks `PIIEngine`'s lazy-loading contract, backend selection, and the engine-side `shift_dates`,
and `test_pii_pure.py` covers OpenMed's pure-Python surface (e.g. `reidentify` round-trips). The
`--run-model` tests — `test_pii_model.py` plus the `@pytest.mark.model` tests in the other two
files — load the real model to verify detection, masking, deterministic replacement, round-trips,
and real date shifting.

## Notes

- All identifiers in the example note are fabricated.
- Smart entity merging is on by default (`use_smart_merging=True`) and recombines
  token-fragmented PII like dates and SSNs into whole spans.
- **`shift_dates` — the API shifts; raw OpenMed 1.5.5 does not.** OpenMed's own
  shift path matches the literal label `"DATE"` (`openmed/core/pii.py:905`), but the
  default model emits lowercase `"date"`, so calling `openmed.deidentify` directly
  *masks* dates instead of shifting them. The service works around this: `PIIEngine`
  handles `method="shift_dates"` itself, so `POST /pii/deidentify` really shifts dates
  (format-preserving, one consistent offset, `keep_year` honored). The standalone demo
  still calls raw OpenMed, detects the no-op, and prints a note.
- More guides: [OpenMed docs](https://openmed.life/docs/) ·
  [PII anonymization](https://openmed.life/docs/anonymization/) ·
  [smart merging](https://openmed.life/docs/pii-smart-merging/).
