# openmed-studio

**A clinical-NLP application built on [OpenMed](https://openmed.life/docs/).**

The goal is an app over OpenMed's full toolkit — clinical NER, PII/PHI de-identification,
anonymization, and zero-shot extraction. **Today it implements PII/PHI de-identification**
(the rest is on the roadmap), shipped as a [Streamlit](https://streamlit.io/) app.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10 (verified on 3.11).

```bash
uv run streamlit run streamlit_app.py
```

`uv` reads [`pyproject.toml`](pyproject.toml), creates a `.venv`, installs the dependencies,
and opens the app at `http://localhost:8501`. The first de-identification downloads a small
(~44M-parameter) clinical PII model — `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` — from
the Hugging Face Hub and caches it under `~/.cache/openmed`, so later runs are fast and offline.

## The app

The model runs **in-process**: [`streamlit_app.py`](streamlit_app.py) calls a reusable,
framework-free [`PIIEngine`](openmed_studio/engine.py) (one shared `ModelLoader`) through the
in-process seam in [`openmed_studio/service.py`](openmed_studio/service.py), which validates each
request and adapts OpenMed's results for the UI. There is no separate service to start.

It opens with four tabs:

- **Detect** — detect PII entities and highlight them (with a color legend) plus an entity table,
  without redacting — for auditing what the model finds before choosing a method.
- **Single note** — de-identify one note; shows the original with detected PII highlighted
  side-by-side with the redacted text (with a one-click **copy** button), plus an entity table and
  (optionally) the mapping.
- **Batch** — edit a table of notes (up to 100) and de-identify them in one go.
- **Re-identify** — restore originals from a kept mapping (auto-filled from the last single-note run).

The sidebar reports the engine's model/backend/load state and holds the shared de-identification
options: method (`mask` / `remove` / `replace` / `hash` / `shift_dates`), language (9 supported),
confidence, `keep_mapping`, and the deterministic-`replace` / `shift_dates` controls. The
confidence slider defaults to **0.5** for higher PHI recall — note the de-identify default is
`0.7`. The model loads on the first request, so the first call shows a spinner and is slower than
the rest.

### How it works

- **Validation is in-process.** The Pydantic models in
  [`openmed_studio/validation.py`](openmed_studio/validation.py) are reused as a validation layer:
  every request is checked before it reaches the model, so the per-request **text cap** (50k chars,
  override with `OPENMED_STUDIO_MAX_TEXT_LENGTH`), the **batch** (≤100) and **mapping** (≤5,000)
  bounds, the language/method enums, and the confidence range all still apply. Invalid input is
  surfaced as an error in the UI with the offending text stripped (so PHI isn't echoed back).
- **Backend.** Inference is auto-detected: MLX on Apple Silicon when the `mlx` extra is installed,
  else Hugging Face/PyTorch (runs everywhere — CPU, CUDA, Apple MPS). Pin it with
  `OPENMED_STUDIO_BACKEND=hf|mlx` (an explicit `mlx` pin *raises* on a non-MLX host).
- **Model reuse.** Streamlit caches the engine (`st.cache_resource`), so the model loads at most
  once per process and is reused across every tab and request.
- **Theme-aware.** The entity highlights and legend adapt to light or dark mode automatically — they
  use a translucent tint plus `color: inherit`, so they read correctly on either with no runtime
  theme detection (both modes are defined in `.streamlit/config.toml`).
- **Isolated reruns.** The Detect/Batch/Re-identify tabs are `st.fragment`s, so an interaction in one
  doesn't rerun the others; Single note stays a full rerun so it can hand its result to Re-identify.

### What we dropped vs the old service

openmed-studio used to be a FastAPI service with the UI as a thin HTTP client. Collapsing to a
single local Streamlit app intentionally drops the guarantees that only existed because of the
HTTP boundary:

- **API-key auth** (`OPENMED_STUDIO_API_KEY` / `X-API-Key`) — there is no network endpoint to
  protect. This app is meant to run **locally for a single trusted user**; put it behind your own
  auth or a reverse proxy before exposing it, and don't run it on a network with real PHI as-is.
- **The JSON error envelope** and the OpenMed-REST `/compat` surface — there is no API to be
  compatible with.

The protections that guard the *model* regardless of transport — the text/batch/mapping caps, the
value checks, backend pinning, and not echoing input on a validation error — are **kept**, enforced
in-process by the service seam. Treat any returned `mapping` as re-identification material: it is as
sensitive as the raw PHI.

## Native Apple Silicon (MLX)

On M-series Macs you can swap the portable Torch/Transformers backend for Apple's native
[MLX](https://github.com/ml-explore/mlx) backend:

```bash
uv sync --extra mlx
```

With the backend unset, openmed auto-detects MLX on Apple Silicon and falls back to
Hugging Face/PyTorch when it's unavailable. Pin it with `OPENMED_STUDIO_BACKEND=mlx` — but note an
explicit `mlx` pin *raises* on a non-MLX host rather than falling back.

The default model runs on MLX directly: it isn't pre-packaged, so openmed converts it on the fly on
first run and caches the result under `~/.cache/openmed/mlx/`. A pre-converted `-mlx` repo (e.g.
`OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx`) is an optional shortcut that skips conversion —
pass it as a **local directory** via `extract_pii(..., model_name=...)` /
`deidentify(..., model_name=...)`. See the [MLX backend docs](https://openmed.life/docs/mlx-backend/).

## Tests

```bash
uv run pytest                # fast tests only (model-loading tests are skipped)
uv run pytest --run-model    # also run tests that load the OpenMed PII model
```

Tests live in [`tests/`](tests/). The fast tests need no model: `test_service.py` covers the
in-process seam (backend resolution, the dict adapters, the error taxonomy) with a stub engine,
`test_validation.py` pins the surviving input guarantees (caps, enums, format checks, the text-cap
knob, the `DeidMethod`↔openmed sync, and PHI-non-echo), `test_engine.py` checks `PIIEngine`'s
lazy-loading contract and that `deidentify` forwards every method (including `shift_dates`) to
OpenMed, `test_pii_pure.py` covers OpenMed's pure-Python surface, and `test_ui_helpers.py` unit-tests
the pure rendering/payload helpers. `test_ui_app.py` drives `streamlit_app.py` via Streamlit's
`AppTest` with the engine stubbed in-process (covering the theme-aware highlighting and the fragmented
tabs). The `--run-model` tests — `test_pii_model.py` plus the
`@pytest.mark.model` tests in `test_engine.py` — load the real model to verify detection, masking,
deterministic replacement, and round-trips.

Lint, format, and type-check with the project-pinned tools:

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run ty check              # type-check
```

CI (`.github/workflows/ci.yml`) runs these checks and the fast test suite on every push and pull
request, across Python 3.10 and 3.13.

## Notes

- All identifiers in the app's sample note are fabricated.
- Smart entity merging is on by default (`use_smart_merging=True`) and recombines
  token-fragmented PII like dates and SSNs into whole spans.
- More guides: [OpenMed docs](https://openmed.life/docs/) ·
  [PII anonymization](https://openmed.life/docs/anonymization/) ·
  [smart merging](https://openmed.life/docs/pii-smart-merging/).
