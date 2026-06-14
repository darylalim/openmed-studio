# openmed-explore

Build with [OpenMed](https://openmed.life/docs/) — runnable examples for clinical NLP.

## PII / PHI de-identification

[`examples/deidentify_pii.py`](examples/deidentify_pii.py) is an end-to-end demo of
OpenMed's de-identification on a synthetic clinical note. It walks through:

1. **`extract_pii`** — detect PII spans (label, text, confidence, character offsets)
2. **`method="mask"`** — replace entities with `[LABEL]` placeholders
3. **`method="remove"`** — delete PII spans entirely
4. **`method="replace"`** — realistic, format-preserving [Faker](https://faker.readthedocs.io/) surrogates, made deterministic with `consistent=True, seed=...`
5. **`method="hash"`** — stable typed digests for linking the same entity across documents
6. **`method="shift_dates"`** — move every date by N days while preserving relative time (see limitation below)
7. **round-trip** — keep the surrogate→original mapping, then `reidentify()` back to the original

The model is loaded once via a shared `ModelLoader` and reused across all calls
(the documented best practice), so the demo initializes the model a single time.

### Run it

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10 (this repo is verified on 3.11).

```bash
uv run python examples/deidentify_pii.py
```

`uv` reads [`pyproject.toml`](pyproject.toml), creates a `.venv`, and installs
`openmed[hf]` automatically. The first run downloads a small (~44M-parameter)
clinical PII model — `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` — from the
Hugging Face Hub and caches it under `~/.cache/openmed`, so later runs are fast
and fully offline.

### Native Apple Silicon (MLX)

On M-series Macs you can swap the portable Torch/Transformers backend for Apple's
native [MLX](https://github.com/ml-explore/mlx) backend:

```bash
uv sync --extra mlx
```

MLX auto-detects on Apple Silicon and falls back to Hugging Face/PyTorch when
unavailable. It expects an MLX-packaged model (a repo id ending in `-mlx`, e.g.
`OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx`); pass it via
`extract_pii(..., model_name=...)` / `deidentify(..., model_name=...)`. See the
[MLX backend docs](https://openmed.life/docs/mlx-backend/).

## Tests

```bash
uv run pytest                # fast tests only (model-loading tests are skipped)
uv run pytest --run-model    # also run tests that load the OpenMed PII model
```

Tests live in [`tests/`](tests/): `test_pii_pure.py` covers the no-model surface (e.g.
`reidentify` round-trips), and `test_pii_model.py` covers real inference (entity detection,
masking, deterministic replacement, reidentify round-trip), gated behind `--run-model`.

## Notes

- All identifiers in the example note are fabricated.
- Smart entity merging is on by default (`use_smart_merging=True`) and recombines
  token-fragmented PII like dates and SSNs into whole spans.
- **`shift_dates` limitation (OpenMed 1.5.5):** with the default model, dates are
  *masked* rather than shifted. OpenMed's shift path matches the literal label
  `"DATE"` (`openmed/core/pii.py:905`), but this model emits lowercase `"date"`,
  so dates fall through to masking. The example detects this and prints a note.
  A model emitting canonical `"DATE"` labels would shift dates as documented.
- More guides: [OpenMed docs](https://openmed.life/docs/) ·
  [PII anonymization](https://openmed.life/docs/anonymization/) ·
  [smart merging](https://openmed.life/docs/pii-smart-merging/).
