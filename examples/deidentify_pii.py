"""Working example: PII / PHI de-identification with OpenMed.

Run it (no manual install needed — uv resolves the deps from pyproject.toml):

    uv run python examples/deidentify_pii.py

The first run downloads a small (~44M-parameter) clinical PII model from the
Hugging Face Hub — OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 — and caches
it under ~/.cache/openmed, so later runs are fast and fully offline.

What it shows, end to end on one synthetic clinical note:
  1. extract_pii      — detect PII spans (label, text, confidence, offsets)
  2. method="mask"    — replace entities with [LABEL] placeholders
  3. method="remove"  — delete PII spans entirely
  4. method="replace" — realistic, format-preserving Faker surrogates (deterministic)
  5. method="hash"    — stable typed digests for cross-document linking
  6. method="shift_dates" — move dates by N days, preserving relative time*
  7. round-trip       — keep the mapping, then reidentify() back to the original

  * See the note printed by section 6: with this model, dates are masked rather
    than shifted because OpenMed 1.5.5's shift path matches a literal "DATE"
    label while this model emits lowercase "date".
"""

from __future__ import annotations

import warnings
from typing import Literal

# pysbd (a transitive dependency) raises SyntaxWarnings from its regex string
# literals on Python 3.12+. They are harmless; silence them before openmed
# imports pysbd so the demo output stays readable.
warnings.filterwarnings("ignore", category=SyntaxWarning)

from openmed import ModelLoader, deidentify, extract_pii, reidentify  # noqa: E402

# The de-identification strategies that deidentify() accepts.
DeidMethod = Literal["mask", "remove", "replace", "hash", "shift_dates"]

# A synthetic clinical note. Every identifier below is fabricated.
NOTE = (
    "Patient: John A. Doe (MRN: 1234567). DOB: 01/15/1970. "
    "Seen on 03/22/2024 by Dr. Emily Carter at Springfield General Hospital. "
    "Contact: john.doe@example.com, phone (415) 555-0137. "
    "SSN: 123-45-6789. Address: 742 Evergreen Terrace, Springfield, IL 62704. "
    "Assessment: Type 2 diabetes mellitus, well controlled on metformin."
)

# Load the clinical PII model once and reuse it across every call below — the
# documented best practice (avoids re-initializing the model on each call).
LOADER = ModelLoader()


def hr(title: str) -> None:
    """Print a section header."""
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def get_entities(result):
    """extract_pii may return a list[PIIEntity] or a result object exposing them."""
    for attr in ("entities", "pii_entities"):
        if hasattr(result, attr):
            return getattr(result, attr)
    return result  # already a list


def deid_text(result) -> str:
    """deidentify() returns a DeidentificationResult; tolerate a plain str too."""
    return result if isinstance(result, str) else result.deidentified_text


def deid(method: DeidMethod, **kwargs) -> str:
    """Run deidentify() with the shared loader and return the de-identified text."""
    return deid_text(deidentify(NOTE, method=method, loader=LOADER, **kwargs))


def main() -> None:
    print("Original note:\n")
    print(f"  {NOTE}")

    # 1) Detect PII entities --------------------------------------------------
    hr("1. extract_pii — what PII is in the text?")
    entities = get_entities(extract_pii(NOTE, use_smart_merging=True, loader=LOADER))
    if not entities:
        print("  (no entities detected — try lowering confidence_threshold)")
    for e in entities:
        conf = f"{e.confidence:.2f}" if e.confidence is not None else "  n/a"
        print(f"  {e.label:<22} {e.text!r:<34} conf={conf}  [{e.start}:{e.end}]")

    # 2) Mask -----------------------------------------------------------------
    hr("2. deidentify(method='mask') — replace with [LABEL] placeholders")
    masked = deid("mask")
    print(masked)

    # 3) Remove ---------------------------------------------------------------
    hr("3. deidentify(method='remove') — delete PII spans entirely")
    print(deid("remove"))

    # 4) Replace with realistic, deterministic fakes --------------------------
    hr("4. deidentify(method='replace') — Faker surrogates, format-preserving")
    print("Deterministic (consistent=True, seed=42) — same input always maps the same:")
    print(deid("replace", consistent=True, seed=42))

    # 5) Hash — stable pseudonyms for cross-document linking ------------------
    hr("5. deidentify(method='hash') — stable typed digests")
    print(deid("hash"))

    # 6) Shift dates while preserving relative time ---------------------------
    hr("6. deidentify(method='shift_dates', date_shift_days=180)")
    shifted = deid("shift_dates", date_shift_days=180)
    print(shifted)
    if shifted == masked:
        print(
            "\n  NOTE: dates were masked, not shifted. OpenMed 1.5.5's shift_dates path\n"
            '  matches the literal label "DATE" (openmed/core/pii.py:905), but this model\n'
            '  emits lowercase "date", so dates fall through to masking. Use a model that\n'
            '  emits canonical "DATE" labels to see real date shifting.'
        )

    # 7) Round-trip: keep the mapping, then reidentify ------------------------
    hr("7. Round-trip — replace with keep_mapping=True, then reidentify()")
    res = deidentify(
        NOTE,
        method="replace",
        consistent=True,
        seed=7,
        keep_mapping=True,
        loader=LOADER,
    )
    redacted = deid_text(res)
    mapping = getattr(res, "mapping", None)
    print("De-identified:\n  " + redacted)
    if mapping:
        restored = reidentify(redacted, mapping)
        match = (
            "exactly matches the original" if restored == NOTE else "restores the text"
        )
        print(
            f"\nMapping holds {len(mapping)} surrogate→original entries; "
            f"reidentify() {match}:"
        )
        print("  " + restored)
    else:
        print(
            "\n(no mapping returned — keep_mapping may be unsupported in this version)"
        )


if __name__ == "__main__":
    main()
