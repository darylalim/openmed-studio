"""Pure, framework-free helpers for the Streamlit UI (``streamlit_app.py``).

Deliberately imports no Streamlit and no network libraries, so the HTML-escaping,
entity-highlighting, and request-payload logic can be unit-tested in isolation
(``tests/test_ui_helpers.py``) without a browser, a server, or the model.
"""

from __future__ import annotations

import html
from typing import Any

# Translucent per-label tints. Each reads as a highlight over either a light or a
# dark page, and marks pair the tint with ``color: inherit`` so the text always
# takes the active theme's color — so the highlight is correct on any theme with no
# runtime theme detection (the alpha is tuned to stay visible on white and on dark).
PALETTE: list[str] = [
    "rgba(253,230,138,.40)",
    "rgba(147,197,253,.42)",
    "rgba(134,239,172,.40)",
    "rgba(252,165,165,.42)",
    "rgba(196,181,253,.45)",
    "rgba(244,164,212,.42)",
    "rgba(110,231,183,.40)",
    "rgba(253,186,116,.42)",
    "rgba(165,180,252,.45)",
    "rgba(94,234,212,.40)",
]


def color_for(label: str) -> str:
    """Stable highlight tint for an entity label (same label → same tint).

    The tint is translucent so it reads over either a light or dark page; marks
    pair it with ``color: inherit`` so the text takes the active theme's color.
    """
    return PALETTE[sum(ord(c) for c in label) % len(PALETTE)]


def _block(body: str) -> str:
    return (
        '<div style="white-space:pre-wrap;line-height:1.9;'
        "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
        f'font-size:.9rem">{body}</div>'
    )


def render_highlighted(text: str, entities: list[dict[str, Any]]) -> str:
    """HTML for ``text`` with non-overlapping entity spans highlighted by label.

    Marks use a translucent per-label tint plus ``color: inherit``, so they read
    correctly on either a light or dark theme with no runtime theme detection. All
    text is HTML-escaped (the clinical note is untrusted input). Entities are
    applied left-to-right; any span that overlaps an already-applied one or falls
    outside ``text`` is skipped, and entities without a ``start`` are ignored.
    """
    spans = sorted(
        (
            e
            for e in entities
            if e.get("start") is not None and e.get("end") is not None
        ),
        key=lambda e: (int(e["start"]), int(e["end"])),
    )
    out: list[str] = []
    cursor = 0
    for entity in spans:
        start, end = int(entity["start"]), int(entity["end"])
        if start < cursor or start >= end or end > len(text):
            continue  # skip overlapping or out-of-range spans
        out.append(html.escape(text[cursor:start]))
        label = str(entity.get("label", ""))
        out.append(
            f'<mark style="background-color:{color_for(label)};color:inherit;'
            'padding:0 .15em;border-radius:.2em" '
            f'title="{html.escape(label)}">{html.escape(text[start:end])}'
            '<span style="font-size:.7em;font-weight:600;opacity:.7;'
            f'margin-left:.25em">{html.escape(label)}</span></mark>'
        )
        cursor = end
    out.append(html.escape(text[cursor:]))
    return _block("".join(out))


def render_plain(text: str) -> str:
    """HTML for ``text`` with no highlighting (escaped, whitespace preserved)."""
    return _block(html.escape(text))


def render_legend(entities: list[dict[str, Any]]) -> str:
    """HTML legend: one pill per distinct label, colored to match the marks.

    Pills use the same translucent tint + ``color: inherit`` as the marks, so they
    read on either theme. Returns an empty string when there are no labelled
    entities. Labels keep first-seen order so the legend is stable across renders.
    """
    labels: list[str] = []
    for entity in entities:
        label = str(entity.get("label", ""))
        if label and label not in labels:
            labels.append(label)
    if not labels:
        return ""
    pills: list[str] = []
    for label in labels:
        pills.append(
            f'<span style="background-color:{color_for(label)};color:inherit;'
            "padding:.05em .45em;border-radius:.7em;font-size:.72rem;"
            f'margin:0 .3em .3em 0;display:inline-block">{html.escape(label)}</span>'
        )
    return f'<div style="margin:.3rem 0 .1rem">{"".join(pills)}</div>'


def build_base_opts(
    *,
    method: str,
    confidence_threshold: float,
    lang: str,
    keep_mapping: bool,
    consistent: bool,
    seed: int,
    locale: str | None = None,
    date_shift_days: int,
    keep_year: bool,
    use_safety_sweep: bool,
) -> dict[str, Any]:
    """Build the shared de-identify request body from the sidebar options.

    ``seed`` is included only when ``consistent`` is on; ``locale`` only for the
    surrogate methods ``replace``/``format_preserve`` (and only when non-empty);
    ``date_shift_days`` and ``keep_year`` only for ``shift_dates`` — so the payload
    carries just the fields the chosen method actually consumes. A ``date_shift_days``
    of 0 (the sidebar default) is
    omitted so openmed applies its per-note random shift rather than shifting by
    zero — shifting by zero would leave dates in the output verbatim.
    """
    opts: dict[str, Any] = {
        "method": method,
        "confidence_threshold": confidence_threshold,
        "lang": lang,
        "keep_mapping": keep_mapping,
        "consistent": consistent,
        "use_safety_sweep": use_safety_sweep,
    }
    if consistent:
        opts["seed"] = int(seed)
    if method in ("replace", "format_preserve") and locale and locale.strip():
        # Faker locale for the surrogate methods; empty means "use openmed's default
        # derived from lang", so omit it. Only replace/format_preserve consume it.
        opts["locale"] = locale.strip()
    if method == "shift_dates":
        # 0 (the number_input default) means "unset": omit it so openmed applies its
        # per-note random shift rather than shifting by zero (which leaves dates verbatim).
        if date_shift_days:
            opts["date_shift_days"] = int(date_shift_days)
        opts["keep_year"] = keep_year
    return opts


def build_policy_opts(
    *,
    policy: str,
    confidence_threshold: float,
    lang: str,
    consistent: bool,
    seed: int,
    locale: str | None = None,
    use_safety_sweep: bool,
) -> dict[str, Any]:
    """Build the policy-anonymize request body from the Policy de-ID tab's controls.

    The policy sibling of :func:`build_base_opts`, with two deliberate omissions: no ``method``
    (the policy selects the per-label action, so a method would be silently overridden) and no
    ``keep_mapping`` (the policy decides reversibility; the service always requests it). ``seed``
    is included only when ``consistent`` is on, and ``locale`` only when non-empty — both apply to
    the surrogate (``replace``-based) policies and are ignored by the masking ones. ``text`` is not
    in the payload (the caller passes it positionally to ``service.anonymize_policy``).
    """
    opts: dict[str, Any] = {
        "policy": policy,
        "confidence_threshold": confidence_threshold,
        "lang": lang,
        "consistent": consistent,
        "use_safety_sweep": use_safety_sweep,
    }
    if consistent:
        opts["seed"] = int(seed)
    if locale and locale.strip():
        opts["locale"] = locale.strip()
    return opts


def build_batch_table(
    notes: list[str], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pair each note with its de-identification result for the batch table.

    The in-process service returns exactly one result per item, in order, so notes and
    results zip 1:1 (``zip`` stops at the shorter if they ever diverge). Each result is
    tagged ``ok``; a failed note (``ok`` False) shows ``Failed`` in the status column with
    its error message in place of the de-identified text, so one bad note stays visible
    rather than aborting the whole batch.
    """
    rows: list[dict[str, Any]] = []
    for note, item in zip(notes, results):
        if item.get("ok", True):
            rows.append(
                {
                    "status": "OK",
                    "original": note,
                    "deidentified": item["deidentified_text"],
                    "entities": len(item["entities"]),
                }
            )
        else:
            rows.append(
                {
                    "status": "Failed",
                    "original": note,
                    "deidentified": item.get("error", "failed"),
                    "entities": 0,
                }
            )
    return rows
