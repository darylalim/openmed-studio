"""Pure, framework-free helpers for the Streamlit UI (``streamlit_app.py``).

Deliberately imports no Streamlit and no network libraries, so the HTML-escaping,
entity-highlighting, and request-payload logic can be unit-tested in isolation
(``tests/test_ui_helpers.py``) without a browser, a server, or the model.
"""

from __future__ import annotations

import html
from typing import Any, NamedTuple


class Tint(NamedTuple):
    """One entity hue as a translucent tint per theme mode (Night / Day Rounds)."""

    dark: str
    light: str

    @property
    def css(self) -> str:
        """``background-color`` declarations that follow the active theme mode.

        ``light-dark()`` resolves against the ``color-scheme`` Streamlit sets on its
        element tree for the active theme, so the browser picks the tint and no Python
        code reads the theme. The plain dark tint comes first as the fallback for a
        browser without ``light-dark()`` (which drops the second declaration): it still
        reads on white, only fainter.
        """
        return (
            f"background-color:{self.dark};"
            f"background-color:light-dark({self.light},{self.dark})"
        )


# The entity hues of ``.streamlit/config.toml``'s two themes as translucent per-label
# tints: nine hues ~40° apart on the OKLCH wheel (dark: lightness 0.72 / chroma 0.13;
# light: 0.70 / 0.15; cyan and teal trimmed to fit sRGB), so no two read as "the same
# blue". Marks pair the tint with ``color: inherit`` so the text always takes the
# active theme's color. Alphas are solved per hue so every tint lifts its canvas by the
# same amount — ≈2.0:1 on Night Rounds' #121b21, ≈1.45:1 on Day Rounds' #fcfeff — with
# body text ≥7:1 (WCAG AAA) on every dark tint and ≥10.8:1 on every light one. Nine
# and not ten is deliberate: ``color_for`` hashes with ``sum(ord(c)) % len(PALETTE)``,
# and at ten the two most common labels in a clinical note — ``first_name`` and
# ``date`` — collide. The ORDER is a choice too: each slot's most frequent PII-model
# labels are noted, and hues are placed so labels that sit side by side in a note
# differ — first/last name (slots 2, 3) are blue vs orange.
PALETTE: list[Tint] = [
    # 0 cyan     date, city
    Tint("rgba(9,183,220,.36)", "rgba(12,176,212,.40)"),
    # 1 green    medical_record_number, postcode
    Tint("rgba(109,186,112,.35)", "rgba(91,182,97,.43)"),
    # 2 blue     first_name
    Tint("rgba(119,164,246,.36)", "rgba(106,156,251,.41)"),
    # 3 orange   last_name
    Tint("rgba(226,141,79,.38)", "rgba(228,130,51,.39)"),
    # 4 yellow   age, date_of_birth, phone_number
    Tint("rgba(194,161,50,.37)", "rgba(190,154,7,.41)"),
    # 5 teal     state, country
    Tint("rgba(16,189,175,.35)", "rgba(12,182,168,.40)"),
    # 6 violet   street_address, occupation; NER DISEASE, CHEM
    Tint("rgba(180,144,232,.37)", "rgba(176,134,235,.40)"),
    # 7 red      email, ssn
    Tint("rgba(235,129,127,.39)", "rgba(237,116,115,.38)"),
    # 8 magenta  ORGANIZATION, gender; NER GENE
    Tint("rgba(222,130,183,.38)", "rgba(222,117,180,.38)"),
]


def color_for(label: str) -> Tint:
    """Stable highlight tint for an entity label (same label → same tint).

    Emit it with ``Tint.css``, which lets the browser pick the tint for the active
    theme mode; marks pair it with ``color: inherit`` so the text takes the theme's
    text color.
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

    Marks use a translucent per-label tint that follows the theme mode via CSS
    ``light-dark()``, plus ``color: inherit``, so they read correctly on either the
    light or dark theme with no runtime theme detection. All
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
            f'<mark style="{color_for(label).css};color:inherit;'
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
            f'<span style="{color_for(label).css};color:inherit;'
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
    """Build the shared de-identify request body from a tab's de-identify controls.

    The controls come from the Single note and Batch tabs (the ``Method`` row, the
    confidence slider, ``Keep mapping`` and the ``Advanced`` knobs) or from the
    Anonymize tab's form, which pins the method, mapping and sweep; the sidebar
    contributes only ``lang``. ``seed`` is included only when ``consistent`` is on; ``locale`` only for the surrogate methods
    ``replace``/``format_preserve`` (and only when non-empty); ``date_shift_days`` and
    ``keep_year`` only for ``shift_dates`` — so the payload carries just the fields the
    chosen method actually consumes. A ``date_shift_days`` of 0 (the control's default)
    is omitted so openmed applies its per-note random shift rather than shifting by
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
    ``keep_mapping`` (the policy decides reversibility; the service sends ``keep_mapping=False`` and
    lets the profile's own flag decide). ``seed`` is included only when ``consistent`` is on, and
    ``locale`` only when non-empty — ``seed`` matters only to the surrogate (``replace``-based)
    policies, while a locale also adds its region's identifier patterns under any policy. ``text``
    is not in the payload (the caller passes it positionally to ``service.anonymize_policy``).
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
