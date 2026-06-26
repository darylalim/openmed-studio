"""Unit tests for the pure UI helpers in ``ui_helpers`` (no Streamlit, no model).

``ui_helpers`` imports only the stdlib (no Streamlit, no network), so they run in
the default fast suite with no extras and no model.
"""

from __future__ import annotations

import re

from ui_helpers import (
    PALETTE,
    build_base_opts,
    build_batch_table,
    color_for,
    render_highlighted,
    render_legend,
    render_plain,
)

_TAG = re.compile(r"<[^>]+>")


def _strip_tags(markup: str) -> str:
    return _TAG.sub("", markup)


# --- color_for ----------------------------------------------------------------
def test_color_for_is_deterministic_and_in_palette():
    assert color_for("ssn") == color_for("ssn")
    assert color_for("ssn") in PALETTE


def test_color_for_handles_empty_label():
    assert color_for("") == PALETTE[0]


def test_color_for_returns_translucent_tint():
    # Tints are translucent (rgba) so the marks read on light and dark themes alike.
    assert color_for("ssn").startswith("rgba(")


# --- render_plain -------------------------------------------------------------
def test_render_plain_escapes_html_special_chars():
    out = render_plain('a < b & "c" > d')
    assert "&lt;" in out
    assert "&amp;" in out
    assert "&gt;" in out
    # the literal "< b" must not survive as a real tag
    assert "< b" not in out


# --- render_highlighted -------------------------------------------------------
def test_render_highlighted_wraps_entity_with_label_and_color():
    text = "SSN 123-45-6789 end"
    ents = [{"label": "ssn", "text": "123-45-6789", "start": 4, "end": 15}]
    out = render_highlighted(text, ents)
    assert out.count("<mark") == 1
    assert "123-45-6789" in out
    assert "ssn" in out  # label is shown
    assert color_for("ssn") in out  # background tint applied
    assert "color:inherit" in out  # text uses the active theme color
    assert "end" in out  # trailing text preserved


def test_render_highlighted_escapes_untrusted_text():
    # The clinical note is untrusted: HTML must be neutralized, not rendered.
    text = "<script>alert(1)</script> John"
    ents = [{"label": "first_name", "start": 26, "end": 30}]  # "John"
    out = render_highlighted(text, ents)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_render_highlighted_skips_overlapping_spans():
    text = "0123456789"
    ents = [
        {"label": "a", "start": 0, "end": 5},
        {"label": "b", "start": 3, "end": 8},  # overlaps the first
    ]
    out = render_highlighted(text, ents)
    assert out.count("<mark") == 1


def test_render_highlighted_skips_out_of_range_spans():
    text = "short"
    ents = [{"label": "x", "start": 2, "end": 99}]  # end past len(text)
    out = render_highlighted(text, ents)
    assert "<mark" not in out


def test_render_highlighted_ignores_entities_without_start():
    text = "hello world"
    ents = [{"label": "x", "text": "world"}]  # no start/end
    out = render_highlighted(text, ents)
    assert "<mark" not in out


def test_render_highlighted_ignores_entity_with_start_but_no_end():
    # A start without a usable end must be skipped, not crash the sort key.
    text = "hello world"
    for ents in (
        [{"label": "x", "start": 4}],
        [{"label": "x", "start": 4, "end": None}],
    ):
        out = render_highlighted(text, ents)
        assert "<mark" not in out


def test_render_highlighted_skips_zero_width_span():
    text = "0123456789"
    ents = [{"label": "x", "start": 3, "end": 3}]  # start == end
    out = render_highlighted(text, ents)
    assert "<mark" not in out


def test_render_highlighted_allows_adjacent_spans():
    text = "0123456789"
    ents = [
        {"label": "a", "start": 0, "end": 5},
        {"label": "b", "start": 5, "end": 10},  # starts exactly where the first ends
    ]
    out = render_highlighted(text, ents)
    assert out.count("<mark") == 2


def test_render_highlighted_sorts_out_of_order_entities():
    text = "John saw Mary"
    ents = [
        {"label": "b", "start": 9, "end": 13},  # "Mary" (given first)
        {"label": "a", "start": 0, "end": 4},  # "John"
    ]
    out = render_highlighted(text, ents)
    assert out.count("<mark") == 2
    assert out.index("John") < out.index("Mary")  # applied left-to-right


def test_render_highlighted_without_entities_equals_render_plain():
    text = "a & b < c"
    assert render_highlighted(text, []) == render_plain(text)


def test_render_highlighted_keeps_all_visible_text():
    text = "Patient John Doe seen today"
    ents = [{"label": "first_name", "start": 8, "end": 12}]  # "John"
    visible = _strip_tags(render_highlighted(text, ents))
    for fragment in ("Patient ", "John", " Doe seen today"):
        assert fragment in visible


# --- render_legend ------------------------------------------------------------
def test_render_legend_one_pill_per_distinct_label():
    ents = [
        {"label": "ssn", "start": 0, "end": 1},
        {"label": "first_name", "start": 2, "end": 3},
        {"label": "ssn", "start": 4, "end": 5},  # duplicate label
    ]
    out = render_legend(ents)
    assert out.count("<span") == 2  # ssn + first_name, deduped
    assert "ssn" in out and "first_name" in out
    assert color_for("ssn") in out and color_for("first_name") in out


def test_render_legend_empty_without_labels():
    assert render_legend([]) == ""
    assert render_legend([{"start": 0, "end": 1}]) == ""  # no label key


def test_render_legend_escapes_label():
    out = render_legend([{"label": "<x>", "start": 0, "end": 1}])
    assert "<x>" not in out
    assert "&lt;x&gt;" in out


# --- build_base_opts ----------------------------------------------------------
def test_build_base_opts_mask_omits_seed_and_date_fields():
    opts = build_base_opts(
        method="mask",
        confidence_threshold=0.5,
        lang="en",
        keep_mapping=True,
        consistent=False,
        seed=7,
        date_shift_days=30,
        keep_year=True,
        use_safety_sweep=True,
    )
    assert opts == {
        "method": "mask",
        "confidence_threshold": 0.5,
        "lang": "en",
        "keep_mapping": True,
        "consistent": False,
        "use_safety_sweep": True,
    }


def test_build_base_opts_consistent_replace_includes_seed():
    opts = build_base_opts(
        method="replace",
        confidence_threshold=0.7,
        lang="fr",
        keep_mapping=False,
        consistent=True,
        seed=42,
        date_shift_days=0,
        keep_year=True,
        use_safety_sweep=True,
    )
    assert opts["consistent"] is True
    assert opts["seed"] == 42
    assert "date_shift_days" not in opts


def test_build_base_opts_seed_tracks_consistent_not_method():
    # seed is gated on `consistent`, independent of method.
    with_seed = build_base_opts(
        method="mask",
        confidence_threshold=0.5,
        lang="en",
        keep_mapping=True,
        consistent=True,
        seed=5,
        date_shift_days=0,
        keep_year=True,
        use_safety_sweep=True,
    )
    assert with_seed["seed"] == 5  # non-replace method still gets seed when consistent
    without_seed = build_base_opts(
        method="replace",
        confidence_threshold=0.5,
        lang="en",
        keep_mapping=True,
        consistent=False,
        seed=5,
        date_shift_days=0,
        keep_year=True,
        use_safety_sweep=True,
    )
    assert "seed" not in without_seed  # replace without consistent omits seed


# --- build_batch_table --------------------------------------------------------
def test_build_batch_table_pairs_notes_with_results():
    notes = ["note A", "note B"]
    results = [
        {"ok": True, "deidentified_text": "A_DEID", "entities": [{"label": "x"}]},
        {"ok": True, "deidentified_text": "B_DEID", "entities": []},
    ]
    assert build_batch_table(notes, results) == [
        {"status": "OK", "original": "note A", "deidentified": "A_DEID", "entities": 1},
        {"status": "OK", "original": "note B", "deidentified": "B_DEID", "entities": 0},
    ]


def test_build_batch_table_marks_failed_items():
    # A note that failed (ok=False) becomes a Failed row with its error in place of the
    # de-identified text, so the bad note stays visible instead of aborting the batch.
    notes = ["good", "bad"]
    results = [
        {"ok": True, "deidentified_text": "G", "entities": []},
        {"ok": False, "error": "boom"},
    ]
    table = build_batch_table(notes, results)
    assert table[0]["status"] == "OK"
    assert table[1] == {
        "status": "Failed",
        "original": "bad",
        "deidentified": "boom",
        "entities": 0,
    }


def test_build_batch_table_truncates_on_length_mismatch():
    notes = ["a", "b", "c"]
    results = [{"deidentified_text": "A", "entities": []}]
    table = build_batch_table(notes, results)
    assert len(table) == 1
    assert table[0]["original"] == "a"


def test_build_base_opts_shift_dates_includes_date_fields():
    opts = build_base_opts(
        method="shift_dates",
        confidence_threshold=0.5,
        lang="en",
        keep_mapping=True,
        consistent=False,
        seed=0,
        date_shift_days=180,
        keep_year=False,
        use_safety_sweep=True,
    )
    assert opts["date_shift_days"] == 180
    assert opts["keep_year"] is False
    assert "seed" not in opts


def test_build_base_opts_shift_dates_omits_zero_days():
    # date_shift_days=0 (the Advanced number_input default) must be omitted so the request falls to
    # openmed's per-note random shift instead of shifting by zero — which, once shift_dates
    # actually runs under openmed 1.6.0, would emit every date verbatim (a PHI leak).
    opts = build_base_opts(
        method="shift_dates",
        confidence_threshold=0.5,
        lang="en",
        keep_mapping=True,
        consistent=False,
        seed=0,
        date_shift_days=0,
        keep_year=True,
        use_safety_sweep=True,
    )
    assert "date_shift_days" not in opts
    assert opts["keep_year"] is True


def test_build_base_opts_replace_includes_nonblank_locale():
    opts = build_base_opts(
        method="replace",
        confidence_threshold=0.5,
        lang="pt",
        keep_mapping=False,
        consistent=False,
        seed=0,
        locale="  pt_BR  ",  # stripped before sending
        date_shift_days=0,
        keep_year=True,
        use_safety_sweep=True,
    )
    assert opts["locale"] == "pt_BR"


def test_build_base_opts_replace_omits_blank_locale():
    # Blank means "use openmed's default for the language" — omit it from the payload.
    for blank in ("", "   ", None):
        opts = build_base_opts(
            method="replace",
            confidence_threshold=0.5,
            lang="en",
            keep_mapping=False,
            consistent=False,
            seed=0,
            locale=blank,
            date_shift_days=0,
            keep_year=True,
            use_safety_sweep=True,
        )
        assert "locale" not in opts


def test_build_base_opts_non_replace_omits_locale():
    # locale is a replace-only knob; other methods must not carry it even if supplied.
    opts = build_base_opts(
        method="mask",
        confidence_threshold=0.5,
        lang="en",
        keep_mapping=False,
        consistent=False,
        seed=0,
        locale="pt_BR",
        date_shift_days=0,
        keep_year=True,
        use_safety_sweep=True,
    )
    assert "locale" not in opts


def test_build_base_opts_carries_use_safety_sweep():
    def _opts(*, use_safety_sweep: bool) -> dict:
        return build_base_opts(
            method="mask",
            confidence_threshold=0.5,
            lang="en",
            keep_mapping=False,
            consistent=False,
            seed=0,
            date_shift_days=0,
            keep_year=True,
            use_safety_sweep=use_safety_sweep,
        )

    assert _opts(use_safety_sweep=True)["use_safety_sweep"] is True
    assert _opts(use_safety_sweep=False)["use_safety_sweep"] is False
