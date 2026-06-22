"""Streamlit app for openmed-studio's PII/PHI de-identification.

This is the project's only delivery surface. It calls the OpenMed model
**in-process** through :mod:`openmed_studio.service` — a framework-free seam that
validates each request (reusing the Pydantic models in
:mod:`openmed_studio.validation`, so the text/batch/mapping caps still apply),
invokes the shared :class:`~openmed_studio.engine.PIIEngine`, and adapts the
result to plain dicts. There is no HTTP service to run and no API key: this is a
local, single-user tool (see the README's "What we dropped vs the old service").

The pure helpers (HTML rendering, payload building) live in ``ui_helpers`` so they
stay unit-testable; this module is the Streamlit glue. The UI lives in ``main()``
(run under ``__main__``) so importing this module for tests has no side effects.

Run it::

    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import streamlit as st

from openmed_studio import DEFAULT_PII_MODEL, __version__, service
from openmed_studio.engine import PIIEngine
from ui_helpers import (
    build_base_opts,
    build_batch_table,
    render_highlighted,
    render_legend,
    render_plain,
)

# Mirror the engine/schema surface (engine.DeidMethod / validation.Lang).
METHODS = ["mask", "remove", "replace", "hash", "shift_dates"]
LANGS = ["en", "fr", "de", "it", "es", "nl", "hi", "te", "pt"]

EXAMPLE_NOTE = (
    "Patient: John A. Doe (MRN 4827193). DOB 03/12/1972.\n"
    "Seen on 04/15/2024 at Lakeside Clinic by Dr. Emily Carter.\n"
    "Contact: john.doe@example.com, (415) 555-0142. SSN 123-45-6789.\n"
    "Lives at 742 Evergreen Terrace, Springfield. Discharged in stable condition."
)


@st.cache_resource(show_spinner=False)
def get_engine() -> PIIEngine:
    """The process-wide engine, built once (model loads lazily on first call)."""
    return service.build_engine()


def _call(
    fn: Callable[..., dict[str, Any]],
    *args: Any,
    action: str,
    model_loaded: bool,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Run a ``service`` call in a spinner; render ``ServiceError`` and return None.

    The first call loads the model, so warn about the wait until it's resident.
    """
    hint = (
        "" if model_loaded else " — the first request loads the model (up to a minute)"
    )
    with st.spinner(f"{action}…{hint}"):
        try:
            return fn(get_engine(), *args, **kwargs)
        except service.ServiceError as exc:
            st.error(str(exc), icon=":material/error:")
            return None


def _render_single(base_opts: dict[str, Any], model_loaded: bool) -> None:
    with st.form("single"):
        text = st.text_area("Clinical note", value=EXAMPLE_NOTE, height=200)
        submitted = st.form_submit_button(
            "De-identify", type="primary", icon=":material/lock:"
        )
    if submitted and not text.strip():
        st.warning("Enter some text to de-identify.")
        return
    if not submitted:
        return

    result = _call(
        service.deidentify,
        text,
        action="De-identifying",
        model_loaded=model_loaded,
        **base_opts,
    )
    if result is None:
        return

    entities = result["entities"]
    # Update both together so the Re-identify tab never prefills this run's text
    # with a previous run's mapping (a result with keep_mapping off has mapping=None,
    # which must clear any earlier mapping).
    st.session_state.last_deidentified = result["deidentified_text"]
    st.session_state.last_mapping = result.get("mapping") or None

    m1, m2 = st.columns(2)
    m1.metric("Entities found", len(entities))
    m2.metric("Method", result["method"])

    left, right = st.columns(2)
    with left.container(border=True, height="stretch"):
        st.caption("Original — detected PII highlighted")
        st.html(render_highlighted(text, entities))
        legend = render_legend(entities)
        if legend:
            st.html(legend)
    with right.container(border=True, height="stretch"):
        st.caption("De-identified")
        st.html(render_plain(result["deidentified_text"]))
        st.download_button(
            "Download",
            result["deidentified_text"],
            file_name="deidentified.txt",
            icon=":material/download:",
            key="dl_single",
        )

    with st.expander(f"Entities ({len(entities)})", icon=":material/table_chart:"):
        st.dataframe(
            entities,
            hide_index=True,
            column_config={
                "confidence": st.column_config.NumberColumn(format="%.2f"),
                "start": st.column_config.NumberColumn(width="small"),
                "end": st.column_config.NumberColumn(width="small"),
            },
        )
    if result.get("mapping"):
        with st.expander("Mapping — re-identification key", icon=":material/key:"):
            st.caption(
                "As sensitive as raw PHI. Held in this session for the Re-identify tab."
            )
            st.json(result["mapping"])


def _render_batch(base_opts: dict[str, Any], model_loaded: bool) -> None:
    st.caption(
        "Edit the table (one note per row, up to 100), then de-identify all at once."
    )
    rows = st.data_editor(
        [{"note": EXAMPLE_NOTE}, {"note": ""}],
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "note": st.column_config.TextColumn("Clinical note", width="large")
        },
        key="batch_editor",
    )
    if not st.button(
        "De-identify all", type="primary", icon=":material/lock:", key="batch_go"
    ):
        return

    notes = [
        r["note"].strip()
        for r in rows
        if isinstance(r.get("note"), str) and r["note"].strip()
    ]
    if not notes:
        st.warning("Add at least one note.")
        return
    if len(notes) > 100:
        st.warning(f"Max 100 notes per batch (got {len(notes)}).")
        return

    result = _call(
        service.deidentify_batch,
        notes,
        action="De-identifying",
        model_loaded=model_loaded,
        **base_opts,
    )
    if result is None:
        return

    # The engine is called once per note, in order, so results and notes line up 1:1.
    results = result["results"]
    table = build_batch_table(notes, results)
    st.metric("Notes de-identified", len(table))
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "original": st.column_config.TextColumn(width="large"),
            "deidentified": st.column_config.TextColumn(width="large"),
            "entities": st.column_config.NumberColumn(width="small"),
        },
    )
    st.download_button(
        "Download all (JSON)",
        json.dumps(results, indent=2),
        file_name="deidentified_batch.json",
        icon=":material/download:",
        key="dl_batch",
    )


def _render_reidentify(model_loaded: bool) -> None:
    st.caption(
        "Restore original text from a kept mapping (turn on 'Keep mapping' before de-identifying)."
    )
    deid_text = st.text_area(
        "De-identified text", value=st.session_state.last_deidentified, height=150
    )
    mapping_text = st.text_area(
        "Mapping (JSON)",
        value=json.dumps(st.session_state.last_mapping or {}, indent=2),
        height=150,
    )
    if st.button(
        "Re-identify", type="primary", icon=":material/lock_open:", key="reid_go"
    ):
        try:
            mapping = json.loads(mapping_text or "{}")
        except json.JSONDecodeError as exc:
            st.error(f"Mapping is not valid JSON: {exc}")
            mapping = None
        if isinstance(mapping, dict) and mapping:
            result = _call(
                service.reidentify,
                deid_text,
                mapping,
                action="Re-identifying",
                model_loaded=model_loaded,
            )
            if result is not None:
                with st.container(border=True):
                    st.caption("Re-identified")
                    st.html(render_plain(result["text"]))
                st.download_button(
                    "Download",
                    result["text"],
                    file_name="reidentified.txt",
                    icon=":material/download:",
                    key="dl_reid",
                )
        elif mapping is not None:
            st.warning("Provide a non-empty mapping object.")
    st.caption(
        ":material/warning: Overlapping keys (e.g. ALIAS_1 vs ALIAS_10) can mis-restore — "
        "a known openmed limitation."
    )


def _render_detect(base_opts: dict[str, Any], model_loaded: bool) -> None:
    st.caption(
        "Detect PII entities without redacting — audit what the model finds (and misses) "
        "before choosing a redaction method."
    )
    with st.form("detect"):
        text = st.text_area("Clinical note to scan", value=EXAMPLE_NOTE, height=200)
        smart = st.toggle(
            "Smart entity merging",
            value=True,
            help="Recombine token-fragmented PII (dates, SSNs) into whole spans.",
        )
        submitted = st.form_submit_button(
            "Detect", type="primary", icon=":material/search:"
        )
    if submitted and not text.strip():
        st.warning("Enter some text to scan.")
        return
    if not submitted:
        return

    result = _call(
        service.extract,
        text,
        action="Detecting",
        model_loaded=model_loaded,
        confidence_threshold=base_opts["confidence_threshold"],
        use_smart_merging=smart,
        lang=base_opts["lang"],
    )
    if result is None:
        return

    entities = result["entities"]
    st.metric("Entities found", len(entities))
    with st.container(border=True):
        st.caption("Detected PII")
        st.html(render_highlighted(text, entities))
        legend = render_legend(entities)
        if legend:
            st.html(legend)
    st.dataframe(
        entities,
        hide_index=True,
        column_config={
            "confidence": st.column_config.NumberColumn(format="%.2f"),
            "start": st.column_config.NumberColumn(width="small"),
            "end": st.column_config.NumberColumn(width="small"),
        },
    )


def _render_sidebar() -> tuple[dict[str, Any], str, bool]:
    """Draw the sidebar; return (base_opts, method, model_loaded)."""
    with st.sidebar:
        st.subheader("Engine")
        engine = get_engine()
        model_loaded = engine.is_loaded
        st.caption(f"Model: {engine.model_name or DEFAULT_PII_MODEL}")
        st.caption(
            f"Backend: {engine.backend or 'auto'} · v{__version__} · "
            + ("model loaded" if model_loaded else "loads on first request")
        )

        st.subheader("De-identification")
        method = st.segmented_control("Method", METHODS, default="mask") or "mask"
        lang = st.selectbox("Language", LANGS, index=0)
        confidence = st.slider(
            "Confidence threshold",
            0.0,
            1.0,
            0.5,
            0.05,
            help="Minimum model confidence to keep an entity. The UI defaults to 0.5 for "
            "higher PHI recall; the de-identify default is 0.7.",
        )
        keep_mapping = st.toggle(
            "Keep mapping",
            value=True,
            help="Return the surrogate→original map; enables the Re-identify tab.",
        )
        with st.expander("Advanced", icon=":material/tune:"):
            consistent = st.toggle(
                "Deterministic replace",
                help="With method=replace, the same input maps to the same surrogate.",
            )
            seed = st.number_input(
                "Seed", value=0, step=1, help="Used with deterministic replace."
            )
            date_shift_days = st.number_input(
                "Date shift days",
                value=0,
                step=1,
                help="Only used by method=shift_dates.",
            )
            keep_year = st.toggle(
                "Keep year", value=True, help="Used by method=shift_dates."
            )

    base_opts = build_base_opts(
        method=method,
        confidence_threshold=confidence,
        lang=lang,
        keep_mapping=keep_mapping,
        consistent=consistent,
        seed=int(seed),
        date_shift_days=int(date_shift_days),
        keep_year=keep_year,
    )
    return base_opts, method, model_loaded


def main() -> None:
    st.set_page_config(
        page_title="OpenMed Studio — de-identification",
        page_icon=":material/health_and_safety:",
        layout="wide",
    )
    st.session_state.setdefault("last_mapping", None)
    st.session_state.setdefault("last_deidentified", "")

    base_opts, method, model_loaded = _render_sidebar()

    st.title("PII / PHI de-identification")
    st.caption(
        "Detect or de-identify clinical text with OpenMed, review the entities, and "
        "round-trip with re-identification. The model runs in-process; pick the method "
        "in the sidebar."
    )
    if method == "shift_dates":
        st.caption(
            ":material/info: With the default model, `shift_dates` masks dates rather than "
            "shifting them (a known openmed behavior)."
        )

    tab_detect, tab_single, tab_batch, tab_reid = st.tabs(
        [
            ":material/search: Detect",
            ":material/description: Single note",
            ":material/stacks: Batch",
            ":material/lock_open: Re-identify",
        ]
    )
    with tab_detect:
        _render_detect(base_opts, model_loaded)
    with tab_single:
        _render_single(base_opts, model_loaded)
    with tab_batch:
        _render_batch(base_opts, model_loaded)
    with tab_reid:
        _render_reidentify(model_loaded)


if __name__ == "__main__":
    main()
