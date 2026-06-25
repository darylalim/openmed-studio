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
from typing import Any, get_args

import streamlit as st

from openmed_studio import DEFAULT_PII_MODEL, NER_MODELS, __version__, service
from openmed_studio.engine import DeidMethod, PIIEngine
from openmed_studio.validation import MAX_BATCH_ITEMS, Lang
from ui_helpers import (
    build_base_opts,
    build_batch_table,
    render_highlighted,
    render_legend,
    render_plain,
)

# Derived from the canonical Literals so the sidebar can't drift from the engine /
# validation surface (a new method or language reaches the widgets automatically).
METHODS = list(get_args(DeidMethod))
LANGS = list(get_args(Lang))

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


def _entity_columns() -> dict[str, Any]:
    """Shared ``st.dataframe`` column config for the entity tables (Detect/NER/Single).

    Confidence renders as a 0–1 progress bar so a reviewer can scan model certainty at a
    glance rather than reading decimals.
    """
    return {
        "confidence": st.column_config.ProgressColumn(
            "confidence", min_value=0.0, max_value=1.0, format="%.2f"
        ),
        "start": st.column_config.NumberColumn(width="small"),
        "end": st.column_config.NumberColumn(width="small"),
    }


def _call(
    fn: Callable[..., dict[str, Any]],
    *args: Any,
    action: str,
    needs_load: bool | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Run a ``service`` call in a spinner; render ``ServiceError`` and return None.

    The first call loads the model, so warn about the wait until it's resident. By default
    that's gauged from the shared engine's ``is_loaded``; callers that load several models
    (the NER tab, one per domain) pass ``needs_load`` explicitly, since ``is_loaded`` only
    tracks whether *a* model has loaded, not which one.
    """
    engine = get_engine()
    pending = (not engine.is_loaded) if needs_load is None else needs_load
    hint = " — the first request loads the model (up to a minute)" if pending else ""
    with st.spinner(f"{action}…{hint}"):
        try:
            return fn(engine, *args, **kwargs)
        except service.ServiceError as exc:
            st.error(str(exc), icon=":material/error:")
            return None


def _render_highlight(text: str, entities: list[dict[str, Any]]) -> None:
    """Render highlighted ``text`` plus its color legend (shared by Detect/Single).

    The marks are theme-agnostic (translucent tint + ``color: inherit``), so this
    needs no theme detection.
    """
    st.html(render_highlighted(text, entities))
    legend = render_legend(entities)
    if legend:
        st.html(legend)


def _toast_downloaded() -> None:
    """Lightweight confirmation when a Download button is clicked (an on_click callback)."""
    st.toast("Downloaded", icon=":material/download:")


def _set_handoff(result: dict[str, Any]) -> None:
    """Hand the de-identified text + mapping to the Re-identify tab.

    Set together so a prior run's mapping never lingers against this run's text (a result
    with ``keep_mapping`` off has ``mapping=None``, which must clear any earlier mapping).
    This is the single, security-relevant copy of that invariant; callers invoke it only on
    an actual submit (never on the panel's persistent re-render), so the handoff always
    reflects the most recent de-identification rather than whichever tab rendered last.
    """
    st.session_state.last_deidentified = result["deidentified_text"]
    st.session_state.last_mapping = result.get("mapping") or None


@st.dialog("Re-identification key")
def _show_mapping_dialog(mapping: dict[str, str]) -> None:
    """Reveal the surrogate→original mapping in a modal (a deliberate 'show the key' action).

    Gating it behind a button + dialog (rather than an always-open expander) makes exposing
    the key — which reverses the de-identification and is as sensitive as raw PHI — explicit.
    """
    st.caption(
        "As sensitive as raw PHI — it reverses the de-identification. Held in this session "
        "for the Re-identify tab."
    )
    st.json(mapping)


def _render_deid_result(
    text: str,
    result: dict[str, Any],
    *,
    out_caption: str,
    out_filename: str,
    dl_key: str,
    export_caveat: str | None = None,
    show_entities: bool = False,
) -> None:
    """Shared result panel for the Single note + Anonymize tabs.

    Renders the original-vs-output columns + Download (with an optional ``export_caveat``
    beside the button), optionally the entity table, and a button that reveals the
    re-identification mapping in a dialog. Pure rendering — callers persist the result and
    call :func:`_set_handoff` on submit, so it's safe to re-run on post-submit reruns (a
    Download or a Show-key click) without the panel blanking or the handoff drifting.
    Callers render their own metric row first, since the labels/values differ per tab.
    """
    entities = result["entities"]
    left, right = st.columns(2)
    with left.container(border=True, height="stretch"):
        st.caption("Original — detected PII highlighted")
        _render_highlight(text, entities)
    with right.container(border=True, height="stretch"):
        st.caption(out_caption)
        st.html(render_plain(result["deidentified_text"]))
        st.download_button(
            "Download",
            result["deidentified_text"],
            file_name=out_filename,
            icon=":material/download:",
            key=dl_key,
            on_click=_toast_downloaded,
        )
        if export_caveat:
            st.caption(export_caveat)

    if show_entities:
        with st.expander(f"Entities ({len(entities)})", icon=":material/table_chart:"):
            st.dataframe(entities, hide_index=True, column_config=_entity_columns())
    if result.get("mapping") and st.button(
        "Show re-identification key", icon=":material/key:", key=f"{dl_key}_showkey"
    ):
        _show_mapping_dialog(result["mapping"])


def _render_deid_controls(*, key_prefix: str, lang: str) -> dict[str, Any]:
    """Render the de-identification controls for a tab and return the request options.

    Shared by the Single note + Batch tabs so the method and its dependent knobs live in
    the tab that performs the de-identification — not the sidebar, which now holds only the
    engine readout and the global ``lang`` filter. Rendered ABOVE the tab's form (like the
    NER domain picker) so changing the method reruns and re-renders only the Advanced knobs
    that method actually consumes (``replace`` → consistent/seed/locale; ``shift_dates`` →
    date_shift_days/keep_year). Every widget key is ``key_prefix``-scoped so Single and
    Batch keep independent state without colliding.
    """
    method = (
        st.segmented_control(
            "Method", METHODS, default="mask", key=f"{key_prefix}_method"
        )
        or "mask"
    )
    c1, c2 = st.columns([3, 2])
    confidence = c1.slider(
        "Confidence threshold",
        0.0,
        1.0,
        0.5,
        0.05,
        key=f"{key_prefix}_conf",
        help="Minimum model confidence to keep an entity. The UI defaults to 0.5 for "
        "higher PHI recall; the de-identify default is 0.7.",
    )
    keep_mapping = c2.toggle(
        "Keep mapping",
        value=True,
        key=f"{key_prefix}_keepmap",
        help="Return the surrogate→original map; enables the Re-identify tab.",
    )

    # Defaults for the method-specific knobs; only the chosen method's are rendered, and
    # build_base_opts omits whichever the selected method doesn't consume.
    consistent = False
    seed = 0
    locale = ""
    date_shift_days = 0
    keep_year = True
    with st.expander("Advanced", icon=":material/tune:"):
        if method == "replace":
            consistent = st.toggle(
                "Deterministic replace",
                key=f"{key_prefix}_consistent",
                help="The same input maps to the same surrogate.",
            )
            if consistent:
                seed = st.number_input(
                    "Seed",
                    value=0,
                    step=1,
                    key=f"{key_prefix}_seed",
                    help="Reproducible surrogates across runs.",
                )
            locale = st.text_input(
                "Replace locale",
                value="",
                placeholder="e.g. en_US, pt_BR",
                key=f"{key_prefix}_locale",
                help="Faker locale for replace surrogates (e.g. pt_BR for Brazilian-format "
                "IDs). Blank uses the default for the selected language.",
            )
        elif method == "shift_dates":
            date_shift_days = st.number_input(
                "Date shift days",
                value=0,
                step=1,
                key=f"{key_prefix}_shift",
                help="0 = random per-note shift.",
            )
            keep_year = st.toggle(
                "Keep year",
                value=True,
                key=f"{key_prefix}_keepyear",
                help="Preserve the year when shifting dates.",
            )
        use_safety_sweep = st.toggle(
            "Safety sweep",
            value=True,
            key=f"{key_prefix}_sweep",
            help="Run a deterministic structured-identifier sweep after model detection. "
            "Recommended on — it catches IDs (SSNs, phones) the model misses; turning it "
            "off lowers PHI recall.",
        )

    return build_base_opts(
        method=method,
        confidence_threshold=confidence,
        lang=lang,
        keep_mapping=keep_mapping,
        consistent=consistent,
        seed=int(seed),
        locale=locale,
        date_shift_days=int(date_shift_days),
        keep_year=keep_year,
        use_safety_sweep=use_safety_sweep,
    )


def _render_single(lang: str) -> None:
    base_opts = _render_deid_controls(key_prefix="single", lang=lang)
    with st.form("single"):
        text = st.text_area("Clinical note", value=EXAMPLE_NOTE, height=200)
        submitted = st.form_submit_button(
            "De-identify", type="primary", icon=":material/lock:"
        )
    if submitted:
        if not text.strip():
            st.warning("Enter some text to de-identify.")
            return
        result = _call(service.deidentify, text, action="De-identifying", **base_opts)
        if result is None:
            return
        # Persist so the panel survives post-submit reruns (Download / Show key clicks),
        # and hand off to Re-identify here (only on submit) so the handoff can't drift.
        st.session_state["single_result"] = {"text": text, "result": result}
        _set_handoff(result)

    stored = st.session_state.get("single_result")
    if not stored:
        return
    text, result = stored["text"], stored["result"]
    entities = result["entities"]
    m1, m2 = st.columns(2)
    m1.metric("Entities found", len(entities), border=True)
    m2.metric("Method", result["method"], border=True)
    _render_deid_result(
        text,
        result,
        out_caption="De-identified",
        out_filename="deidentified.txt",
        dl_key="dl_single",
        show_entities=True,
    )


@st.fragment
def _render_batch(lang: str) -> None:
    base_opts = _render_deid_controls(key_prefix="batch", lang=lang)
    st.caption(
        f"Edit the table (one note per row, up to {MAX_BATCH_ITEMS}), then "
        "de-identify all at once."
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
    if len(notes) > MAX_BATCH_ITEMS:
        st.warning(f"Max {MAX_BATCH_ITEMS} notes per batch (got {len(notes)}).")
        return

    result = _call(
        service.deidentify_batch, notes, action="De-identifying", **base_opts
    )
    if result is None:
        return

    # The engine is called once per note, in order, so results and notes line up 1:1.
    # Each result carries ok/error: a bad note is isolated (shown as a Failed row) instead
    # of aborting the whole batch.
    results = result["results"]
    table = build_batch_table(notes, results)
    n_ok = sum(1 for r in results if r.get("ok", True))
    n_failed = len(results) - n_ok
    st.metric("Notes de-identified", n_ok, border=True)
    if n_failed:
        st.warning(
            f"{n_failed} of {len(results)} note(s) failed — see the Status column.",
            icon=":material/error:",
        )
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "status": st.column_config.TextColumn(width="small"),
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
        on_click=_toast_downloaded,
    )


def _render_anonymize(lang: str) -> None:
    """Replace detected PII/PHI with realistic fake surrogates (surrogate replacement).

    A focused, surrogate-first view over ``service.deidentify(method="replace")``: the
    capability already exists (it's the sidebar's ``replace`` method), this just surfaces it
    as its own workflow with the determinism/locale knobs in-tab. Like ``_render_single`` it
    is intentionally **not** an ``@st.fragment`` — its form submit must trigger a full rerun
    so the Re-identify fragment re-reads the ``last_deidentified``/``last_mapping`` handed off
    here (``replace`` + ``keep_mapping`` is the canonical reversible-pseudonymization round trip).
    """
    st.caption(
        "Replace *detected* PII/PHI with realistic *fake* surrogates rather than redacting. Like "
        "all model-based de-identification, anything the model misses is left in place — review "
        "before sharing. Repeated mentions stay one identity with 'Deterministic'; the mapping "
        "round-trips through the Re-identify tab."
    )
    with st.form("anonymize"):
        text = st.text_area(
            "Clinical note to anonymize",
            value=EXAMPLE_NOTE,
            height=200,
            key="anon_text",
        )
        c1, c2 = st.columns(2)
        confidence = c1.slider(
            "Confidence threshold",
            0.0,
            1.0,
            0.5,
            0.05,
            key="anon_conf",
            help="Lower keeps more entities (higher PHI recall = more replaced).",
        )
        consistent = c2.toggle(
            "Deterministic",
            value=True,
            key="anon_consistent",
            help="Same input → same surrogate, so repeated mentions resolve to one identity.",
        )
        seed = c1.number_input(
            "Seed",
            value=42,
            step=1,
            key="anon_seed",
            help="Reproducible surrogates across runs (used when Deterministic is on).",
        )
        locale = c2.text_input(
            "Locale",
            value="",
            placeholder="e.g. en_US, pt_BR",
            key="anon_locale",
            help="Faker locale for surrogates (e.g. pt_BR). Blank derives it from the language.",
        )
        submitted = st.form_submit_button(
            "Anonymize", type="primary", icon=":material/masks:"
        )
    if submitted:
        if not text.strip():
            st.warning("Enter some text to anonymize.")
            return
        # Pin method=replace; only add seed when deterministic and locale when set (mirrors
        # build_base_opts, so openmed gets its per-call random surrogates / lang-derived
        # locale rather than a zero seed or empty locale).
        opts: dict[str, Any] = {
            "method": "replace",
            "confidence_threshold": confidence,
            "lang": lang,
            "consistent": consistent,
            "keep_mapping": True,
        }
        if consistent:
            opts["seed"] = int(seed)
        if locale.strip():
            opts["locale"] = locale.strip()
        result = _call(service.deidentify, text, action="Anonymizing", **opts)
        if result is None:
            return
        # Persist (with the determinism flag for the metric) and hand off on submit only,
        # so the panel survives post-submit reruns (Download / Show key).
        st.session_state["anon_result"] = {
            "text": text,
            "result": result,
            "consistent": consistent,
        }
        _set_handoff(result)

    stored = st.session_state.get("anon_result")
    if not stored:
        return
    text, result, consistent = stored["text"], stored["result"], stored["consistent"]
    entities = result["entities"]
    m1, m2 = st.columns(2)
    m1.metric("Entities replaced", len(entities), border=True)
    m2.metric("Deterministic", "On" if consistent else "Off", border=True)
    _render_deid_result(
        text,
        result,
        out_caption=(
            "Anonymized — synthetic surrogates · review for residual identifiers before sharing"
        ),
        out_filename="anonymized.txt",
        dl_key="dl_anon",
        export_caveat="May still contain any PII the model missed — review before sharing.",
    )


@st.fragment
def _render_reidentify() -> None:
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
                service.reidentify, deid_text, mapping, action="Re-identifying"
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
                    on_click=_toast_downloaded,
                )
                st.toast("Re-identified", icon=":material/lock_open:")
        elif mapping is not None:
            st.warning("Provide a non-empty mapping object.")


@st.fragment
def _render_detect(lang: str) -> None:
    st.caption(
        "Detect PII entities without redacting — audit what the model finds. De-identification "
        "may redact more than is shown here: it keeps smart merging on and runs a deterministic "
        "structured-identifier safety sweep (toggleable per de-identification tab) that catches "
        "IDs the model misses."
    )
    with st.form("detect"):
        text = st.text_area("Clinical note to scan", value=EXAMPLE_NOTE, height=200)
        c1, c2 = st.columns([3, 2])
        confidence = c1.slider(
            "Confidence threshold",
            0.0,
            1.0,
            0.5,
            0.05,
            key="detect_conf",
            help="Minimum model confidence to keep an entity (UI default 0.5 for recall).",
        )
        smart = c2.toggle(
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
        confidence_threshold=confidence,
        use_smart_merging=smart,
        lang=lang,
    )
    if result is None:
        return

    entities = result["entities"]
    st.metric("Entities found", len(entities), border=True)
    with st.container(border=True):
        st.caption("Detected PII")
        _render_highlight(text, entities)
    st.dataframe(entities, hide_index=True, column_config=_entity_columns())


@st.fragment
def _render_ner() -> None:
    st.caption(
        "Detect clinical entities (diseases, drugs, anatomy, genes, …) with an OpenMed "
        "NER model — distinct from the PII models the other tabs use. Each domain loads a "
        "specialized model on first use; switching domains loads another."
    )
    # The domain picker and its preview live OUTSIDE the form, so choosing a domain reruns
    # the fragment and refreshes the entity preview and the slider's recommended default.
    domain = st.selectbox("Entity domain", list(NER_MODELS), key="ner_domain")
    model = NER_MODELS[domain]
    detects = ", ".join(model.entity_types) if model.entity_types else "not declared"
    st.caption(
        f"**{model.display_name}** · {model.params} · detects: {detects}"
        + (
            "  ·  broad-coverage model (~3× the others)"
            if not model.entity_types
            else ""
        )
    )
    with st.form("ner"):
        text = st.text_area(
            "Clinical note to analyze", value=EXAMPLE_NOTE, height=200, key="ner_text"
        )
        confidence = st.slider(
            "Confidence threshold",
            0.0,
            1.0,
            model.recommended_confidence,
            0.05,
            # Per-domain key so each domain's slider defaults to its own recommendation
            # and remembers a manual override independently.
            key=f"ner_conf_{domain}",
            help="Minimum model confidence to keep an entity "
            "(default = this model's recommended threshold).",
        )
        submitted = st.form_submit_button(
            "Analyze", type="primary", icon=":material/biotech:"
        )
    if submitted and not text.strip():
        st.warning("Enter some text to analyze.")
        return
    if not submitted:
        return

    # is_loaded flips True after the FIRST model loads, so it can't tell whether THIS
    # domain's model is resident. Track analyzed domains so the wait hint fires on a fresh
    # domain (a 141–434MB download) rather than only the very first NER call.
    analyzed: set[str] = st.session_state.setdefault("ner_analyzed_domains", set())
    result = _call(
        service.analyze,
        text,
        action="Analyzing",
        needs_load=domain not in analyzed,
        model_name=model.alias,
        confidence_threshold=confidence,
    )
    if result is None:
        return
    analyzed.add(domain)

    entities = result["entities"]
    st.metric("Entities found", len(entities), border=True)
    st.caption(f"Model: {model.display_name} (`{model.alias}`)")
    with st.container(border=True):
        st.caption(f"Detected {domain.lower()} entities")
        _render_highlight(text, entities)
    st.dataframe(entities, hide_index=True, column_config=_entity_columns())


def _render_sidebar() -> str:
    """Draw the sidebar (engine status + the global Language filter); return the language.

    Per Streamlit layout guidance the sidebar holds only app-level state: the engine
    readout and the one cross-cutting filter — ``lang``, which selects the per-language
    detection model/locale for the Detect, Single note, Batch, and Anonymize tabs. The
    de-identification *method* and its dependent knobs live in the tabs that perform
    de-identification (Single note + Batch, via :func:`_render_deid_controls`), so tabs
    that don't de-identify (Clinical NER, Re-identify) show no stray controls.
    """
    with st.sidebar:
        st.subheader("Engine")
        engine = get_engine()
        st.caption(
            f"Model: {engine.model_name or DEFAULT_PII_MODEL} (English) — non-English "
            "languages auto-load a language-specific model on first use."
        )
        st.caption(
            f"Backend: {engine.backend or 'auto'} · v{__version__} · "
            + ("model loaded" if engine.is_loaded else "loads on first request")
        )
        lang = st.selectbox(
            "Language",
            LANGS,
            index=0,
            help="Detection language for the Detect, Single note, Batch, and Anonymize "
            "tabs (selects the per-language model/locale). Clinical NER and Re-identify "
            "don't use it.",
        )
    return lang


def main() -> None:
    st.set_page_config(
        page_title="OpenMed Studio — de-identification",
        page_icon=":material/health_and_safety:",
        layout="wide",
    )
    st.session_state.setdefault("last_mapping", None)
    st.session_state.setdefault("last_deidentified", "")

    lang = _render_sidebar()

    st.title("PII / PHI de-identification")
    st.caption(
        "Detect PII, run clinical NER, or de-identify clinical text with OpenMed, review "
        "the entities, and round-trip with re-identification. The model runs in-process; "
        "pick the de-identification method in the Single note and Batch tabs."
    )

    tab_detect, tab_ner, tab_single, tab_batch, tab_anon, tab_reid = st.tabs(
        [
            ":material/search: Detect",
            ":material/biotech: Clinical NER",
            ":material/description: Single note",
            ":material/stacks: Batch",
            ":material/masks: Anonymize",
            ":material/lock_open: Re-identify",
        ]
    )
    with tab_detect:
        _render_detect(lang)
    with tab_ner:
        _render_ner()
    with tab_single:
        _render_single(lang)
    with tab_batch:
        _render_batch(lang)
    with tab_anon:
        _render_anonymize(lang)
    with tab_reid:
        _render_reidentify()


if __name__ == "__main__":
    main()
