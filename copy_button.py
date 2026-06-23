"""A self-contained Custom Component v2 copy-to-clipboard button (Streamlit >= 1.58).

Streamlit ships no native "copy" button, so this fills that gap with a tiny inline
CCv2 element (``st.components.v2.component`` — never the deprecated v1 API). It is
theme-aware through the injected ``--st-*`` CSS variables and renders inside a
shadow root (``isolate_styles=True``, the default), so it neither leaks styles into
the app nor inherits the app's.

PHI note: the text to copy is sent to the browser once via ``data`` and copied
entirely client-side. The component declares no state and no trigger, so nothing —
least of all the (potentially sensitive) text — is ever emitted back to Python.

The component cannot be exercised by ``streamlit.testing.v1.AppTest`` (it runs
headless, with no browser to execute the JS); the tests only assert that mounting
it raises nothing and does not break sibling rendering.
"""

from __future__ import annotations

from streamlit.components.v2 import component

_HTML = """
<button id="copy-btn" type="button" aria-live="polite">
  <span class="label">Copy</span>
</button>
"""

_CSS = """
#copy-btn {
  display: inline-flex;
  align-items: center;
  gap: .4em;
  font-family: var(--st-font);
  font-size: var(--st-base-font-size);
  color: var(--st-text-color);
  background: var(--st-secondary-background-color);
  border: 1px solid var(--st-border-color);
  border-radius: var(--st-button-radius);
  padding: .35em .8em;
  cursor: pointer;
  line-height: 1.2;
}
#copy-btn:hover { border-color: var(--st-primary-color); }
#copy-btn:disabled { opacity: .5; cursor: default; }
#copy-btn.copied {
  color: var(--st-green-text-color);
  background: var(--st-green-background-color);
  border-color: var(--st-green-color);
}
"""

_JS = """
export default function (component) {
  const { data, parentElement } = component
  const btn = parentElement.querySelector("#copy-btn")
  const label = parentElement.querySelector(".label")
  if (!btn || !label) return

  // Hydrate from data on every run (Python -> JS). Held on the element only, never
  // echoed back to Python, so the text stays client-side.
  const text = (data && typeof data.text === "string") ? data.text : ""
  const idle = (data && data.label) || "Copy"
  const done = (data && data.copied_label) || "Copied"
  btn.dataset.text = text
  btn.disabled = text.length === 0
  if (!btn.classList.contains("copied")) label.textContent = idle

  let timer = null
  const reset = () => {
    btn.classList.remove("copied")
    label.textContent = idle
  }
  // Show a transient status (success or failure) that auto-resets to idle.
  const showFor = (msg, ok) => {
    btn.classList.toggle("copied", ok)
    label.textContent = msg
    if (timer) clearTimeout(timer)
    timer = setTimeout(reset, 1500)
  }

  // Synchronous fallback for non-secure contexts (e.g. http://0.0.0.0) or when the
  // async Clipboard API rejects. Returns whether the copy actually succeeded.
  const legacyCopy = (value) => {
    const ta = document.createElement("textarea")
    ta.value = value
    ta.style.position = "fixed"
    ta.style.opacity = "0"
    parentElement.appendChild(ta)
    ta.select()
    let ok = false
    try {
      ok = document.execCommand("copy")
    } catch (err) {
      ok = false
    }
    ta.remove()
    return ok
  }

  btn.onclick = async () => {
    const value = btn.dataset.text || ""
    if (!value) return
    let ok = false
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(value)
        ok = true
      }
    } catch (err) {
      ok = false  // permission denied / not focused / transient — try the fallback
    }
    if (!ok) ok = legacyCopy(value)
    showFor(ok ? done : "Copy failed", ok)
  }

  return () => { if (timer) clearTimeout(timer) }
}
"""

_NAME = (
    "openmed_studio.copy_button"  # namespaced to avoid clashing with other components
)


def copy_button(
    text: str,
    *,
    key: str,
    label: str = "Copy",
    copied_label: str = "Copied",
) -> None:
    """Render a theme-aware copy-to-clipboard button for ``text``.

    ``text`` is sent to the browser and copied client-side; it is never emitted
    back to Python. ``key`` must be unique within the page.
    """
    # The component is declared here (not once at import) because this module is
    # imported a single time, but a CCv2 component registers into the *current*
    # Streamlit runtime, and the runtime is recreated per session/script run
    # (notably under AppTest). Re-declaring an identical definition is idempotent:
    # Streamlit silently overwrites a same-named, same-content component (it only
    # warns when the definition differs), so this neither warns nor leaks.
    renderer = component(_NAME, html=_HTML, css=_CSS, js=_JS)
    renderer(
        key=key,
        data={"text": text, "label": label, "copied_label": copied_label},
    )
