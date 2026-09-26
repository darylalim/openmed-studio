"""Pins the shipped ``.streamlit/config.toml`` keys that are security posture, not style.

Six of its keys protect PHI or the host rather than the look of the app. Four under
``[server]`` decide who may open the app's WebSocket — the channel that runs it — for an
unauthenticated UI: ``address`` keeps it on loopback (Streamlit's own default is every
interface), ``allowedHosts`` refuses any other ``Host`` header (a loopback bind alone still
admits a DNS-rebinding page), and ``enableCORS``/``corsAllowedOrigins`` are pinned at
Streamlit's defaults so no other config can open the socket to a cross-site page. Then
``[client] showErrorDetails`` keeps tracebacks — which can quote note text — out of the
browser, and ``[browser] gatherUsageStats`` keeps a clinical-text tool from phoning home.
Streamlit doesn't fail on a config parse error (``streamlit/config.py`` logs a traceback,
drops the WHOLE file and starts anyway), so one stray typo would quietly revert all six; the
static tests below fail on it instead.

The last two tests are a **Streamlit-drift guard**, the analogue of the openmed-sync guards
in ``test_validation.py``: they drive Streamlit's own config loader and WebSocket check —
private entry points, in a subprocess, because its config is process-global state the
``AppTest`` suite shares — to prove the file still outranks a ``~/.streamlit`` and a
working-directory config holding the opposite values, and that the socket still refuses a
rebinding and a cross-site page. If they fail on a Streamlit bump, re-verify the precedence
and Host-matching claims in CLAUDE.md ("Known gotchas" → *The UI's bind is two coupled
Streamlit settings*) before touching the assertions. No model load, no network, no server.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# Streamlit's own parser (streamlit/config.py), for parity: NOT tomllib, which accepts TOML 1.0
# that toml 0.10 rejects (e.g. `x = [1, "a"]`), and Streamlit drops such a file whole. Left
# undeclared: it arrives only with streamlit, so an ImportError means Streamlit changed parsers.
import toml

# tests/ is not a package (uv non-package project; pythonpath = ["."]), so derive the repo
# root from this file rather than importing a shared constant.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / ".streamlit" / "config.toml"

UI_ADDRESS = "127.0.0.1"  # the API's OPENMED_STUDIO_HOST default too
UI_ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
EXPECTED = {
    "server.address": UI_ADDRESS,
    "server.allowedHosts": UI_ALLOWED_HOSTS,
    "server.enableCORS": True,
    "server.corsAllowedOrigins": [],
    "client.showErrorDetails": "none",
    "browser.gatherUsageStats": False,
}


def _config() -> dict[str, Any]:
    return toml.loads(CONFIG.read_text(encoding="utf-8"))


def test_ui_binds_loopback() -> None:
    assert _config()["server"]["address"] == UI_ADDRESS, (
        "the shipped config must keep the unauthenticated UI on loopback; Streamlit's own "
        "default is every interface (dual-stack '::'), reachable from the network"
    )


def test_ui_accepts_only_loopback_host_headers() -> None:
    server = _config()["server"]
    # Bare hostnames only: Streamlit ignores the Host header's port, so 'localhost:8501'
    # silently never matches, and an IPv6 literal goes in unbracketed ('::1', not '[::1]').
    assert server["allowedHosts"] == UI_ALLOWED_HOSTS
    # Streamlit prints and opens http://<address>:<port>, so the bind address must itself be
    # an allowed Host — otherwise the URL it opens hangs on "Please wait…".
    assert server["address"] in server["allowedHosts"]


def test_ui_refuses_cross_site_websockets() -> None:
    # Streamlit's defaults, pinned: enableCORS = false, or any corsAllowedOrigins entry, lets
    # that origin's pages open the socket — their Host is still 127.0.0.1, so allowedHosts
    # can't help. Set here, a ~/.streamlit or working-directory config can't switch them.
    server = _config()["server"]
    assert server["enableCORS"] is True
    assert server["corsAllowedOrigins"] == []


def test_ui_never_renders_error_details() -> None:
    assert _config()["client"]["showErrorDetails"] == "none"


def test_ui_sends_no_usage_stats() -> None:
    assert _config()["browser"]["gatherUsageStats"] is False


# The opposite of every EXPECTED value, planted as both ~/.streamlit/config.toml and
# $CWD/.streamlit/config.toml. Empty directories would only prove the shipped file is READ; a
# decoy proves it is read LAST. Each decoy also sets one key the shipped file doesn't — a
# control showing that decoy really was loaded.
_DECOY = """
[server]
address = "0.0.0.0"
allowedHosts = ["*"]
enableCORS = false
corsAllowedOrigins = ["http://evil.test"]
[client]
showErrorDetails = "full"
[browser]
gatherUsageStats = true
[runner]
{control} = false
"""
_CONTROLS = {"cwd": "runner.magicEnabled", "home": "runner.postScriptGC"}

# What `streamlit run <script>` does before it serves (streamlit/web/cli.py): point the config
# system at the main script, so the .streamlit/ beside it is read last, then load every
# source. Then ask the WebSocket's own Origin/Host gate (with the machine-IP lookups stubbed,
# so nothing leaves the host) about same-origin local pages, a DNS-rebinding page, and a
# cross-site page aimed at 127.0.0.1.
_PROBE = """
import json, sys
from streamlit import config, net_util
from streamlit.web import bootstrap
from streamlit.web.server.starlette.starlette_websocket import _is_origin_allowed

net_util.get_internal_ip = net_util.get_external_ip = lambda: None
config._main_script_path = sys.argv[1]
bootstrap.load_config_options(flag_options={})
pages = {
    "local": ("http://localhost:8501", "localhost:8501"),
    "loopback": ("http://127.0.0.1:8501", "127.0.0.1:8501"),
    "rebinding": ("http://evil.test:8501", "evil.test:8501"),
    "cross-site": ("http://evil.test", "127.0.0.1:8501"),
}
print(json.dumps({
    "options": {k: [config.get_option(k), config.get_where_defined(k)] for k in sys.argv[2:]},
    "pages": {name: _is_origin_allowed(*headers) for name, headers in pages.items()},
}))
"""


@pytest.fixture(scope="module")
def effective(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Streamlit's effective config, launched with a decoy config in its CWD and its HOME."""
    dirs = {
        where: tmp_path_factory.mktemp(f"decoy-{where}").resolve()
        for where in _CONTROLS
    }
    decoys = {}
    for where, key in _CONTROLS.items():
        decoy = dirs[where] / ".streamlit" / "config.toml"
        decoy.parent.mkdir()
        decoy.write_text(_DECOY.format(control=key.split(".")[1]), encoding="utf-8")
        decoys[key] = str(decoy)
    # STREAMLIT_* vars are dropped as future-proofing: get_config_options' docstring lists them
    # as a source, though 1.64 reads only sensitive ones there. The documented flag/env
    # override goes through click in `streamlit run`, which this probe bypasses.
    env = {k: v for k, v in os.environ.items() if not k.startswith("STREAMLIT_")}
    env["HOME"] = str(dirs["home"])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            str(REPO_ROOT / "streamlit_app.py"),
            *EXPECTED,
            *decoys,
        ],
        cwd=dirs["cwd"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Streamlit logs a parse error and drops the whole file, so the probe still exits 0.
    assert "Error parsing config toml" not in result.stderr, result.stderr
    return {**json.loads(result.stdout.strip().splitlines()[-1]), "decoys": decoys}


def test_shipped_keys_outrank_home_and_working_directory_configs(
    effective: dict[str, Any],
) -> None:
    options = dict(effective["options"])
    # Both decoys were loaded (each control key comes from its own decoy file) ...
    for key, decoy in effective["decoys"].items():
        assert options.pop(key) == [False, decoy]
    # ... yet every pinned value comes from THIS file, which Streamlit reads last.
    assert options == {k: [v, str(CONFIG)] for k, v in EXPECTED.items()}


def test_websocket_refuses_rebinding_and_cross_site_pages(
    effective: dict[str, Any],
) -> None:
    # A DNS-rebinding page sends its own name as Host with a matching Origin, which Streamlit's
    # same-origin rule would accept but allowedHosts refuses; a cross-site page sends
    # Host 127.0.0.1 with a foreign Origin, which enableCORS refuses. The decoys would let both.
    assert effective["pages"] == {
        "local": True,
        "loopback": True,
        "rebinding": False,
        "cross-site": False,
    }
