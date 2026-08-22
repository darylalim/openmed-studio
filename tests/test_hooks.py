"""Tests for the repo's Claude Code hooks — the PHI-path guard in particular.

``.claude/hooks/block-phi-paths.sh`` is a PreToolUse hook that denies reads and writes of
the gitignored files which can carry PHI (the app's ``Download`` outputs) or local secrets.
Its own header calls the ``case`` arms "the single source of truth", but nothing enforced
that: ``policy_anonymized.txt`` shipped with the ``Policy de-ID`` tab, was gitignored the
same day, and stayed readable through the plain ``Read`` tool until the arms were re-synced
a month later. These are the guards that would have caught it — the repo-tooling analogue of
the openmed-sync guards in ``test_validation.py``.

They **execute** the hook rather than parse it, so what is pinned is behavior (exit 2 denies,
exit 0 allows) rather than the text of a shell script. ``streamlit_app.py``'s ``Download``
filenames are the source of truth for what must be blocked, so adding a de-identifying tab
fails this file until the hook is updated. No model load, no network.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# tests/ is not a package (uv non-package project; pythonpath = ["."]), so derive the repo
# root from this file rather than importing a shared constant.
REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / ".claude" / "hooks" / "block-phi-paths.sh"

# The local-secret basenames the hook guards. Unlike the Download outputs there is no code
# literal to derive these from, so they are declared here and cross-checked against
# .gitignore below — adding an arm to the hook without a matching entry here fails the
# dead-arm test, which is the point: a new deny is a deliberate decision, not a drive-by.
SECRET_ARMS = frozenset({"secrets.toml", ".env", ".env.*"})

# Concrete paths standing in for the SECRET_ARMS globs, since the hook matches basenames.
SECRET_PROBES = (".streamlit/secrets.toml", ".env", ".env.local", ".env.production")

pytestmark = pytest.mark.skipif(
    shutil.which("sh") is None or shutil.which("python3") is None,
    reason="the hook is a POSIX sh script that shells out to python3",
)


def _download_filenames() -> set[str]:
    """Every literal filename streamlit_app.py hands to a ``Download`` button."""
    src = (REPO_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    # Quoted literals only: `file_name=out_filename` is the shared render helper forwarding
    # a caller's value, not an output name of its own.
    return set(re.findall(r'(?:file_name|out_filename)\s*=\s*"([^"]+)"', src))


def _gitignore_patterns() -> set[str]:
    """The .gitignore patterns, comments and blanks stripped, leading `/` normalized off."""
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return {
        stripped.lstrip("/")
        for line in lines
        if (stripped := line.strip()) and not stripped.startswith("#")
    }


def _case_arm_tokens() -> set[str]:
    """The names the hook's ``case`` arms deny, flattened out of the `a|b|c)` alternations."""
    src = HOOK.read_text(encoding="utf-8")
    arms = re.findall(r"^\s{2}([\w.|*-]+)\)$", src, re.MULTILINE)
    assert arms, f"found no case arms in {HOOK.name} — has its shape changed?"
    return {token for arm in arms for token in arm.split("|")}


def _run_hook(payload: str) -> subprocess.CompletedProcess[str]:
    """Invoke the hook the way settings.json does, feeding it a tool call on stdin."""
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["sh", str(HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _denies(file_path: str) -> bool:
    """True when the hook refuses a tool call touching ``file_path`` (exit 2 = deny)."""
    payload = json.dumps({"tool_input": {"file_path": f"/tmp/{file_path}"}})
    return _run_hook(payload).returncode == 2


def test_phi_hook_blocks_every_app_download_output() -> None:
    # THE guard this file exists for. Every filename the app hands to a Download button can
    # land on disk holding PHI or its surrogates, so the hook must deny it. streamlit_app.py
    # is the source of truth, so a new de-identifying tab fails here until the arm is synced.
    outputs = _download_filenames()
    assert outputs, (
        "no Download filenames found in streamlit_app.py — has the regex drifted?"
    )
    unguarded = sorted(name for name in outputs if not _denies(name))
    assert not unguarded, (
        f"{HOOK.name} does not block {unguarded}. These are Download outputs that can carry "
        "PHI or its surrogates; add them to the download-output case arm."
    )


def test_app_download_outputs_are_gitignored() -> None:
    # The other half of the same invariant: an output the hook blocks but git tracks would
    # still reach a public repo. Pins the intent of accf1dd ("ignore the Policy de-ID Download
    # output, which can carry PHI surrogates") against a future tab that forgets .gitignore.
    tracked = sorted(_download_filenames() - _gitignore_patterns())
    assert not tracked, f".gitignore is missing the Download outputs {tracked}"


def test_phi_hook_blocks_local_secrets() -> None:
    # Secrets are the hook's second job. OPENMED_STUDIO_API_KEY (the FastAPI service's auth
    # key) would live in a local .env, and .streamlit/secrets.toml holds Streamlit's — both
    # gitignored, neither safe to pull into a context window.
    unguarded = sorted(path for path in SECRET_PROBES if not _denies(path))
    assert not unguarded, f"{HOOK.name} does not block the local secrets {unguarded}"


def test_secret_arms_are_gitignored() -> None:
    # Keeps SECRET_ARMS honest: every basename the hook denies as a "secret" must correspond
    # to something .gitignore actually treats as one, so the two lists can't drift apart.
    ignored_basenames = {
        pattern.rsplit("/", 1)[-1] for pattern in _gitignore_patterns()
    }
    orphaned = sorted(SECRET_ARMS - ignored_basenames)
    assert not orphaned, f"the hook guards {orphaned}, which .gitignore does not ignore"


def test_phi_hook_case_arms_have_no_dead_entries() -> None:
    # The reverse direction: nothing is denied that isn't a current Download output or a
    # declared secret. Catches an arm left behind when a tab is removed or renamed.
    expected = _download_filenames() | set(SECRET_ARMS)
    assert _case_arm_tokens() == expected


def test_phi_hook_allows_ordinary_source_files() -> None:
    # A hook that denied everything would satisfy every test above while making the repo
    # unworkable, so pin the allow path too.
    for name in ("streamlit_app.py", "README.md", "pyproject.toml", "CLAUDE.md"):
        assert not _denies(name), f"{HOOK.name} wrongly blocks {name}"


def test_phi_hook_allows_tool_calls_carrying_no_file_path() -> None:
    # A parsed payload with no .tool_input.file_path is a non-file tool (Bash and friends),
    # which this hook has no opinion on. Distinct from the unparseable case below.
    result = _run_hook(
        json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}})
    )
    assert result.returncode == 0


@pytest.mark.parametrize(
    ("payload", "label"),
    [("not json at all", "malformed JSON"), ("", "empty stdin")],
)
def test_phi_hook_fails_closed_on_unparseable_input(payload: str, label: str) -> None:
    # Deliberate posture, easy to "simplify" back into a silent fail-open: a guard that
    # cannot see what it is guarding must deny, not wave the call through. Before this was
    # fixed, both of these returned 0 and the PHI arms never ran.
    result = _run_hook(payload)
    assert result.returncode == 2, f"the hook fails open on {label}"
    assert "Blocked" in result.stderr


def test_phi_hook_is_registered_as_a_pretooluse_hook() -> None:
    # A correct hook nobody wired up guards nothing. Pins that settings.json still points at
    # it from PreToolUse and still covers the tools that read and write files.
    settings = json.loads(
        (REPO_ROOT / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    entries = settings["hooks"]["PreToolUse"]
    matchers = [
        entry["matcher"]
        for entry in entries
        if any(HOOK.name in hook["command"] for hook in entry["hooks"])
    ]
    assert matchers, f"{HOOK.name} is not registered under PreToolUse in settings.json"
    for tool in ("Read", "Edit", "Write"):
        assert any(tool in matcher for matcher in matchers), (
            f"the {HOOK.name} matcher no longer covers {tool}"
        )
