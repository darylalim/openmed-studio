"""Regenerate the README screenshots in docs/screenshots/.

Starts the Streamlit app on a spare loopback port, drives each tab in a headless browser, and
saves a dark and a light capture of it (the browser's `prefers-color-scheme` picks the app's
Night Rounds / Day Rounds theme, so no menu clicks are needed). Run from the repo root:

    uv run python scripts/capture_screenshots.py

playwright is in the `dev` group, but its wheel carries no browser. The script uses the
installed Google Chrome (`--channel chrome`, the default), so no browser download is needed;
without Chrome, run `uv run playwright install chromium` once and pass `--channel chromium`.

The first run of each tab downloads its model if it isn't cached (the PII model, and the
Disease NER model for Clinical NER). Once they are cached, prefix the command with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to skip the Hub checks.

Every capture shows synthetic text only: the app's own EXAMPLE_NOTE, plus NER_NOTE below for
Clinical NER (EXAMPLE_NOTE holds identifiers and no conditions, so the Disease model finds
nothing in it). Keep it that way — these images are published in the README.

The PNGs are quantized with pngquant when it is on PATH (`brew install pngquant`), which cuts
them by about two-thirds with no visible change; without it they are saved uncompressed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal, NamedTuple

from playwright.sync_api import Locator, Page, sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "screenshots"

# Viewport in CSS pixels, captured at 2x so text stays sharp on high-density screens.
WIDTH, HEIGHT, SCALE = 1600, 940, 2
# Crop the empty band above the title (where the collapsed sidebar's expand chevron sits).
TOP_CROP = 56

NER_NOTE = (
    "72-year-old man with type 2 diabetes mellitus, hypertension and stage 3 chronic kidney "
    "disease, admitted with community-acquired pneumonia. History of atrial fibrillation and a "
    "prior myocardial infarction. Chest X-ray also noted mild pulmonary fibrosis. Discharged on "
    "oral antibiotics; follow up for diabetic neuropathy and worsening heart failure symptoms."
)


class Shot(NamedTuple):
    slug: str  # file stem: docs/screenshots/<slug>-{dark,light}.png
    tab: str  # tab label, as shown in the tab strip
    button: str  # the tab's submit button
    ready_text: str  # text that appears in the tab only once its result has rendered
    # "table" cuts just above the entity table (Detect/Clinical NER show it expanded, and it
    # runs past the fold); "panel" keeps the whole tab (the de-id tabs fold it into an expander).
    crop: Literal["table", "panel"]
    note: str | None = None  # replaces the text area's EXAMPLE_NOTE when set


SHOTS = (
    Shot("single-note", "Single note", "De-identify", "Detected in original", "panel"),
    Shot("detect", "Detect", "Detect", "Distinct types", "table"),
    Shot(
        "clinical-ner", "Clinical NER", "Analyze", "Distinct types", "table", NER_NOTE
    ),
    Shot(
        "policy-de-id",
        "Policy de-ID",
        "Anonymize under policy",
        "Detected in original",
        "panel",
    ),
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_healthy(
    url: str, server: subprocess.Popen[bytes], timeout: float = 60
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            sys.exit(f"Streamlit exited early with code {server.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    sys.exit(f"Streamlit did not become healthy within {timeout:.0f}s")


def _start_server(port: int) -> subprocess.Popen[bytes]:
    # The shipped .streamlit/config.toml already pins the loopback bind; the minimal toolbar
    # hides the developer-only "Deploy" button a localhost run would otherwise show. No file
    # watcher: nothing here reloads, and its sys.modules walk trips transformers' lazy
    # vision modules into thousands of lines of `No module named 'torchvision'` tracebacks.
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--client.toolbarMode",
            "minimal",
            "--server.fileWatcherType",
            "none",
        ],
        cwd=REPO_ROOT,
    )


def _collapse_sidebar(page: Page) -> None:
    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar.wait_for()
    sidebar.hover()  # the collapse button only shows on hover
    page.locator('[data-testid="stSidebarCollapseButton"] button').click()
    page.wait_for_timeout(600)  # let the slide-out animation finish


def _crop_bottom(panel: Locator, crop: str) -> float:
    if crop == "table":
        box = panel.locator('[data-testid="stDataFrame"]').first.bounding_box()
        assert box is not None, "entity table not rendered"
        return box["y"] - 4
    box = panel.bounding_box()
    assert box is not None, "tab panel not rendered"
    return box["y"] + box["height"] + 36


def _capture(page: Page, url: str, shot: Shot, path: Path) -> None:
    page.goto(url)
    _collapse_sidebar(page)
    page.get_by_role("tab", name=shot.tab).click()
    # The open tab's panel is the one holding this tab's submit button.
    submit = page.get_by_role("button", name=shot.button)
    panel = page.get_by_role("tabpanel").filter(has=submit)
    if shot.note is not None:
        panel.locator("textarea").first.fill(shot.note)
    panel.get_by_role("button", name=shot.button).click()
    # Generous: a cold run may download the model first.
    panel.get_by_text(shot.ready_text).first.wait_for(timeout=600_000)
    page.wait_for_timeout(2500)  # let charts and highlight marks settle
    bottom = _crop_bottom(panel, shot.crop)
    page.screenshot(
        path=path,
        clip={"x": 0, "y": TOP_CROP, "width": WIDTH, "height": bottom - TOP_CROP},
    )


def _compress(paths: list[Path]) -> None:
    pngquant = shutil.which("pngquant")
    if pngquant is None:
        print(
            "pngquant not found; screenshots saved uncompressed (brew install pngquant)"
        )
        return
    # --skip-if-larger exits 98/99 when it leaves a file as is, which is fine here.
    result = subprocess.run(
        [
            pngquant,
            "--quality",
            "70-90",
            "--strip",
            "--skip-if-larger",
            "--force",
            "--ext",
            ".png",
            *map(str, paths),
        ],
        check=False,
    )
    if result.returncode not in (0, 98, 99):
        sys.exit(f"pngquant failed with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[shot.slug for shot in SHOTS],
        help="capture just these shots (default: all)",
    )
    parser.add_argument(
        "--channel",
        default="chrome",
        help="playwright browser channel (default: chrome, the installed Google Chrome)",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    args = parser.parse_args()

    shots = [shot for shot in SHOTS if args.only is None or shot.slug in args.only]
    args.out.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    server = _start_server(port)
    written: list[Path] = []
    try:
        _wait_healthy(url, server)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel)
            for shot in shots:
                for scheme in ("dark", "light"):
                    page = browser.new_page(
                        viewport={"width": WIDTH, "height": HEIGHT},
                        device_scale_factor=SCALE,
                        color_scheme=scheme,
                    )
                    path = args.out / f"{shot.slug}-{scheme}.png"
                    _capture(page, url, shot, path)
                    page.close()
                    written.append(path)
                    print(f"captured {path.relative_to(REPO_ROOT)}")
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    _compress(written)
    total = sum(path.stat().st_size for path in written)
    print(f"{len(written)} screenshots, {total / 1024:.0f} KiB total")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
