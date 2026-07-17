"""Run the openmed-studio API with uvicorn: ``python -m openmed_studio``.

Host/port are configurable via ``OPENMED_STUDIO_HOST`` / ``OPENMED_STUDIO_PORT``
(defaults: 127.0.0.1:8080). This is the HTTP surface; the Streamlit app is launched
separately with ``uv run streamlit run streamlit_app.py``.
"""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("OPENMED_STUDIO_HOST", "127.0.0.1")
    port = int(os.environ.get("OPENMED_STUDIO_PORT", "8080"))
    uvicorn.run("openmed_studio.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
