"""Run the de-identification API with uvicorn: ``python -m openmed_deid``.

Host/port are configurable via ``OPENMED_DEID_HOST`` / ``OPENMED_DEID_PORT``
(defaults: 127.0.0.1:8080).
"""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("OPENMED_DEID_HOST", "127.0.0.1")
    port = int(os.environ.get("OPENMED_DEID_PORT", "8080"))
    uvicorn.run("openmed_deid.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
