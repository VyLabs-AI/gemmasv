"""Run the hosted demo API.

Example:
    HERO_ENGINE=gemma HERO_ALLOWED_ORIGINS=https://example.github.io \
      python -m gemma_sv.demo_server
"""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    uvicorn.run(
        "gemma_sv.demo_server.api:app",
        host=os.getenv("HERO_HOST", "127.0.0.1"),
        port=int(os.getenv("HERO_PORT", "8001")),
        workers=1,  # in-memory sessions and one shared model must stay in one worker
        access_log=os.getenv("HERO_ACCESS_LOG", "0") == "1",
        proxy_headers=os.getenv("HERO_PROXY_HEADERS", "1") == "1",
    )


if __name__ == "__main__":
    main()
