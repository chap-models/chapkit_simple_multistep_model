"""Entry point for `python -m simple_multistep_model` — boots the chapkit service."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "simple_multistep_model.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
