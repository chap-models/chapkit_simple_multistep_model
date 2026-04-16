"""Shared test fixtures.

Drives the chapkit FastAPI app in-process via Starlette's TestClient — no
real server required. A per-session temp SQLite file is used so background
jobs persist across the polling requests within a single test.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Must be set BEFORE importing the app, since main.py reads DATABASE_URL at
# module import time. Using a file (not :memory:) so chapkit's connection pool
# sees a single shared database across requests.
_DB_DIR = Path(tempfile.mkdtemp(prefix="chapkit_test_"))
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_DB_DIR}/test.db")

from fastapi.testclient import TestClient  # noqa: E402

from simple_multistep_model.main import app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DATA = REPO_ROOT / "example_data"


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def example_data_dir() -> Path:
    return EXAMPLE_DATA
