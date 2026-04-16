"""End-to-end smoke test against the chapkit FastAPI app using example_data/.

Runs entirely in-process via Starlette's TestClient — no Docker, no port,
no real server.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

JOB_TIMEOUT_SECONDS = 180


def _df_payload(df: pd.DataFrame) -> dict:
    rows = [
        [None if isinstance(v, float) and np.isnan(v) else v for v in row]
        for row in df.itertuples(index=False, name=None)
    ]
    return {"columns": df.columns.tolist(), "data": rows}


def _wait_for_job(client: TestClient, job_id: str) -> None:
    deadline = time.time() + JOB_TIMEOUT_SECONDS
    while time.time() < deadline:
        j = client.get(f"/api/v1/jobs/{job_id}").json()
        if j["status"] == "completed":
            return
        if j["status"] == "failed":
            pytest.fail(f"Job {job_id} failed: {j}")
        time.sleep(0.5)
    pytest.fail(f"Job {job_id} did not complete within {JOB_TIMEOUT_SECONDS}s")


def _latest_artifact(client: TestClient, predicate) -> dict:
    arts = client.get("/api/v1/artifacts?limit=200").json()
    matching = [a for a in arts if predicate(a)]
    assert matching, "no matching artifact found"
    return sorted(matching, key=lambda a: a["created_at"])[-1]


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_info(client: TestClient) -> None:
    info = client.get("/api/v1/info").json()
    assert info["id"] == "chapkit-simple-multistep-template"
    assert info["period_type"] == "monthly"


def test_train_and_predict_against_example_data(client: TestClient, example_data_dir: Path) -> None:
    historic = pd.read_csv(example_data_dir / "historic_data.csv")
    future = pd.read_csv(example_data_dir / "future_data.csv")
    future["disease_cases"] = np.nan  # future inputs don't carry the target

    cfg = client.post(
        "/api/v1/configs",
        json={
            "name": "pytest-smoke",
            "data": {
                "prediction_periods": 12,
                "n_target_lags": 6,
                "n_samples": 50,
                "rf_max_depth": 10,
                "rf_min_samples_leaf": 5,
            },
        },
    ).json()
    cfg_id = cfg["id"]

    train_resp = client.post(
        "/api/v1/ml/$train",
        json={"config_id": cfg_id, "data": _df_payload(historic)},
    ).json()
    _wait_for_job(client, train_resp["job_id"])
    training_artifact = _latest_artifact(
        client,
        lambda a: a["data"]["type"] == "ml_training_workspace" and a["data"]["metadata"]["config_id"] == cfg_id,
    )

    predict_resp = client.post(
        "/api/v1/ml/$predict",
        json={
            "artifact_id": training_artifact["id"],
            "historic": _df_payload(historic),
            "future": _df_payload(future),
        },
    ).json()
    _wait_for_job(client, predict_resp["job_id"])
    prediction_artifact = _latest_artifact(
        client,
        lambda a: a.get("parent_id") == training_artifact["id"] and a["data"]["type"] == "ml_prediction",
    )

    preds = pd.DataFrame(client.get(f"/api/v1/artifacts/{prediction_artifact['id']}/$download").json())
    sample_cols = [c for c in preds.columns if c.startswith("sample_")]

    assert len(preds) == len(future)
    assert len(sample_cols) == 50
    assert set(preds["location"]) == set(future["location"])

    mean = preds[sample_cols].to_numpy().mean()
    assert 10.0 < mean < 500.0, f"prediction mean {mean:.2f} outside plausible range"
