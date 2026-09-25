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

from chapkit_simple_multistep_model.config import CLIMATE_FEATURES

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


def _wait_for_failed_job(client: TestClient, job_id: str) -> dict:
    """Poll until the job fails and return it, failing the test if it succeeds."""
    deadline = time.time() + JOB_TIMEOUT_SECONDS
    while time.time() < deadline:
        j = client.get(f"/api/v1/jobs/{job_id}").json()
        if j["status"] == "failed":
            return j
        if j["status"] == "completed":
            pytest.fail(f"Job {job_id} was expected to fail but completed")
        time.sleep(0.5)
    pytest.fail(f"Job {job_id} did not finish within {JOB_TIMEOUT_SECONDS}s")


def _latest_artifact(client: TestClient, predicate) -> dict:
    arts = client.get("/api/v1/artifacts?limit=200").json()
    matching = [a for a in arts if predicate(a)]
    assert matching, "no matching artifact found"
    return sorted(matching, key=lambda a: a["created_at"])[-1]


def _example_frames(example_data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    historic = pd.read_csv(example_data_dir / "historic_data.csv")
    future = pd.read_csv(example_data_dir / "future_data.csv")
    future["disease_cases"] = np.nan  # future inputs don't carry the target
    return historic, future


def _create_config(client: TestClient, name: str, covariates: list[str], n_samples: int = 50) -> str:
    cfg = client.post(
        "/api/v1/configs",
        json={
            "name": name,
            "data": {
                "prediction_periods": 12,
                "n_target_lags": 6,
                "n_samples": n_samples,
                "rf_max_depth": 10,
                "rf_min_samples_leaf": 5,
                "additional_continuous_covariates": covariates,
            },
        },
    ).json()
    return cfg["id"]


def _train_and_predict(
    client: TestClient,
    cfg_id: str,
    historic: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    """Run $train then $predict for a config and return the downloaded predictions."""
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

    return pd.DataFrame(client.get(f"/api/v1/artifacts/{prediction_artifact['id']}/$download").json())


def _assert_sane_predictions(preds: pd.DataFrame, future: pd.DataFrame) -> None:
    sample_cols = [c for c in preds.columns if c.startswith("sample_")]

    assert len(preds) == len(future)
    assert len(sample_cols) == 50
    assert set(preds["location"]) == set(future["location"])

    mean = preds[sample_cols].to_numpy().mean()
    assert 10.0 < mean < 500.0, f"prediction mean {mean:.2f} outside plausible range"


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_info(client: TestClient) -> None:
    info = client.get("/api/v1/info").json()
    assert info["id"] == "chapkit-simple-multistep-model"
    assert info["period_type"] == "monthly"
    # The service needs nothing beyond disease_cases; covariates come from the
    # configuration, so CHAP must not be told to demand climate columns.
    assert info["required_covariates"] == []


def test_train_and_predict_against_example_data(client: TestClient, example_data_dir: Path) -> None:
    historic, future = _example_frames(example_data_dir)
    cfg_id = _create_config(client, "pytest-smoke", CLIMATE_FEATURES)
    preds = _train_and_predict(client, cfg_id, historic, future)
    _assert_sane_predictions(preds, future)


def test_train_and_predict_without_covariates(client: TestClient, example_data_dir: Path) -> None:
    """The covariate-free configuration must really run, not just validate.

    This is the default configuration and the marketplace's self-history
    variant: lagged case history plus per-location one-hots, no climate data.
    """
    historic, future = _example_frames(example_data_dir)
    cfg_id = _create_config(client, "pytest-selfhistory", [])
    preds = _train_and_predict(client, cfg_id, historic, future)
    _assert_sane_predictions(preds, future)


def test_missing_covariate_is_reported_by_name(client: TestClient, example_data_dir: Path) -> None:
    """A configured covariate the data lacks fails with a named error, not a KeyError."""
    historic, _ = _example_frames(example_data_dir)
    cfg_id = _create_config(client, "pytest-missing-covariate", ["nonexistent_covariate"])

    train_resp = client.post(
        "/api/v1/ml/$train",
        json={"config_id": cfg_id, "data": _df_payload(historic)},
    ).json()
    job = _wait_for_failed_job(client, train_resp["job_id"])

    error = job["error"]
    assert error.startswith("ValueError:"), error
    assert "nonexistent_covariate" in error, error
    assert "additional_continuous_covariates" in error, error
