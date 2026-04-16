"""Chapkit service wrapping the simple_multistep_model recursive forecaster."""

from __future__ import annotations

import os
from pathlib import Path

from chapkit.api import (
    AssessedStatus,
    MLServiceBuilder,
    MLServiceInfo,
    ModelMetadata,
    PeriodType,
)
from chapkit.artifact import ArtifactHierarchy
from chapkit.ml import FunctionalModelRunner

from .config import MultistepConfig
from .predict import on_predict
from .train import on_train

runner: FunctionalModelRunner[MultistepConfig] = FunctionalModelRunner(
    on_train=on_train,
    on_predict=on_predict,
)

info = MLServiceInfo(
    id="chapkit-simple-multistep-template",
    display_name="Simple Multistep Model (chapkit)",
    version="0.1.0",
    description=(
        "Multistep recursive forecaster: RandomForest + skpro ResidualDouble "
        "wrapped in a recursive multi-step predictor with per-location lag features."
    ),
    model_metadata=ModelMetadata(
        author="CHAP team",
        author_assessed_status=AssessedStatus.gray,
        organization="HISP Centre, University of Oslo",
        contact_email="morten@dhis2.org",
    ),
    period_type=PeriodType.monthly,
    allow_free_additional_continuous_covariates=True,
    required_covariates=[],
    min_prediction_periods=1,
    max_prediction_periods=100,
)

hierarchy = ArtifactHierarchy(
    name="simple_multistep_model",
    level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/chapkit.db")
if DATABASE_URL.startswith("sqlite") and ":///" in DATABASE_URL:
    db_path = Path(DATABASE_URL.split("///")[1])
    db_path.parent.mkdir(parents=True, exist_ok=True)

app = (
    MLServiceBuilder(
        info=info,
        config_schema=MultistepConfig,
        hierarchy=hierarchy,
        runner=runner,
        database_url=DATABASE_URL,
    )
    .with_registration(keepalive_interval=15)
    .build()
)
