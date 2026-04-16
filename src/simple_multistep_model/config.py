"""Shared configuration + canonical column constants.

`MultistepConfig` and the canonical column names are imported by both
`train.py` and `predict.py`. They live here (rather than in either of them)
so train and predict stay independent siblings — neither needs the other.
"""

from __future__ import annotations

from chapkit import BaseConfig
from pydantic import Field

INDEX_COLS = ["time_period", "location"]
TARGET_VARIABLE = "disease_cases"
DEFAULT_FEATURES = ["rainfall", "mean_temperature", "mean_relative_humidity"]


class MultistepConfig(BaseConfig):
    prediction_periods: int = Field(
        default=12,
        description="Number of periods to forecast into the future.",
    )
    n_target_lags: int = Field(
        default=6,
        description="Number of lagged target values to feed the one-step regressor.",
    )
    n_samples: int = Field(
        default=100,
        description="Number of trajectory samples drawn per location at predict time.",
    )
    rf_max_depth: int = Field(
        default=10,
        description="Max depth of the underlying RandomForest regressor.",
    )
    rf_min_samples_leaf: int = Field(
        default=5,
        description="Minimum samples per leaf for the underlying RandomForest regressor.",
    )
    additional_continuous_covariates: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FEATURES),
        description=(
            "Continuous covariates to include as exogenous features. Defaults match the "
            "pedagogical pipeline (rainfall, mean_temperature, mean_relative_humidity)."
        ),
    )
