"""Shared configuration + canonical column constants.

`MultistepConfig` and the canonical column names are imported by both
`train.py` and `predict.py`. They live here (rather than in either of them)
so train and predict stay independent siblings — neither needs the other.
"""

from __future__ import annotations

from collections.abc import Iterable

from chapkit import BaseConfig
from pydantic import Field

INDEX_COLS = ["time_period", "location"]
TARGET_VARIABLE = "disease_cases"
# The climate trio the marketplace's climate configuration asks for. Deliberately
# not the default: a configuration names the covariates it wants, so the service
# never demands columns it does not declare as required.
CLIMATE_FEATURES = ["rainfall", "mean_temperature", "mean_relative_humidity"]


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
        default_factory=list,
        description=(
            "Continuous covariates to include as exogenous features. Empty means the model fits on "
            "lagged case history alone; a configuration names the covariates it wants (the climate "
            "configuration asks for rainfall, mean_temperature, mean_relative_humidity)."
        ),
    )


def check_feature_columns(columns: Iterable[str], feature_cols: Iterable[str]) -> None:
    """Raise if the data lacks covariate columns the configuration asked for.

    Called before the frame is indexed so a mismatch names the columns and the
    way out, rather than surfacing as a pandas KeyError.
    """
    available = set(columns)
    missing = [col for col in feature_cols if col not in available]
    if missing:
        raise ValueError(
            f"data is missing configured covariate columns: {', '.join(missing)}; "
            "set `additional_continuous_covariates` to covariates the data carries, "
            "or leave it empty to fit on lagged case history alone"
        )
