"""Config + train callable wired into chapkit's FunctionalModelRunner."""

from __future__ import annotations

from typing import Any

from chapkit import BaseConfig
from chapkit.data import DataFrame as ChapDataFrame
from geojson_pydantic import FeatureCollection
from pydantic import Field
from sklearn.ensemble import RandomForestRegressor
from skpro.regression.residual import ResidualDouble

from . import DataFrameMultistepModel, SkproWrapper
from .transformations import transform_data

INDEX_COLS = ["time_period", "location"]
TARGET_VARIABLE = "disease_cases"
DEFAULT_FEATURES = ["rainfall", "mean_temperature", "mean_relative_humidity"]


class MultistepConfig(BaseConfig):
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


async def on_train(
    config: MultistepConfig,
    data: ChapDataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    df = data.to_pandas()
    feature_cols = list(config.additional_continuous_covariates)
    y = df[INDEX_COLS + [TARGET_VARIABLE]]
    X = df[INDEX_COLS + feature_cols]
    X = transform_data(X)

    regressor = RandomForestRegressor(
        max_depth=config.rf_max_depth,
        min_samples_leaf=config.rf_min_samples_leaf,
        max_features="sqrt",
    )
    skpro_model = ResidualDouble(regressor)
    one_step = SkproWrapper(skpro_model)
    model = DataFrameMultistepModel(one_step, config.n_target_lags, TARGET_VARIABLE)
    model.fit(X, y)
    model.feature_columns_ = [c for c in X.columns if c not in INDEX_COLS]
    return model
