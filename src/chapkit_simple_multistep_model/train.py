"""Train callable wired into chapkit's FunctionalModelRunner."""

from __future__ import annotations

from typing import Any

from chapkit.data import DataFrame as ChapDataFrame
from geojson_pydantic import FeatureCollection
from sklearn.ensemble import RandomForestRegressor
from skpro.regression.residual import ResidualDouble

from . import DataFrameMultistepModel, SkproWrapper
from .config import INDEX_COLS, TARGET_VARIABLE, MultistepConfig
from .transformations import transform_data


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
        max_features="sqrt",  # type: ignore[arg-type]  # sklearn stub omits Literal["sqrt","log2"]
    )
    skpro_model = ResidualDouble(regressor)
    one_step = SkproWrapper(skpro_model)
    model = DataFrameMultistepModel(one_step, config.n_target_lags, TARGET_VARIABLE)
    model.fit(X, y)
    return model
