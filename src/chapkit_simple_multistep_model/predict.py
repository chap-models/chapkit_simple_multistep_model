"""Predict callable wired into chapkit's FunctionalModelRunner."""

from __future__ import annotations

import pandas as pd
from chapkit.data import DataFrame as ChapDataFrame
from geojson_pydantic import FeatureCollection

from . import DataFrameMultistepModel
from .config import INDEX_COLS, TARGET_VARIABLE, MultistepConfig, check_feature_columns
from .transformations import transform_data


async def on_predict(
    config: MultistepConfig,
    model: DataFrameMultistepModel,
    historic: ChapDataFrame,
    future: ChapDataFrame,
    geo: FeatureCollection | None = None,
) -> ChapDataFrame:
    historic_df = historic.to_pandas()
    future_df = future.to_pandas()
    feature_cols = list(config.additional_continuous_covariates)
    check_feature_columns(historic_df.columns, feature_cols)
    check_feature_columns(future_df.columns, feature_cols)
    cols = INDEX_COLS + feature_cols
    n_steps = int(future_df.groupby("location").size().iloc[0])
    features = pd.concat([historic_df[cols], future_df[cols]], ignore_index=True)
    features = features.sort_values(by=["time_period", "location"])
    X = transform_data(features)
    y_historic = historic_df[INDEX_COLS + [TARGET_VARIABLE]]
    predictions = model.predict(y_historic, X, n_steps, config.n_samples)
    return ChapDataFrame.from_pandas(predictions)
