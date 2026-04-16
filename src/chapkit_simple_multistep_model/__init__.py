"""Simple multistep recursive forecasting model.

Compose:
1. An sklearn regressor (any regressor)
2. A probabilistic wrapper (ResidualBootstrapModel or SkproWrapper)
3. A MultistepModel for recursive forecasting
4. A data-transform function (plain DataFrame -> DataFrame)
"""

from .multistep import (
    DataFrameMultistepModel,
    DeterministicMultistepModel,
    MultistepDistribution,
    MultistepModel,
    features_to_xarray,
    future_features_to_xarray,
    target_to_xarray,
)
from .one_step_model import (
    ResidualBootstrapModel,
    ResidualDistribution,
    SkproWrapper,
)

__all__ = [
    "DataFrameMultistepModel",
    "DeterministicMultistepModel",
    "MultistepModel",
    "MultistepDistribution",
    "ResidualBootstrapModel",
    "ResidualDistribution",
    "SkproWrapper",
    "target_to_xarray",
    "features_to_xarray",
    "future_features_to_xarray",
]
