"""Console-script entry points: `train` and `predict`.

These wrap the same `on_train` / `on_predict` callables used by the chapkit
service so a future `ShellModelRunner` variant of this template can shell out
to them with no code changes:

    train_command="train --data {data_file}"
    predict_command="predict --historic {historic_file} --future {future_file} --output {output_file}"

Each command:
  * reads `config.yml` from the working directory if present (chapkit writes it)
    and falls back to MultistepConfig defaults otherwise
  * reads CSV inputs given by the named flags
  * emits `model.pickle` (train) or the predictions CSV at `--output` (predict)
"""

from __future__ import annotations

import argparse
import asyncio
import pickle
from pathlib import Path

import pandas as pd
import yaml
from chapkit.data import DataFrame as ChapDataFrame

from .predict import on_predict
from .train import MultistepConfig, on_train


def _load_config(config_path: Path = Path("config.yml")) -> MultistepConfig:
    if not config_path.exists():
        return MultistepConfig()
    raw = yaml.safe_load(config_path.read_text()) or {}
    return MultistepConfig.model_validate(raw)


def train_cli() -> None:
    parser = argparse.ArgumentParser(description="Train the multistep forecaster.")
    parser.add_argument("--data", required=True, help="Path to training data CSV.")
    parser.add_argument(
        "--model",
        default="model.pickle",
        help="Output model pickle path (default: model.pickle).",
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Path to chapkit-emitted config.yml (default: config.yml).",
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    data = ChapDataFrame.from_pandas(pd.read_csv(args.data))
    model = asyncio.run(on_train(config, data))
    Path(args.model).write_bytes(pickle.dumps(model))
    print(f"Model saved to {args.model}")


def predict_cli() -> None:
    parser = argparse.ArgumentParser(description="Predict with the multistep forecaster.")
    parser.add_argument("--historic", required=True, help="Path to historic data CSV.")
    parser.add_argument("--future", required=True, help="Path to future data CSV.")
    parser.add_argument("--output", required=True, help="Path to write predictions CSV.")
    parser.add_argument(
        "--model",
        default="model.pickle",
        help="Path to trained model pickle (default: model.pickle).",
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Path to chapkit-emitted config.yml (default: config.yml).",
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    model = pickle.loads(Path(args.model).read_bytes())
    historic = ChapDataFrame.from_pandas(pd.read_csv(args.historic))
    future = ChapDataFrame.from_pandas(pd.read_csv(args.future))
    predictions = asyncio.run(on_predict(config, model, historic, future))
    predictions.to_pandas().to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
