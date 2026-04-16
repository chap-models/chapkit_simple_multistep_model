# chapkit_simple_multistep_template

Chapkit-based service wrapping a multistep recursive disease forecaster (sklearn `RandomForestRegressor` + skpro `ResidualDouble` wrapped in a per-location lag-feature multistep predictor).

See [`docs/migration-to-chapkit.md`](docs/migration-to-chapkit.md) for the full Python conversion guide.

## Quickstart

```
uv sync
uv run python -m simple_multistep_model
```

(or `uv run uvicorn simple_multistep_model.main:app --host 0.0.0.0 --port 8000`)

Then in another shell:

```
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/info
```

## Docker

```
make run            # docker compose up (foreground, builds if needed)
make run-ghcr       # pull prebuilt image from GHCR
```

## Test

```
make test           # pytest end-to-end (in-process via FastAPI TestClient)
```

`tests/test_smoke.py` drives the chapkit ASGI app directly through Starlette's `TestClient` — no docker, no port, no `make run` required. It posts `historic_data.csv` to `$train`, `future_data.csv` to `$predict`, downloads the prediction artifact, and asserts shape + sane mean. Runs in ~3s on a fresh in-memory SQLite scratch dir.

### Note on `chapkit test`

`chapkit test` (the bundled CLI smoke runner) injects synthetic data with only **2 historic periods** per location at predict time and a configurable but fixed location count. With `n_target_lags=6` the lag window underflows; with location-conditional one-hot encoding the column count drifts between train and predict. Both are real shape mismatches under that synthetic contract, but neither reflects how this model is meant to be used in production. Rather than patching the model around them, this repo ships its own pytest smoke against `example_data/` (which has 140 historic periods × 18 locations).

## Layout

```
src/simple_multistep_model/
├── __init__.py
├── __main__.py          # python -m simple_multistep_model
├── main.py              # service composition (info, hierarchy, builder)
├── train.py             # MultistepConfig + on_train
├── predict.py           # on_predict
├── multistep.py         # MultistepModel and friends (the algorithm)
├── one_step_model.py    # SkproWrapper, ResidualBootstrapModel
└── transformations.py   # feature engineering helpers
```

## Example data

`example_data/` ships three CSVs in CHAP canonical format suitable for `POST /api/v1/ml/$train` and `POST /api/v1/ml/$predict`:

| File | Use as |
|---|---|
| `training_data.csv` | `data` body of `$train` (full series, 1998-01 to 2010-08, 18 Lao provinces) |
| `historic_data.csv` | `historic` body of `$predict` (first 140 periods) |
| `future_data.csv` | `future` body of `$predict` (last 12 periods, target column dropped) |

## Lint

```
make lint     # ruff format + ruff check --fix (modifies files)
make check    # CI-friendly: format check + lint without fixing
```
