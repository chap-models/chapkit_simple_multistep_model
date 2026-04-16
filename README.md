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
make test           # run `chapkit test` against the running service
```

### Note on `chapkit test`

`chapkit test` injects synthetic test data with only **2 historic periods** at predict time. The model defaults to `n_target_lags=6`, so the lag window is shorter than expected. The model handles this by left-padding `previous_y` with zeros — the synthetic zeros propagate out of the lag window after a few recursive steps, and `chapkit test` passes. If you need a stricter contract (e.g. enforce `len(previous_y) >= n_target_lags`), tighten the validation in `simple_multistep_model.multistep._pad_or_trim`.

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
