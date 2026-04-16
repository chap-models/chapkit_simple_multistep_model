# Migrating a Python ML model repo to chapkit

This guide walks through converting a standalone Python ML model (e.g. a couple of `argparse`-driven `train.py` / `predict.py` scripts) into a chapkit-based service that CHAP can discover, configure, train, and query over HTTP.

The recommended path is to scaffold a fresh chapkit project with the `chapkit` CLI and then port your model code into it. That is what this guide covers.

For the **R / non-Python** equivalent of this guide (shell runner, R-INLA Dockerfile, column adapters), see [`chapkit_ewars_template/docs/migration-to-chapkit.md`](https://github.com/chap-models/chapkit_ewars_template/blob/main/docs/migration-to-chapkit.md).

## Worked example: `chapkit_simple_multistep_model` to `chapkit_simple_multistep_model`

The concrete before/after used throughout this guide:

- **Starting point:** a flat Python repo with `train.py`, `predict.py`, a `chapkit_simple_multistep_model/` package implementing a recursive multistep forecaster (sklearn `RandomForestRegressor` wrapped in skpro `ResidualDouble`), `transformations.py` with feature engineering, and `training_data.csv`. The CLI is `argparse` with positional args (`python train.py training_data.csv model.pkl`).
- **Ending point:** the same repo restructured to an installable `src/chapkit_simple_multistep_model/` package containing the model algorithm plus three thin chapkit-binding modules — `main.py` (~67 lines), `train.py` (~68 lines), `predict.py` (~31 lines), and `__main__.py` (12 lines for `python -m chapkit_simple_multistep_model`). A `uv_build`-backed `pyproject.toml`, a Python 3.13 + uv Dockerfile based on `ghcr.io/astral-sh/uv:0.10-python3.13-trixie-slim`, and CI that runs `chapkit test` against the built container. The original positional-arg `train.py` / `predict.py` were removed once the wrapper was verified to produce statistically identical output (mean-of-means within ~0.15% over 216 location/period predictions).

Every file path and code snippet in this guide is lifted verbatim from this repository.

## Universal vs Python-specific changes

A migration has three kinds of changes. Knowing which is which saves time:

1. **Universal — every migration needs these.** Replacing your CLI orchestration with chapkit's `main.py`. Defining a `BaseConfig` subclass for tunable parameters. Wiring a `FunctionalModelRunner` to your train and predict callables. Layering a Python + uv image. Deleting old positional-arg CLIs from the boot path.
2. **Python-specific — needed for any pure-Python model.** Wrapping your existing functions in `async on_train` / `async on_predict` callables that accept `chapkit.data.DataFrame` (a Pydantic schema, not pandas) and return a pickleable model / a `chapkit.data.DataFrame`. Calling `.to_pandas()` on inputs and `DataFrame.from_pandas(...)` on outputs.
3. **Model-specific — needed only if your model needs it.** This repo's example: handling NaN-filled recent rows in the training CSV; tolerating multi-location pooling with one-hot encoded `location` features. If your model already deals cleanly with these, leave them alone.

The guide flags each section with which category it falls into.

## Guide structure

- **Part A — Required.** The minimum set of steps to produce a functioning chapkit service. Skip nothing here.
- **Part B — Optional.** Lint, CI, compose, docs. Add as you see fit.

A manual scaffolding appendix is included at the end for situations where you cannot use the CLI.

---

## What chapkit gives you

Before changing anything, it helps to know what you get in return.

A running chapkit service exposes these endpoints out of the box — you do not write any route code:

- `GET /health` — health check, used by CI and CHAP.
- `GET /api/v1/info` — service metadata (id, covariates, period type, prediction bounds).
- `GET /api/v1/configs/$schema` — JSON schema of your config class.
- `POST /api/v1/configs` — create a config for a run.
- `POST /api/v1/ml/$train` — submit a training job.
- `POST /api/v1/ml/$predict` — submit a prediction job, given a training-artifact id.
- `GET /api/v1/jobs/{id}` — poll job status.
- `GET /api/v1/artifacts` — retrieve trained models and predictions.

You also get:

- A SQLite-backed job and artifact store (default path `data/chapkit.db`).
- Pydantic validation of every config the service receives.
- A `chapkit test` CLI that drives the service end-to-end over HTTP — used for local smoke tests and CI.
- A `chapkit init` CLI that scaffolds the entire project layout (main.py, Dockerfile, compose, README, Postman collection) in one command.

What chapkit does **not** provide: your modeling code, your data adapters, or your specific Dockerfile when you have heavy native deps. Those stay with you.

---

## Prerequisites

- A working Python model that can train and predict from CSV inputs as standalone scripts.
- Python 3.13 and [uv](https://github.com/astral-sh/uv).
- Docker.
- Familiarity with the CHAP canonical column names: `disease_cases`, `population`, `location`, `time_period`, and optional continuous covariates like `rainfall`, `mean_temperature`, `mean_relative_humidity`.

---

# Part A — Required

After completing Part A you can run `uvicorn main:app` locally, hit `/health`, and pass an end-to-end train + predict cycle.

## A.1 Install the chapkit CLI

Install chapkit as a global uv tool so `chapkit` is available on your `PATH`:

```
uv tool install chapkit
```

Upgrade later with:

```
uv tool upgrade chapkit
```

Verify:

```
chapkit --version
chapkit --help
```

You should see `init` and `artifact` as subcommands when you run `chapkit --help` outside a chapkit project. Inside a chapkit project, `init` is hidden and `test` is shown instead.

## A.2 Scaffold a new project with `chapkit init`

> **Important: run `chapkit init` from *outside* any existing chapkit project.** The CLI walks up from the current directory looking for a `pyproject.toml` that depends on `chapkit`. If it finds one, the `init` command is hidden entirely and you will only see `test` and `artifact`. Move to a parent directory first — for example `cd ~/dev` or `cd /tmp` — before running init.

For a Python model, use the default `ml` template (which uses `FunctionalModelRunner`, not `ShellModelRunner`):

```
chapkit init my-model --template ml
```

### Template options

The `--template` flag picks the scaffold shape:

| Template | Use when |
|---|---|
| `ml` *(default)* | Model logic lives inline in `main.py` as Python `on_train` / `on_predict` callables wired to `FunctionalModelRunner`. **Use this for pure-Python models.** This is the template this repo is based on. |
| `ml-shell` | Model logic lives in external scripts invoked via `ShellModelRunner`. Use for R, Julia, or any non-Python model. |
| `task` | Generic task runner, not a machine-learning service. Ignore for model migrations. |

### Optional flags

- `--path <dir>` — target parent directory (default: current directory).
- `--with-monitoring` — also generate a Prometheus + Grafana monitoring stack under `monitoring/`.

### What the scaffold generates

For `--template ml` you will get a flat layout, exactly seven files:

```
my-model/
├── main.py                  # chapkit service definition with placeholder on_train/on_predict
├── pyproject.toml           # pinned chapkit dep + dev group with uvicorn[standard]
├── Dockerfile               # multi-stage Python 3.13 + uv builder, gunicorn+uvicorn worker, urllib healthcheck, non-root
├── compose.yml              # local build + run with named volume, healthcheck, restart policy, bridge network
├── README.md                # per-project quickstart
├── postman_collection.json  # importable API collection (bonus)
└── .gitignore               # Python defaults
```

Then:

```
cd my-model
uv sync
uv run uvicorn main:app
```

The skeleton boots and serves on `:8000`. Hit `http://localhost:8000/health` to confirm.

The placeholder `on_train` computes per-column means; `on_predict` returns the average of those means as a single `sample_0` value for every future row. Trivial, but enough that **the pristine scaffold passes `chapkit test` out of the box** (verified) — useful as a sanity check before you start replacing things.

The rest of Part A is swapping those placeholders for your real model. Part B covers optional structural improvements you typically reach for once your codebase outgrows a single `main.py` (src layout, separate train/predict modules, console scripts, etc.).

## A.3 Customize `main.py`

The scaffold's `main.py` already wires up imports, the `Config` class, two placeholder async callables, `FunctionalModelRunner`, `ArtifactHierarchy`, and `MLServiceBuilder`. You only need to edit four things: config fields, the two callables, service info, and (optionally) the hierarchy name. Each is walked through below, with concrete examples from this repo's [`src/chapkit_simple_multistep_model/main.py`](../src/chapkit_simple_multistep_model/main.py).

> **Layout note.** This repo eventually moved `main.py` into `src/chapkit_simple_multistep_model/` with a `uv_build` build backend, plus separate `train.py` and `predict.py` modules. The flat layout the scaffold ships is what you should start with — restructure only when `main.py` actually outgrows itself. See [B.10](#b10-outgrow-the-flat-scaffold-layout-srcuv_build--separate-trainpy--predictpy).

### A.3.1 Config class

Find the `Config` class in `main.py` and replace its fields with your model's tunable parameters. Subclass stays `BaseConfig`. Use `Field(default=..., description=...)` so the generated schema is self-documenting:

```python
from pydantic import Field
from chapkit import BaseConfig


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
        default_factory=lambda: ["rainfall", "mean_temperature", "mean_relative_humidity"],
        description="Continuous covariates to include as exogenous features.",
    )
```

Every field here replaces what used to live as module-level constants in `train.py` (`N_TARGET_LAGS`, `N_SAMPLES`, hardcoded RF hyperparams). Validation and defaults now live in Python, and the schema is auto-exposed at `/api/v1/configs/$schema`. `additional_continuous_covariates` is a reserved `BaseConfig` field — overriding its default here means the model uses all three climate covariates by default, but deployments can create variant configs (e.g. `[]` for a population-only run) via `POST /api/v1/configs` without forking the repo.

### A.3.2 The two async callables — `on_train` and `on_predict`

This is the heart of the Python migration. Where R models hand off to a shell command, Python models hand off to two `async` functions you define in `main.py`. The signatures are fixed by `FunctionalModelRunner`:

```python
async def on_train(
    config: ConfigT,
    data: ChapDataFrame,
    geo: FeatureCollection | None = None,
) -> Any:                          # must be pickleable
    ...

async def on_predict(
    config: ConfigT,
    model: Any,                    # the same object on_train returned
    historic: ChapDataFrame,
    future: ChapDataFrame,
    geo: FeatureCollection | None = None,
) -> ChapDataFrame:                # predictions, wide format
    ...
```

> **`ChapDataFrame` is `chapkit.data.DataFrame` — a Pydantic schema, NOT a pandas DataFrame.** Pure pandas methods like `.sort_values()` will fail on it. Convert at both boundaries:
> - Inputs: `df = data.to_pandas()`
> - Output: `return ChapDataFrame.from_pandas(predictions)`
>
> This is the single most common mistake when migrating a Python model. The `chapkit test` CLI surfaces it as `AttributeError: 'DataFrame' object has no attribute 'sort_values'` on the first training job.

In this repo the two callables live in separate files — [`src/chapkit_simple_multistep_model/train.py`](../src/chapkit_simple_multistep_model/train.py) and [`src/chapkit_simple_multistep_model/predict.py`](../src/chapkit_simple_multistep_model/predict.py) — and the `MultistepConfig` Pydantic class plus the canonical column constants (`INDEX_COLS`, `TARGET_VARIABLE`, `DEFAULT_FEATURES`) live in a sibling [`config.py`](../src/chapkit_simple_multistep_model/config.py) so neither train nor predict depends on the other. `main.py` just imports and wires them all. That keeps each module focused: `config.py` is the schema, `train.py` is the training callable, `predict.py` is the prediction callable, `main.py` is service composition. The old positional-arg `train.py` / `predict.py` from the original repo were the obvious template for this split:

```python
from chapkit.data import DataFrame as ChapDataFrame
from chapkit_simple_multistep_model import DataFrameMultistepModel, SkproWrapper
from transformations import transform_data
from sklearn.ensemble import RandomForestRegressor
from skpro.regression.residual import ResidualDouble

INDEX_COLS = ["time_period", "location"]
TARGET_VARIABLE = "disease_cases"


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
    return model


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
    cols = INDEX_COLS + feature_cols
    n_steps = int(future_df.groupby("location").size().iloc[0])
    features = pd.concat([historic_df[cols], future_df[cols]], ignore_index=True)
    features = features.sort_values(by=["time_period", "location"])
    X = transform_data(features)
    y_historic = historic_df[INDEX_COLS + [TARGET_VARIABLE]]
    predictions = model.predict(y_historic, X, n_steps, config.n_samples)
    return ChapDataFrame.from_pandas(predictions)
```

Things to notice:

- The body of each callable is essentially the body of the old `train()` / `predict()` functions, with hardcoded constants replaced by `config` reads and the conversion calls bracketing them.
- The model object returned from `on_train` is whatever you want — chapkit pickles it for you and hands the same object back to `on_predict`. No file paths, no `pickle.dump` / `pickle.load` in your code.
- Predictions must be a wide-format DataFrame with the index columns plus `sample_0`, `sample_1`, … `sample_N` columns (one per probabilistic trajectory). For deterministic models, return a single `sample_0`. CHAP and `chapkit test` rely on the `sample_*` naming convention.
- No CLI argument parsing, no file I/O, no print statements. Logging happens via `structlog` if you want it.

### A.3.3 Wire the runner

Once the two callables exist, the runner is a one-liner:

```python
from chapkit.ml import FunctionalModelRunner

runner: FunctionalModelRunner[MultistepConfig] = FunctionalModelRunner(
    on_train=on_train,
    on_predict=on_predict,
)
```

The scaffold ships this; you just rename the type parameter to your config class.

### A.3.4 Service metadata

`MLServiceInfo` is what CHAP uses to discover your service. Be honest about covariates and bounds — CHAP validates requests against them.

```python
from chapkit.api import AssessedStatus, MLServiceInfo, ModelMetadata, PeriodType

info = MLServiceInfo(
    id="chapkit-simple-multistep-model",
    display_name="Simple Multistep Model (chapkit)",
    version="0.1.0",
    description=(
        "Pedagogical multistep recursive forecaster: RandomForest + skpro ResidualDouble "
        "wrapped in a recursive multi-step predictor with per-location lag features."
    ),
    model_metadata=ModelMetadata(
        author="CHAP team",
        author_assessed_status=AssessedStatus.orange,
        organization="HISP Centre, University of Oslo",
        contact_email="morten@dhis2.org",
    ),
    period_type=PeriodType.monthly,
    allow_free_additional_continuous_covariates=True,
    required_covariates=[],
    min_prediction_periods=1,
    max_prediction_periods=100,
)
```

Fields you almost always need to customize:

- `id` — unique service identifier. Lowercase, hyphenated.
- `display_name`, `version`, `description` — human-readable metadata.
- `model_metadata` — author, organization, assessed confidence, citation.
- `period_type` — `PeriodType.monthly` or `PeriodType.weekly`.
- `required_covariates` — list of canonical CHAP column names your model requires in addition to `disease_cases`. Leave `[]` if everything beyond `disease_cases` is optional.
- `allow_free_additional_continuous_covariates` — whether the model can accept extra continuous covariates beyond those required.
- `min_prediction_periods`, `max_prediction_periods` — forecast horizon bounds.

### A.3.5 Artifact hierarchy (usually no edit needed)

The scaffold ships a reasonable default:

```python
from chapkit.artifact import ArtifactHierarchy

hierarchy = ArtifactHierarchy(
    name="chapkit_simple_multistep_model",
    level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
)
```

Rename `name` to match your project; you usually don't need to touch the level labels.

### A.3.6 Database and builder (usually no edit needed)

The scaffold already wires `MLServiceBuilder` and the SQLite database path:

```python
import os
from pathlib import Path
from chapkit.api import MLServiceBuilder

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
```

`.with_registration()` enables self-registration with chap-core on startup via the `SERVICEKIT_ORCHESTRATOR_URL` environment variable. When unset, registration is skipped and the service runs standalone. See section B.7 for details.

Leave the rest untouched unless you need a different persistence backend.

## A.4 Bring your model code into the scaffold

The most natural workflow when starting from `chapkit init`:

1. **Treat the scaffold as the new repo.** Copy your Python package, helper modules, and any other assets your model needs into the scaffolded project. Add their import deps to `pyproject.toml` (`uv add scikit-learn skpro pandas xarray ...`), commit, push to a new git repo. This is the path the rest of the guide assumes.
2. **Copy the scaffold into your existing repo.** Cherry-pick `main.py`, `pyproject.toml`, `Dockerfile`, `compose.yml`, and `.gitignore` out of the scaffold and paste them into your existing repo alongside your model package. Useful when you have substantial git history to preserve. (This repo took this path.)

Either way, remove every file that existed only to run the old non-chapkit pipeline:

- `MLproject` (MLflow manifest), if you had one.
- Old positional-arg `train.py` / `predict.py`. Once you've verified the chapkit wrapper produces equivalent output, delete them — they no longer have a role at runtime.
- Pre-computed prediction CSVs and model binaries committed to the repo — they bloat the repo and go stale.
- Custom `cli.py` or runner shims that only existed to glue the scripts together.

Keep: your model package, transformation helpers, example data inputs, and anything `main.py` imports at runtime.

## A.5 Adjust the Dockerfile (usually nothing to do)

The scaffolded Dockerfile is **production-ready out of the box**: multi-stage uv builder + `python:3.13-slim` runtime, gunicorn + uvicorn worker, healthcheck via `urllib.request` (no curl dep), non-root user, OCI labels, environment knobs for workers / timeout / log format. You typically don't need to touch it for a pure-Python model — `docker compose up --build` just works.

Only edit it when you have a specific reason:

- **Native build deps** — your model needs C/Fortran extensions that don't ship as wheels (e.g. some scientific libraries):
  ```dockerfile
  RUN apt-get install -y --no-install-recommends build-essential gfortran libopenblas-dev
  ```
  Add this in the `builder` stage before `uv sync`.
- **GPU support** — swap the base image for an NVIDIA CUDA + Python 3.13 image and add the appropriate extras to `pyproject.toml`.
- **amd64-only native deps** — pin `--platform=linux/amd64` (and add `platform: linux/amd64` to `compose.yml`).
- **Personal preference for a simpler / smaller single-stage image** — see the alternative below. This repo took that route, but the scaffold's multi-stage image is equally valid and arguably more production-shaped.

### Alternative: single-stage in the style of chap-core

If you prefer the [`chap-core`](https://github.com/dhis2/chap-core) Dockerfile shape — single stage, direct `uvicorn` CMD, fewer layers — this repo ships one as a reference ([`Dockerfile`](../Dockerfile)):

```dockerfile
FROM ghcr.io/astral-sh/uv:0.10-python3.13-trixie-slim

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLCONFIGDIR=/tmp
ENV PORT=8000
ENV PATH="/app/.venv/bin:$PATH"

ARG GIT_REVISION=""
ENV GIT_REVISION=${GIT_REVISION}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends git curl tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    useradd --no-create-home --shell /usr/sbin/nologin chap

WORKDIR /app

COPY --chown=root:root pyproject.toml uv.lock README.md ./
COPY --chown=root:root chapkit_simple_multistep_model ./chapkit_simple_multistep_model

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY --chown=root:root main.py transformations.py ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev && \
    python -m compileall -q chapkit_simple_multistep_model

RUN mkdir -p /app/data && chown chap:chap /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/health || exit 1

USER chap

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
```

Differences from the scaffolded multi-stage Dockerfile:

- **Single stage** instead of builder + runtime split — fewer moving parts, comparable image size for typical Python ML deps.
- **Direct `uvicorn` CMD** instead of gunicorn + worker config — simpler to debug; switch back to gunicorn when you want multi-worker prefork.
- **`uv sync --frozen` driven by `uv.lock`** — same as the scaffold; run `uv lock` after every dep change and commit the lockfile.
- **Two-step copy** for layer caching: model package first (changes rarely), then thin glue modules (change often).
- **Non-root user, healthcheck, `tini` as init** — same production hardening as the scaffold, just expressed differently.

## A.6 Verify the migration

This is the gate — if a basic train + predict cycle succeeds, your required migration is done.

1. Sync deps and start the service:

   ```
   uv sync
   uv run uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. In another shell, hit `/health` and `/api/v1/info`:

   ```
   curl http://localhost:8000/health        # {"status":"healthy", ...}
   curl http://localhost:8000/api/v1/info   # should reflect your MLServiceInfo
   ```

3. Run the built-in end-to-end test. Inside a chapkit project the CLI exposes the `test` subcommand:

   ```
   chapkit test --url http://localhost:8000 --verbose
   ```

   This creates a config, submits one training job, submits one prediction job, and polls for completion. On slow models bump `--timeout` (default 60s). The pristine scaffold passes this; if your real model fails it, see the heads-up below.

   > **Heads up — `chapkit test` synthetic data shape.** The data `chapkit test` generates is deliberately tiny: by default 100 rows total, 5 locations, ~20 train periods, and only **2 historic periods + 2 future periods** at predict time. Two real-model assumptions commonly break under that contract:
   >
   > 1. **Lag windows longer than 2.** A model with `n_target_lags=6` expects 6 historic periods per location, but `chapkit test` only sends 2.
   > 2. **Location-conditional features.** One-hot encoding produces a different column count when train and predict see different location sets.
   >
   > Both are real shape mismatches under the synthetic contract, but neither reflects how a forecasting model is meant to be used in production. Two paths to deal with this:
   >
   > - **Patch the model to be robust to those shapes** — left-pad `previous_y` with zeros, capture the post-transform column list at train time and reindex on predict. Cheap, but bakes chapkit-test-specific behavior into the model.
   > - **Skip `chapkit test` and write your own pytest smoke** that posts `example_data/` against the chapkit FastAPI app. This repo took this path — see [B.4](#b4-pytest-end-to-end-smoke-in-process-via-fastapi-testclient).
   >
   > Pick whichever fits.

4. Build the container and rerun step 3 against it:

   ```
   docker compose up --build
   # in another shell:
   curl http://localhost:8000/health
   chapkit test --url http://localhost:8000 --timeout 180 --verbose
   ```

When your smoke (`chapkit test` or your own pytest) reports passing, Part A is complete.

---

# Part B — Optional

Everything below is polish. Add whatever pays off for your workflow.

## B.1 Makefile

A `Makefile` saves contributors from memorizing `docker build` flags and `uv run` invocations. See this repo's [`Makefile`](../Makefile) for `build`, `run`, `run-ghcr`, `lint`, and `check` targets:

- `make lint` runs `ruff format` + `ruff check --fix` (modifies files locally).
- `make check` runs the format / lint without `--fix` (use this in CI).

`make build` uses `--no-cache` so CI-equivalent images are reproducible; drop the flag if iteration speed matters more.

## B.2 Ruff lint and format

Add a `[tool.ruff]` section to `pyproject.toml` and wire it into `make lint` / `make check`. Useful because `main.py` is often the only Python file at the repo root in a chapkit project — without a linter, drift accumulates fast. See this repo's [`pyproject.toml`](../pyproject.toml) for the config used here.

## B.3 Docker Compose

Two small compose files make local development painless:

- [`compose.yml`](../compose.yml) — builds the image locally and runs it on `:8000`.
- [`compose.ghcr.yml`](../compose.ghcr.yml) — pulls the prebuilt image from GHCR instead of building.

The scaffold already ships `compose.yml`. Add `compose.ghcr.yml` once you start publishing images.

## B.4 Pytest end-to-end smoke (in-process via FastAPI TestClient)

Because chapkit is built on FastAPI, you can drive the chapkit app in-process via Starlette's [`TestClient`](https://fastapi.tiangolo.com/tutorial/testing/) — no docker, no port, no `uvicorn` running in another shell. This repo's [`tests/test_smoke.py`](../tests/test_smoke.py) does exactly that against `example_data/`:

```python
# tests/conftest.py
import os, tempfile
from pathlib import Path
import pytest

# DATABASE_URL must be set BEFORE importing the app
os.environ.setdefault(
    "DATABASE_URL",
    f"sqlite+aiosqlite:///{tempfile.mkdtemp(prefix='chapkit_test_')}/test.db",
)

from fastapi.testclient import TestClient  # noqa: E402
from chapkit_simple_multistep_model.main import app  # noqa: E402

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
```

```python
# tests/test_smoke.py
def test_train_and_predict_against_example_data(client, example_data_dir):
    historic = pd.read_csv(example_data_dir / "historic_data.csv")
    future = pd.read_csv(example_data_dir / "future_data.csv")
    future["disease_cases"] = np.nan

    cfg = client.post("/api/v1/configs", json={...}).json()
    train = client.post("/api/v1/ml/$train", json={"config_id": cfg["id"], "data": _df_payload(historic)}).json()
    _wait_for_job(client, train["job_id"])
    # ... fetch artifact, post $predict, poll, download, assert
```

Three tests (`test_health`, `test_info`, `test_train_and_predict_against_example_data`) run in ~3s on a fresh in-memory SQLite scratch dir. Add `pytest`, `httpx` (TestClient depends on it), and any HTTP-related fixtures to your dev dep group. Wire it into a `make test` target. This is what CI runs (B.5).

Two API gotchas to handle in the test helpers:

- **NaN serialization.** Pandas `NaN` is not JSON-compliant. Convert row-by-row, replacing `NaN` with `None`, then `json.dumps(payload, allow_nan=False)`.
- **Required `prediction_periods`.** `BaseConfig` declares `prediction_periods` without a default — every config payload needs `"prediction_periods": <int>`.

### Parity check vs the original CLI (during the migration only)

While you still have the original `train.py` / `predict.py` checked in, a one-shot parity script gives you confidence the chapkit wrapper produces the same output before you delete the legacy CLI. Pattern: split a CSV into historic + future, run the original CLI on it, run the chapkit service on it (over HTTP or via TestClient), download the prediction artifact (`GET /api/v1/artifacts/{id}/$download`), compare per-cell means.

For a stochastic model with independently seeded RNGs, expect mean-of-means within ~1% and per-cell relative differences in the 5–15% range from sampling noise. Identical means across thousands of cells = same model.

Once parity is confirmed, delete the legacy CLI and the parity script — neither belongs in a pure chapkit repo long-term. (This repo went through that exercise during conversion and reported `mean-of-means within ~0.15% over 216 location/period predictions` before deleting the originals.)

## B.5 GitHub Actions CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs two jobs on push and pull request to `main`:

- **lint-and-test** — installs uv and Python 3.13, runs `make check` (ruff) and `make test` (the pytest smoke described in B.4 below). Fast, no docker.
- **docker-build** — builds the image via buildx with `load: true`, starts the container, polls `/health`, hits `/api/v1/info`, dumps logs on failure. A pure container-boot smoke; the functional verification already happened in `lint-and-test`.

This split is intentional: the expensive end-to-end check (real train + predict against `example_data/`) runs as a fast unit test via `TestClient`, and the docker job stays focused on "does the container come up and serve". You could equally well combine them — earlier iterations of this repo ran `chapkit test` against the container in CI; both shapes work.

## B.6 GHCR publish workflow

[`.github/workflows/publish-docker.yml`](../.github/workflows/publish-docker.yml) builds and pushes the image to `ghcr.io/<org>/<repo>` on pushes to `main` and on version tags. It uses `docker/metadata-action` to tag with `latest`, the short SHA, branch names, and semver.

## B.7 CHAP Core self-registration (important for live deployments)

If you are going to run this service against a real CHAP Core instance, the service should register itself on startup so CHAP Core knows about it. Without registration the service still works over HTTP — you just have to point CHAP Core at it by hand. With registration, CHAP Core picks it up automatically and keeps track of its health.

Add `.with_registration(keepalive_interval=15)` to the `MLServiceBuilder` chain (already shown in section A.3.6). Registration is controlled entirely by environment variables — when `SERVICEKIT_ORCHESTRATOR_URL` is set, the service registers on startup; when unset, registration is skipped and the service runs standalone.

What you get:

- **Automatic registration after app startup**, with retries on failure (default: 5 retries, 2s apart).
- **Hostname auto-detection** — works inside a Docker container.
- **Keepalive pings** — the service re-announces itself every 15s with a 30s TTL.
- **Auto re-registration on 404** — if chap-core's registry loses the service entry, the keepalive loop re-registers automatically.
- **Graceful deregistration on shutdown.**

Environment variables:

- `SERVICEKIT_ORCHESTRATOR_URL` — the registration endpoint URL (e.g. `http://chap:8000/v2/services/$register`). When unset, registration is skipped entirely.
- `SERVICEKIT_REGISTRATION_KEY` — the shared secret (only if CHAP Core has registration keys enabled).

## B.8 `train` and `predict` console scripts (future shell-runner readiness)

Even though this repo uses `FunctionalModelRunner`, exposing `train` and `predict` as console-script entry points (via `[project.scripts]`) costs almost nothing and makes a future swap to `ShellModelRunner` a one-line change in `main.py`. See [`src/chapkit_simple_multistep_model/cli.py`](../src/chapkit_simple_multistep_model/cli.py) for the pattern:

```toml
# pyproject.toml
[project.scripts]
train = "chapkit_simple_multistep_model.cli:train_cli"
predict = "chapkit_simple_multistep_model.cli:predict_cli"
```

```python
# src/chapkit_simple_multistep_model/cli.py
def train_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="model.pickle")
    parser.add_argument("--config", default="config.yml")
    args = parser.parse_args()

    config = _load_config(Path(args.config))   # reads chapkit-emitted config.yml
    data = ChapDataFrame.from_pandas(pd.read_csv(args.data))
    model = asyncio.run(on_train(config, data))
    Path(args.model).write_bytes(pickle.dumps(model))
```

`predict_cli()` follows the same shape with `--historic`, `--future`, `--output`. The flag names match the `ShellModelRunner` placeholder contract (`{data_file}`, `{historic_file}`, `{future_file}`, `{output_file}`) — so when you eventually want a shell-runner variant, the only change is in `main.py`:

```python
# functional (this repo)
runner = FunctionalModelRunner(on_train=on_train, on_predict=on_predict)

# shell-runner variant (later, no script rewrite needed)
runner = ShellModelRunner(
    train_command="train --data {data_file}",
    predict_command="predict --historic {historic_file} --future {future_file} --output {output_file}",
)
```

> **Heads up on naming.** `train` and `predict` are very generic console-script names — on systems with `groff` installed there's a `/opt/homebrew/bin/train` (a roff macro tool) that may shadow this repo's script. `uv run train` always resolves to the local venv, but bare `train` from a fresh shell may hit the system tool. If that bothers you, namespace the scripts (e.g. `simple-multistep-train`) — but do so in both `[project.scripts]` and your future `train_command` template.

## B.9 Documentation

- `README.md` — overview, quickstart, how to run `make test` locally, link to this guide. The scaffold generates a starter README.
- `CLAUDE.md` or equivalent contributor docs — project conventions (commit style, branch naming, code-style rules).

## B.10 Outgrow the flat scaffold layout (`src/`+`uv_build` + separate `train.py` / `predict.py`)

The flat scaffold (`main.py` at the repo root) is fine for the smallest models. As the codebase grows — model package + transformations + multiple glue modules + tests — `main.py` collects three concerns it shouldn't (model definition, training logic, prediction logic), and module imports get awkward. Two complementary improvements:

### Move to a `src/` layout with `uv_build`

Restructure so the package lives under `src/<your_package>/` and is built/installed by [`uv_build`](https://docs.astral.sh/uv/concepts/projects/build/) instead of being run as scripts at the repo root:

```
my-model/
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── compose.yml
├── compose.ghcr.yml
├── Makefile
├── README.md
├── example_data/
├── docs/
├── tests/
└── src/
    └── chapkit_simple_multistep_model/
        ├── __init__.py
        ├── __main__.py            # `python -m chapkit_simple_multistep_model` -> uvicorn
        ├── main.py                # service composition (info, hierarchy, builder)
        ├── config.py              # MultistepConfig + canonical column constants
        ├── train.py               # on_train
        ├── predict.py             # on_predict
        ├── multistep.py           # MultistepModel and friends (the algorithm)
        ├── one_step_model.py      # SkproWrapper, ResidualBootstrapModel
        └── transformations.py     # feature engineering helpers
```

`pyproject.toml` switches to `uv_build`:

```toml
[build-system]
requires = ["uv_build>=0.5,<0.12"]
build-backend = "uv_build"

[project]
name = "chapkit_simple_multistep_model"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = ["chapkit>=0.17.1", "numpy", "pandas", "xarray", "scikit-learn", "skpro"]
```

`uv_build` auto-detects `src/<package_name>/` so no extra config is required. After moving files in, run `uv lock && uv sync`. `from chapkit_simple_multistep_model.main import app` then works from anywhere — including the pytest TestClient fixture in B.4.

Inside the package, switch to relative imports (`from . import DataFrameMultistepModel`, `from .transformations import transform_data`) so the modules are decoupled from the install layout.

### Split `main.py` into `train.py` + `predict.py`

Once `on_train` and `on_predict` start growing real logic, give them their own modules — and put the `MultistepConfig` Pydantic class plus shared constants in a sibling `config.py` so neither callable depends on the other. `main.py` shrinks to pure service composition (`from .config import MultistepConfig`, `from .train import on_train`, `from .predict import on_predict`, then info / hierarchy / builder). In this repo the split lands at ~40 LOC for `config.py`, ~40 for `train.py`, ~30 for `predict.py`, ~67 for `main.py`.

### `__main__.py` for `python -m`

A 12-line `src/<your_package>/__main__.py` lets contributors and Docker boot the service with `python -m <your_package>` instead of memorizing the uvicorn invocation:

```python
import os
import uvicorn


def main() -> None:
    uvicorn.run(
        "chapkit_simple_multistep_model.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
```

The Dockerfile `CMD` then becomes a one-liner: `CMD ["python", "-m", "chapkit_simple_multistep_model"]`.

None of this is required to ship a chapkit service. Defer it until `main.py` is genuinely too big or you want pytest-driven testing (B.4) — then do the whole restructure in one commit.

---

# Migration checklist

Required — do not ship without these:

- [ ] `chapkit` CLI installed globally (`uv tool install chapkit`)
- [ ] Project scaffolded with `chapkit init <name> --template ml` from *outside* any existing chapkit project (or scaffold copied into your existing repo)
- [ ] `main.py` config class, `on_train` / `on_predict` callables, and `MLServiceInfo` customized for your model
- [ ] Inputs converted with `data.to_pandas()` / `historic.to_pandas()` / `future.to_pandas()`
- [ ] `on_predict` returns `ChapDataFrame.from_pandas(predictions)` with `sample_*` columns
- [ ] Original positional-arg CLIs removed (after a parity check confirms equivalent output)
- [ ] Scaffolded `Dockerfile` left as-is unless your model has native build deps, GPU needs, or amd64-only requirements
- [ ] `uv lock` run and `uv.lock` committed
- [ ] Service boots locally (`uv run uvicorn main:app`) and `/health` returns healthy
- [ ] At minimum a manual `POST /api/v1/configs` + `POST /api/v1/ml/$train` + `POST /api/v1/ml/$predict` cycle succeeds against the built container, OR `chapkit test` passes, OR your own pytest TestClient smoke passes

Optional — add as useful:

- [ ] Makefile (`build`, `run`, `run-ghcr`, `test`, `lint`, `check`)
- [ ] Ruff lint and format
- [ ] `compose.yml` (shipped by scaffold) and `compose.ghcr.yml`
- [ ] pytest end-to-end smoke via FastAPI TestClient (B.4)
- [ ] Parity check script vs original CLI (during the migration only)
- [ ] CI workflow (lint-and-test + docker-build)
- [ ] GHCR publish workflow
- [ ] Example data committed
- [ ] `train` / `predict` console scripts (future shell-runner readiness)
- [ ] CHAP Core self-registration (live deployments only)
- [ ] `src/` layout + `uv_build` + separate `train.py`/`predict.py` (when codebase grows)
- [ ] README and contributor docs

---

# Reference: files in this repo

The canonical worked example for every section above.

| File | What to look at |
|---|---|
| [`src/chapkit_simple_multistep_model/main.py`](../src/chapkit_simple_multistep_model/main.py) | Service composition — info, hierarchy, builder. Imports `on_train` / `on_predict` from sibling modules |
| [`src/chapkit_simple_multistep_model/__main__.py`](../src/chapkit_simple_multistep_model/__main__.py) | `python -m chapkit_simple_multistep_model` entry point (uvicorn boot) |
| [`src/chapkit_simple_multistep_model/config.py`](../src/chapkit_simple_multistep_model/config.py) | `MultistepConfig` (Pydantic) + canonical column constants — imported by both `train.py` and `predict.py` |
| [`src/chapkit_simple_multistep_model/train.py`](../src/chapkit_simple_multistep_model/train.py) | `on_train` callable |
| [`src/chapkit_simple_multistep_model/predict.py`](../src/chapkit_simple_multistep_model/predict.py) | `on_predict` callable |
| [`src/chapkit_simple_multistep_model/multistep.py`](../src/chapkit_simple_multistep_model/multistep.py), [`one_step_model.py`](../src/chapkit_simple_multistep_model/one_step_model.py) | The model algorithm itself — pure pandas / xarray, no chapkit dependency |
| [`src/chapkit_simple_multistep_model/transformations.py`](../src/chapkit_simple_multistep_model/transformations.py) | Feature-engineering helpers (lag features + one-hot location) |
| [`pyproject.toml`](../pyproject.toml) | `uv_build` build backend, chapkit + sklearn + skpro deps, src layout |
| [`Dockerfile`](../Dockerfile) | Single-stage Python 3.13 + uv image (chap-core style) — alternative to the scaffold's multi-stage default |
| [`Makefile`](../Makefile) | `build`, `run`, `run-ghcr`, `test`, `lint` (auto-fix), `check` (CI) targets |
| [`compose.yml`](../compose.yml), [`compose.ghcr.yml`](../compose.ghcr.yml) | Local and remote run recipes |
| [`example_data/`](../example_data/) | `training_data.csv`, `historic_data.csv`, `future_data.csv` ready for `$train` / `$predict` |
| [`tests/test_smoke.py`](../tests/test_smoke.py) + [`conftest.py`](../tests/conftest.py) | pytest end-to-end via FastAPI TestClient (in-process, ~3s) |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | `lint-and-test` (ruff + pytest) + `docker-build` (image boots, /health serves) |
| [`.github/workflows/publish-docker.yml`](../.github/workflows/publish-docker.yml) | GHCR publish on push to main and version tags |

---

# Appendix: manual scaffolding (when you cannot use `chapkit init`)

If for any reason you cannot run `chapkit init` (e.g. you are patching an existing repo in place), you can write the same files by hand. The scaffold is not magic — it is a small set of templates.

The minimum that gets you a working chapkit service (mirrors what the scaffold generates):

**`pyproject.toml`**

```toml
[project]
name = "your_model"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "chapkit>=0.17.1",
    "pandas>=2.3.3",
    # plus your model deps (sklearn, torch, statsmodels, ...)
]

[dependency-groups]
dev = ["uvicorn[standard]>=0.30.0"]
```

Then run `uv lock && uv sync` to generate `uv.lock` and the `.venv`.

**`main.py`**

Start from the imports, config class, callables, runner, service info, hierarchy, and builder shown in sections A.3.1 through A.3.6 above. The scaffolded `main.py` from `chapkit init` is a complete working reference — copy it and edit the fields.

**`Dockerfile`**

The scaffolded multi-stage Dockerfile is the recommended starting point. Reproducing it here would be redundant; just `chapkit init` a throwaway project to grab it. If you prefer the single-stage chap-core style, copy the Dockerfile shown in section A.5.

Everything from A.4 onward (move code, verify) applies identically. Once your codebase outgrows the flat layout, follow B.10 to move to `src/` + `uv_build` + separate `train.py` / `predict.py` (which is what *this* repo runs).

---

# Out of scope

This guide deliberately does not cover:

- Writing Python modeling code from scratch.
- Choosing an ML library — use whatever your model already uses.
- CHAP platform registration or deployment beyond building a container.
- Multi-model repositories — the template assumes one model per repo.
