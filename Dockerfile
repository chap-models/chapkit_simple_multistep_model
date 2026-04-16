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

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY --chown=root:root src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev && \
    python -m compileall -q src/simple_multistep_model

RUN mkdir -p /app/data && chown chap:chap /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/health || exit 1

USER chap

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "simple_multistep_model"]
