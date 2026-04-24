FROM ghcr.io/dhis2-chap/chapkit-py:latest

ENV MPLCONFIGDIR=/tmp
ENV PORT=8000

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    useradd --no-create-home --shell /usr/sbin/nologin chap

WORKDIR /app

COPY --chown=root:root pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY --chown=root:root src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev && \
    python -m compileall -q src/chapkit_simple_multistep_model

RUN mkdir -p /app/data && chown chap:chap /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/health || exit 1

USER chap

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "chapkit_simple_multistep_model"]
