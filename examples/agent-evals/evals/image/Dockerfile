FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /srv/app

# Dependencies first, so they stay cached when only the app code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
ENV PATH="/srv/app/.venv/bin:$PATH"

COPY evals/image/shared/ /opt/harness/
COPY app/ ./app/

WORKDIR /case
