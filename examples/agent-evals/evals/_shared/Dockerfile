FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /srv/app

# Dependencies first, so they stay cached when only the app code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
# PYTHONPATH: `python -m app.<module>` works from any directory, so services
# and test.sh scripts don't depend on the working directory.
ENV PATH="/srv/app/.venv/bin:$PATH" PYTHONPATH=/srv/app

# Only the app: the runner uploads its scripts when it runs them, and the
# tasks' tests/ (the reference answers) never enter the image.
COPY app/ ./app/

WORKDIR /case
