# syntax=docker/dockerfile:1
ARG PYTHON=3.13
FROM python:${PYTHON}-slim-bookworm AS deps
RUN pip install --no-cache-dir httpx

FROM python:${PYTHON}-slim-bookworm
COPY --from=deps /usr/local/lib /usr/local/lib
RUN useradd -m app
COPY app/ /srv/app/app/
USER app
WORKDIR /srv/app
