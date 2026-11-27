FROM node:24-trixie-slim

WORKDIR /srv/app
RUN npm install -g pnpm@12.23.0
# Dependencies first, so they stay cached when only the app code changes.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile --store-dir /tmp/pnpm-store && rm -rf /tmp/pnpm-store
COPY src/ ./src/
# Where run_command writes the reply (a declared artifact).
RUN mkdir -p /srv/app/out

WORKDIR /case
