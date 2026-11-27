# Agent evaluation example: services, no Dockerfile

A billing support agent for a metered API, written in TypeScript, with an [agent evaluation](https://laminar.sh/docs/agent-evaluations/introduction) dataset you run with `lmnr-cli eval run`. The agent reads the customer's plan from Postgres and their usage from ClickHouse, and can credit their account. Each case starts with its own Postgres and ClickHouse, migrated and seeded with that case's data, and a verifier checks the credits in the database and the reply.

Compared with [`../agent-evals`](../agent-evals), which builds its image from a Dockerfile and keeps its data in SQLite, this example shows:

- **An image without a Dockerfile.** `eval.py` declares it with `Image`'s methods: Node 24, the app's dependencies with `pnpm_install` (pnpm at the version in `package.json`'s `packageManager`, from `pnpm-lock.yaml`), then the code. `lmnr-cli` compiles and builds it.
- **Declared services.** `services = [Postgres(...), ClickHouse(...)]` installs both into the image and, in every trial, starts them, runs the migrations, loads the case's seed and gives the app `DATABASE_URL` / `CLICKHOUSE_URL` before anything else runs. There is no `setup()`.
- **ClickHouse the way production runs it.** Migrations are applied with the [`clickhouse-migrations`](https://www.npmjs.com/package/clickhouse-migrations) CLI (the default; `migrator="clickhouse-client"` applies plain `*.sql` files instead), and `config=` / `users=` merge production's settings over the defaults.
- **Per-case seeds the agent can't read.** Each case's rows are SQL files in its `tests/fixtures/`, loaded with `seed=case_fixture(...)`, so they never sit in the agent's container as files.
- **Grading through the services.** `verify()` queries Postgres with `self.service(Postgres).sql(...)`.

## Layout

```
package.json, pnpm-lock.yaml  # the app's dependencies, installed into the image
pnpm-workspace.yaml           # pnpm settings (which dependency build scripts may run)
src/agent.ts                  # the agent: Anthropic tool loop, instrumented with Laminar
db/migrations/                # Postgres schema, applied in natural order
clickhouse/
  migrations/1_usage_events.sql   # applied with clickhouse-migrations
  config.xml, users.xml           # production's server and user settings
evals/
  pyproject.toml              # what eval.py imports (the judge's Anthropic SDK), installed next to Harbor
  billing/                    # a dataset
    agent/
      eval.py                 # the dataset in one file: image, services, run / verify
    tasks/
      dataset.toml
      duplicate-events/       # duplicated events were billed: credit them
        environment/case.json # the ticket
        tests/
          expected.json       # the reference answer
          fixtures/
            accounts.sql      # this case's Postgres rows
            usage.sql         # this case's ClickHouse rows
      legitimate-usage/       # a busy month: no credit
```

`lmnr-cli eval describe billing` prints what `eval.py` resolves to (the image's steps and services, each with the line that declared it), and every run keeps the generated Dockerfile in its job folder.

## Run it

You need [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/), and Node.js.

1. Run every command from this directory.
2. Link it to a Laminar project. This writes `LMNR_PROJECT_API_KEY` to `.env`:

   ```bash
   npx lmnr-cli@latest setup
   ```

3. Add `ANTHROPIC_API_KEY` to `.env` (see `.env.example`).
4. Run the dataset:

   ```bash
   npx lmnr-cli@latest eval run
   ```

Each trial shows up in Laminar as a datapoint with a `credit` score and a `reply` score, next to the trace of the agent's model and tool calls. The first run pulls the Postgres, ClickHouse and Node images; later runs reuse the cached layers, and a change to `src/` rebuilds only the last one.

To add a case, copy a case folder, change its `[task] name` in `task.toml`, and edit its ticket, fixtures and expected answer.
