# Agent evaluation example

A billing support agent for a metered API, written in TypeScript, with [agent evaluation](https://laminar.sh/docs/agent-evaluations/introduction) datasets you run with `lmnr-cli eval run`. Customers ask about their bill in Slack. The agent reads the customer's plan from Postgres and their usage from ClickHouse, checks recent incidents, can credit the account, and answers in the customer's thread.

Two datasets test it:

- **`billing`** is the recommended setup, and the one to copy. Each case starts its own Postgres, ClickHouse and a simulated Slack workspace, loaded with that case's data, and mocks the incidents service. The verifier checks the credits in the database, what the agent posted to Slack, and the reply's content.
- **`billing-alternatives`** runs the same agent the other ways `lmnr.evals` supports: an existing Dockerfile, an HTTP mock, seeding in `setup()`, one shell command instead of `run()`, and grading by each case's own `tests/test.sh` in a separate container. Use these when the recommended way doesn't fit.

## Layout

```
package.json, pnpm-lock.yaml  # the app's dependencies, installed into the image
pnpm-workspace.yaml           # pnpm settings (which dependency build scripts may run)
src/
  agent.ts                    # the agent: Anthropic tool loop, instrumented with Laminar
  slack.ts                    # reads the customer's message, replies in its thread
  incidents.ts                # observed call to the status service (mocked in evals)
db/migrations/                # Postgres schema, applied in natural order
clickhouse/
  migrations/1_usage_events.sql   # applied with clickhouse-migrations
  config.xml, users.xml           # production's server and user settings
evals/
  pyproject.toml              # what eval.py imports (the judge's Anthropic SDK), installed next to Harbor
  billing/                    # the recommended dataset
    agent/
      eval.py                 # the dataset in one file: image, services, mocks, run / verify
    tasks/
      dataset.toml            # the dataset's name in Laminar, and its task versions
      duplicate-events/       # duplicated events were billed: credit them
        task.toml             # [task] name, the image, timeouts
        instruction.md
        tests/
          test.sh             # a stub: eval.py's verify() grades the case
          expected.json       # the reference answer
          fixtures/           # this case's data, never visible to the agent as files
            accounts.sql      # Postgres rows
            usage.sql         # ClickHouse rows
            slack.json        # the Slack workspace: the customer, #billing-help, their message
            incidents.json    # what the incidents mock answers
      legitimate-usage/       # a busy month: no credit
  billing-alternatives/       # the other ways, same agent
    agent/
      eval.py
      Dockerfile
    tasks/duplicate-events/   # environment/ticket.json, tests/test.sh, tests/fixtures/
```

## The `billing` dataset

`evals/billing/agent/eval.py` holds one `EvalRunner` (from `lmnr.evals`):

- **`image`**, without a Dockerfile: Node 24, the app's dependencies with `pnpm_install` (pnpm at the version in `package.json`'s `packageManager`, from `pnpm-lock.yaml`), then the code. `lmnr-cli` compiles and builds it.
- **`services`**: in every trial, before anything else runs, the runner starts each one, runs its migrations and loads the case's seed from `tests/fixtures/` (`seed=case_fixture(...)`), and gives the app its URL.
  - `Postgres` and `ClickHouse` are the app's own databases. ClickHouse is migrated with the [`clickhouse-migrations`](https://www.npmjs.com/package/clickhouse-migrations) CLI (the default), and `config=` / `users=` merge production's settings over the defaults.
  - `Rystic("slack", ...)` is a [Rystic](https://www.rystic.ai) Slack simulator. The case's `slack.json` seeds the workspace with the customer's message, and the app talks to it with the official Slack SDK through `SLACK_API_URL`.
- **`mocks`**: `list_incidents` answers the app's observed `incidents.list_recent` (`src/incidents.ts`) from the case's `incidents.json`, so the status service doesn't have to run. The handler runs in the container, in a mock server, and the app's Laminar SDK asks that server before running the function. Function mocks need `@lmnr-ai/lmnr` 0.9.8 or later in the app; with an older SDK the real function runs and the tool reports an error.
- **`run`** runs the agent (`node src/agent.ts slack`).
- **`verify`** runs in the process that runs Harbor, not in the container. It reads the credits from Postgres (`self.service(Postgres).sql(...)`), the messages the agent posted from the Slack simulator's call log (`self.mirror("slack").calls(...)`), and asks an LLM judge (a plain function calling the Anthropic SDK, listed in `evals/pyproject.toml`) whether the reply meets `expected.json`'s `reply_should`. The reference answer never enters the container.

`lmnr-cli` builds the image, writes the job config Harbor needs, and runs every case through Harbor. There is no `agent.py`, verifier script or `harbor.yaml`. `npx lmnr-cli@latest eval describe billing` prints what `eval.py` resolves to: the image's steps and services, each with the line that declared it, and the mocks.

## The `billing-alternatives` dataset

Each choice in `evals/billing-alternatives/agent/eval.py` replaces one from `billing`:

| `billing` (recommended) | `billing-alternatives` | When to use it |
| -- | -- | -- |
| `Image.node(...)` builder | `Image.from_dockerfile("Dockerfile", context=...)` | you already have a Dockerfile |
| function mock of an observed function | `mock.http(handler, base_url_env="INCIDENTS_API_URL")` | the call isn't an observed function, or the app doesn't use Laminar's SDK |
| `seed=case_fixture(...)` on each service | `setup()` loads the rows with `load_sql` / `execute` | the rows need computing, or come from several files |
| `run()` | `run_command = "node ... > /srv/app/out/reply.txt"` | one shell command is enough (it gets no `ctx`, so no `--model`) |
| `verify()` | `grade_with_test_sh = True`: each case's `tests/test.sh` writes the rewards | imported Harbor datasets, or cases that each check something different |
| grading next to the live services | `[verifier] environment_mode = "separate"`, `Postgres(snapshot=["credits"])`, `artifacts = [Artifact(...)]` | grading must not see anything the run left besides what you collected |

Its ticket comes from `environment/ticket.json` instead of Slack, so it runs without a Rystic key.

## Run it

You need [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/), and Node.js.

1. Run every command from this directory.
2. Link it to a Laminar project. This writes `LMNR_PROJECT_API_KEY` to `.env`:

   ```bash
   npx lmnr-cli@latest setup
   ```

3. Add `ANTHROPIC_API_KEY` to `.env`, and for the `billing` dataset `RYSTIC_TOKEN`, your [Rystic API key](https://www.rystic.ai/account/api-keys) (see `.env.example`). The Slack simulator is on Rystic's free plan.
4. Run the datasets:

   ```bash
   npx lmnr-cli@latest eval run --env RYSTIC_TOKEN
   ```

   `--env RYSTIC_TOKEN` passes your key into each trial, where the simulator starts. Remote runs (`eval run --remote`) don't need it: Laminar gives each trial its own Rystic access. To run one dataset, name it: `eval run billing-alternatives`.

Each trial shows up in Laminar as a datapoint with its scores (`credit`, `reply` and `in_thread` for `billing`), next to the trace of the agent's model and tool calls. The mocked `incidents.list_recent` call is marked as mocked. The first run pulls the Postgres, ClickHouse and Node images; later runs reuse the cached layers, and a change to `src/` rebuilds only the last one. Each trial's folder in `evals/<dataset>/jobs/` keeps the services' logs (`agent/`) and what was collected for grading (`artifacts/`, including `mocks.jsonl`, every call the mocks answered).

To add a case to `billing`, create a folder in `evals/billing/tasks/` with its `tests/expected.json` and `tests/fixtures/`. The next `eval run` writes the rest (`task.toml` with the case's name, `instruction.md`, and the stub `tests/test.sh`), or create them first with `npx lmnr-cli@latest eval task new evals/billing/tasks/<name>`.

## Store the cases in Laminar

The case folders are committed here so the example runs as it is. In your own project they live in Laminar instead, versioned by content, and git only keeps `tasks/dataset.toml`:

```bash
npx lmnr-cli@latest eval dataset push
```

This uploads each case, pins its version in `tasks/dataset.toml`, and adds `*/tasks/*` (except `dataset.toml`) to `evals/.gitignore`. Teammates get the pinned cases with `npx lmnr-cli@latest eval dataset pull` before running them: `eval run` runs the cases in `tasks/` as they are and never pulls. The project's **Eval datasets** page lists each case and its versions.
