# Agent evaluation example

A support agent that handles billing tickets, with an [agent evaluation](https://laminar.sh/docs/agent-evaluations/introduction) dataset you run with `lmnr-cli eval run`. The agent looks up a customer's charges in a SQLite database and can refund one. Each case starts with its own copy of the database, and a verifier checks both the database and the reply.

The [quickstart](https://laminar.sh/docs/agent-evaluations/quickstart#get-started-manually) walks through the files.

## Layout

```
app/
  support_agent.py            # the agent: Anthropic tool loop, instrumented with Laminar
pyproject.toml, uv.lock       # the app's dependencies, installed into the image
evals/
  pyproject.toml              # what eval.py imports (the judge's Anthropic SDK), installed next to Harbor
  support/                    # a dataset
    agent/
      eval.py                 # the dataset in one file: image, secrets, setup / run / verify
      Dockerfile              # the image every case starts from
      seed.py                 # creates this case's database from its seed.json (eval.py runs it)
    tasks/
      dataset.toml            # the dataset's name in Laminar, and its task versions
      duplicate-charge/       # a case
        task.toml             # [task] name, the image, timeouts
        instruction.md
        environment/
          case.json           # the ticket
          seed.json           # the database rows for this case
        tests/
          test.sh             # a stub: eval.py's verify() grades the case
          expected.json       # the reference answer
```

`eval.py` holds one `EvalRunner` (from `lmnr.evals`):

- `setup` uploads `seed.py` into the container and runs it, so this case's database exists before the agent starts.
- `run` calls the agent on the case's ticket and records its reply with `ctx.result(...)`.
- `verify` runs in the process that runs Harbor, not in the container. It checks the refunds in the database (a command in the container) and asks an LLM judge (a plain function in `eval.py` calling the Anthropic SDK, listed in `evals/pyproject.toml`) whether the reply meets `expected.json`'s `reply_should`. The reference answer never enters the container.

`lmnr-cli` builds the image, writes the job config Harbor needs, and runs every case through Harbor. There is no `agent.py`, verifier script or `harbor.yaml`.

## Run it

You need [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/), and Node.js.

1. Run every command from this directory. `lmnr-cli` looks for `evals/` and `.env` in the current directory.
2. Link it to a Laminar project. This writes `LMNR_PROJECT_API_KEY` to `.env`:

   ```bash
   npx lmnr-cli@latest setup
   ```

3. Add `ANTHROPIC_API_KEY` to `.env` (see `.env.example`).
4. Run the dataset:

   ```bash
   npx lmnr-cli@latest eval run
   ```

Each trial shows up in Laminar as a datapoint with a `refund` score and a `reply` score, next to the trace of the agent's model and tool calls.

To add a case, create a folder in `evals/support/tasks/` with its `environment/case.json`, `environment/seed.json` and `tests/expected.json`. The next `eval run` writes the rest (`task.toml` with the case's name, `instruction.md`, and the stub `tests/test.sh`), or create them first with `npx lmnr-cli@latest eval task new evals/support/tasks/<name>`.

## Store the cases in Laminar

The case folders are committed here so the example runs as it is. In your own project they live in Laminar instead, versioned by content, and git only keeps `tasks/dataset.toml`:

```bash
npx lmnr-cli@latest eval dataset push
```

This uploads each case, pins its version in `tasks/dataset.toml`, and adds `*/tasks/*` (except `dataset.toml`) to `evals/.gitignore`. Teammates get the pinned cases with `npx lmnr-cli@latest eval dataset pull` before running them: `eval run` runs the cases in `tasks/` as they are and never pulls. The project's **Eval datasets** page lists each case and its versions.
