# Agent evaluation example

A support agent that handles billing tickets, with [agent evaluation](https://laminar.sh/docs/agent-evaluations/introduction) datasets you run with `lmnr-cli eval run`. The agent looks up a customer's charges in a SQLite database and can refund one. Each case starts with its own copy of the database, and a verifier checks both the database and the reply. In the `support` dataset the agent also posts its reply through a helpdesk API: each case mocks the CRM and the helpdesk, and the verifier checks what the agent sent to the helpdesk.

The [quickstart](https://laminar.sh/docs/agent-evaluations/quickstart#get-started-manually) walks through the files.

## Layout

```
app/
  support_agent.py            # the agent: Anthropic tool loop, instrumented with Laminar
  ticket_server.py            # the same agent behind an HTTP API (used by support-api)
pyproject.toml, uv.lock       # the app's dependencies, installed into the image
evals/
  pyproject.toml              # what eval.py imports (the judge's Anthropic SDK), installed next to Harbor
  support/                    # a dataset
    agent/
      eval.py                 # the dataset in one file: image, secrets, mocks, setup / run / verify
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
          fixtures/
            crm.json          # what the CRM mock answers in this case
  _shared/                    # shared runner code, not a dataset (see "More datasets")
  support-api/                # the agent behind a service, graded in a separate container
  support-test-sh/            # cases graded by their own tests/test.sh
```

`eval.py` holds one `EvalRunner` (from `lmnr.evals`):

- `mocks` replaces two production services. `helpdesk` is an HTTP mock: the runner points the app's `HELPDESK_API_URL` at a mock server in the container, and the handler answers `POST .../replies`. `get_customer` mocks the app's observed `crm.get_customer` function (`@observe(name="crm.get_customer")`) and reads the case's `tests/fixtures/crm.json`: the app's Laminar SDK asks the mock server before running the function and returns the handler's value instead. This needs `lmnr` 0.9.3 or later in the app; with an older SDK the function runs and calls `CRM_API_URL`, and the tool reports the CRM as unavailable. The handlers run in the container, in the mock server, and are never visible to the agent as files.
- `setup` uploads `seed.py` into the container and runs it, so this case's database exists before the agent starts.
- `run` calls the agent on the case's ticket and records its reply with `ctx.set_result(...)`.
- `verify` runs in the process that runs Harbor, not in the container. It checks the refunds in the database (a command in the container, `ctx.container.sh`), asks an LLM judge (a plain function in `eval.py` calling the Anthropic SDK, listed in `evals/pyproject.toml`) whether the reply meets `expected.json`'s `reply_should`, and reads the mocks' call log (`ctx.mocks.calls(...)`) to check the reply was posted once. The reference answer never enters the container.

`lmnr-cli` builds the image, writes the job config Harbor needs, and runs every case through Harbor. There is no `agent.py`, verifier script or `harbor.yaml`.

## Run it

You need [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/), and Node.js.

1. Run every command from this directory. `lmnr-cli` looks for `evals/` and `.env` in the current directory.
2. Link it to a Laminar project. This writes `LMNR_PROJECT_API_KEY` to `.env`:

   ```bash
   npx lmnr-cli@latest setup
   ```

3. Add `ANTHROPIC_API_KEY` to `.env` (see `.env.example`).
4. Run the `support` dataset:

   ```bash
   npx lmnr-cli@latest eval run support
   ```

   Without a name, `eval run` runs every dataset in `evals/`, including the two in "More datasets".

Each trial shows up in Laminar as a datapoint with `refund`, `reply` and `posted` scores, next to the trace of the agent's model and tool calls. The trial's `artifacts/` folder in `evals/support/jobs/` holds `mocks.jsonl`, every call the mocks answered.

To add a case, create a folder in `evals/support/tasks/` with its `environment/case.json`, `environment/seed.json`, `tests/expected.json` and `tests/fixtures/crm.json`. The next `eval run` writes the rest (`task.toml` with the case's name, `instruction.md`, and the stub `tests/test.sh`), or create them first with `npx lmnr-cli@latest eval task new evals/support/tasks/<name>`.

## More datasets

`support` keeps everything in one `eval.py`. The other two datasets share a runner and show the rest of `lmnr.evals`:

```
evals/
  _shared/                    # not a dataset: no agent/eval.py. On the import path of every eval.py
    billing.py                # BillingRunner: image, model, secrets, setup (seed), submit_ticket()
    Dockerfile                # the image both datasets build (each as <folder>-evals)
    seed.py                   # creates the case's database (run by BillingRunner.setup)
    submit_ticket.py          # queues the case's ticket with the ticket service
    judge.py                  # the LLM judge, imported by support-api's verify()
  support-api/
    agent/eval.py             # class SupportApiEval(BillingRunner)
    tasks/triple-charge/      # three identical charges: two refunds expected
  support-test-sh/
    agent/eval.py             # class SupportTestShEval(BillingRunner): four lines
    tasks/different-products/ # tests/test.sh: no refund is owed
    tasks/other-customers-charge/  # tests/test.sh: another customer's charge stays unrefunded
```

- **Shared runner.** Each `eval.py` subclasses `BillingRunner` from `evals/_shared/billing.py`. A relative path resolves against the file whose code uses it: `Image.from_dockerfile("Dockerfile")` and `ctx.container.run_file("seed.py")` in `billing.py` find the files next to it, whichever dataset runs. That's why `submit_ticket()` is a `BillingRunner` method: called from a dataset's `eval.py`, `run_file("submit_ticket.py")` would look in that dataset's `agent/` folder.
- **`support-api`** runs the agent the way it runs in production, behind `app/ticket_server.py`:
  - `setup` starts it with `ctx.container.service(..., ready=http(...))`, which returns once `/health` answers. Its output goes to the trial's `agent/api.log`.
  - `run` queues the ticket, then `ctx.container.wait_for(cmd("test -f ..."))` until the service writes the ticket's record, and reads it with `ctx.container.read_json`.
  - `collect` saves the database with `ctx.artifacts.save("/srv/app/app.db")`. `collect` runs after `run`, also when it failed.
  - `artifacts = [Artifact("/srv/app/audit", destination="audit")]` collects the service's records too.
  - The task sets `[verifier] environment_mode = "separate"`, so grading gets a fresh container that sees only the artifacts. `verify` doesn't use the container at all: it opens the saved database with `sqlite3` on the host (`ctx.artifacts.path(...)`), reads the record (`ctx.artifacts.read_json(...)`), and sends the ticket (`ctx.case.file("case.json")`, read on the host) and the reply to the shared judge.
- **`support-test-sh`** grades like an imported Harbor dataset:
  - `run_command` replaces `run()`. A plain command gets no `ctx`, so the app uses its default model.
  - `grade_with_test_sh = True` replaces `verify()`: Harbor runs each case's own `tests/test.sh` in the container after the agent, and the script writes `/logs/verifier/reward.txt`. Each case checks something different without a shared `verify()`.

## Store the cases in Laminar

The case folders are committed here so the example runs as it is. In your own project they live in Laminar instead, versioned by content, and git only keeps `tasks/dataset.toml`:

```bash
npx lmnr-cli@latest eval dataset push
```

This uploads each case, pins its version in `tasks/dataset.toml`, and adds `*/tasks/*` (except `dataset.toml`) to `evals/.gitignore`. Teammates get the pinned cases with `npx lmnr-cli@latest eval dataset pull` before running them: `eval run` runs the cases in `tasks/` as they are and never pulls. The project's **Eval datasets** page lists each case and its versions.
