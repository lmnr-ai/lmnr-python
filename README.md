# Laminar Python

Python SDK for [Laminar](https://www.lmnr.ai).

[Laminar](https://www.lmnr.ai) is an open-source platform for engineering LLM products. Trace, evaluate, annotate, and analyze LLM data. Bring LLM applications to production with confidence.

Check our [open-source repo](https://github.com/lmnr-ai/lmnr) and don't forget to star it ⭐

 <a href="https://pypi.org/project/lmnr/"> ![PyPI - Version](https://img.shields.io/pypi/v/lmnr?label=lmnr&logo=pypi&logoColor=3775A9) </a>
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmnr)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lmnr)


## Quickstart

First, install the package, specifying the instrumentations you want to use.

For example, to install the package with OpenAI and Anthropic instrumentations:

```sh
pip install 'lmnr[anthropic,openai]'
```

To install all possible instrumentations, use the following command:

```sh
pip install 'lmnr[all]'
```

Initialize Laminar in your code:

```python
from lmnr import Laminar

Laminar.initialize(project_api_key="<PROJECT_API_KEY>")
```

You can also skip passing the `project_api_key`, in which case it will be looked
in the environment (or local .env file) by the key `LMNR_PROJECT_API_KEY`.

Note that you need to only initialize Laminar once in your application. You should
try to do that as early as possible in your application, e.g. at server startup.

## Set-up for self-hosting

If you self-host a Laminar instance, the default connection settings to it are
`http://localhost:8000` for HTTP and `http://localhost:8001` for gRPC. Initialize
the SDK accordingly:

```python
from lmnr import Laminar

Laminar.initialize(
    project_api_key="<PROJECT_API_KEY>",
    base_url="http://localhost",
    http_port=8000,
    grpc_port=8001,
)
```

## Instrumentation

### Manual instrumentation

To instrument any function in your code, we provide a simple `@observe()` decorator.
This can be useful if you want to trace a request handler or a function which combines multiple LLM calls.

```python
import os
from openai import OpenAI
from lmnr import Laminar

Laminar.initialize(project_api_key=os.environ["LMNR_PROJECT_API_KEY"])

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def poem_writer(topic: str):
    prompt = f"write a poem about {topic}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    # OpenAI calls are still automatically instrumented
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    poem = response.choices[0].message.content

    return poem

@observe()
def generate_poems():
    poem1 = poem_writer(topic="laminar flow")
    poem2 = poem_writer(topic="turbulence")
    poems = f"{poem1}\n\n---\n\n{poem2}"
    return poems
```

Also, you can use `Laminar.start_as_current_span` if you want to record a chunk of your code using `with` statement.

```python
def handle_user_request(topic: str):
    with Laminar.start_as_current_span(name="poem_writer", input=topic):
        poem = poem_writer(topic=topic)
        # Use set_span_output to record the output of the span
        Laminar.set_span_output(poem)
```

### Automatic instrumentation

Laminar allows you to automatically instrument majority of the most popular LLM, Vector DB, database, requests, and other libraries.

If you want to automatically instrument a default set of libraries, then simply do NOT pass `instruments` argument to `.initialize()`.
See the full list of available instrumentations in the [enum](https://github.com/lmnr-ai/lmnr-python/blob/main/src/lmnr/opentelemetry_lib/instruments.py).

If you want to automatically instrument only specific LLM, Vector DB, or other
calls with OpenTelemetry-compatible instrumentation, then pass the appropriate instruments to `.initialize()`.
For example, if you want to only instrument OpenAI and Anthropic, then do the following:

```python
from lmnr import Laminar, Instruments

Laminar.initialize(project_api_key=os.environ["LMNR_PROJECT_API_KEY"], instruments={Instruments.OPENAI, Instruments.ANTHROPIC})
```

If you want to fully disable any kind of autoinstrumentation, pass an empty set as `instruments=set()` to `.initialize()`. 

Autoinstrumentations are provided by Traceloop's [OpenLLMetry](https://github.com/traceloop/openllmetry).

## Evaluations

### Quickstart

Install the package:

```sh
pip install lmnr
```

Create a file named `my_first_eval.py` with the following code:

```python
from lmnr import evaluate

def write_poem(data):
    return f"This is a good poem about {data['topic']}"

def contains_poem(output, target):
    return 1 if output in target['poem'] else 0

# Evaluation data
data = [
    {"data": {"topic": "flowers"}, "target": {"poem": "This is a good poem about flowers"}},
    {"data": {"topic": "cars"}, "target": {"poem": "I like cars"}},
]

evaluate(
    data=data,
    executor=write_poem,
    evaluators={
        "containsPoem": contains_poem
    },
    group_id="my_first_feature"
)
```

Run the following commands:

```sh
export LMNR_PROJECT_API_KEY=<YOUR_PROJECT_API_KEY>  # get from Laminar project settings
lmnr eval my_first_eval.py  # run in the virtual environment where lmnr is installed
```

Visit the URL printed in the console to see the results.

### Overview

Bring rigor to the development of your LLM applications with evaluations.

You can run evaluations locally by providing executor (part of the logic used in your application) and evaluators (numeric scoring functions) to `evaluate` function.

`evaluate` takes in the following parameters:
- `data` – an array of `EvaluationDatapoint` objects, where each `EvaluationDatapoint` has two keys: `target` and `data`, each containing a key-value object. Alternatively, you can pass in dictionaries, and we will instantiate `EvaluationDatapoint`s with pydantic if possible
- `executor` – the logic you want to evaluate. This function must take `data` as the first argument, and produce any output. It can be both a function or an `async` function.
- `evaluators` – Dictionary which maps evaluator names to evaluators. Functions that take output of executor as the first argument, `target` as the second argument and produce a numeric scores. Each function can produce either a single number or `dict[str, int|float]` of scores. Each evaluator can be both a function or an `async` function.
- `name` – optional name for the evaluation. Automatically generated if not provided.
- `group_id` – optional group name for the evaluation. Evaluations within the same group can be compared visually side-by-side

\* If you already have the outputs of executors you want to evaluate, you can specify the executor as an identity function, that takes in `data` and returns only needed value(s) from it.

Read the [docs](https://docs.lmnr.ai/evaluations/introduction) to learn more about evaluations.

## Client for HTTP operations

Various interactions with Laminar [API](https://docs.lmnr.ai/api-reference/) are available in `LaminarClient`
and its asynchronous version `AsyncLaminarClient`.

### Agent

To run Laminar agent, you can invoke `client.agent.run`

```python
from lmnr import LaminarClient

client = LaminarClient(project_api_key="<YOUR_PROJECT_API_KEY>")

response = client.agent.run(
    prompt="What is the weather in London today?"
)

print(response.result.content)
```

#### Streaming

Agent run supports streaming as well.

```python
from lmnr import LaminarClient

client = LaminarClient(project_api_key="<YOUR_PROJECT_API_KEY>")

for chunk in client.agent.run(
    prompt="What is the weather in London today?",
    stream=True
):
    if chunk.chunk_type == 'step':
        print(chunk.summary)
    elif chunk.chunk_type == 'finalOutput':
        print(chunk.content.result.content)
```

#### Async mode

```python
from lmnr import AsyncLaminarClient

client = AsyncLaminarClient(project_api_key="<YOUR_PROJECT_API_KEY>")

response = await client.agent.run(
    prompt="What is the weather in London today?"
)

print(response.result.content)
```

#### Async mode with streaming

```python
from lmnr import AsyncLaminarClient

client = AsyncLaminarClient(project_api_key="<YOUR_PROJECT_API_KEY>")

# Note that you need to await the operation even though we use `async for` below
response = await client.agent.run(
    prompt="What is the weather in London today?",
    stream=True
)
async for chunk in client.agent.run(
    prompt="What is the weather in London today?",
    stream=True
):
    if chunk.chunk_type == 'step':
        print(chunk.summary)
    elif chunk.chunk_type == 'finalOutput':
        print(chunk.content.result.content)
```
