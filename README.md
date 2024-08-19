# Laminar AI

This repo provides core for code generation, Laminar CLI, and Laminar SDK.

## Quickstart
```sh
python3 -m venv .myenv
source .myenv/bin/activate  # or use your favorite env management tool

pip install lmnr
```


## Automatic instrumentation example

For easy automatic instrumentation, we provide you two simple primitives:

- `observe` - a multi-purpose automatic decorator that starts traces and spans when functions are entered, and finishes them when functions return
- `wrap_llm_call` - a function that takes in your LLM call and return a "decorated" version of it. This does all the same things as `observe`, plus
a few utilities around LLM-specific things, such as counting tokens and recording model params.

You can also import `lmnr_context` in order to interact and have more control over the context of the current span.

```python
import os
from openai import OpenAI

from lmnr import observe, wrap_llm_call, lmnr_context
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

@observe()  # annotate all functions you want to trace
def poem_writer(topic="turbulence"):
    prompt = f"write a poem about {topic}"

    # wrap the actual final call to LLM with `wrap_llm_call`
    response = wrap_llm_call(client.chat.completions.create)(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    poem = response.choices[0].message.content

    if topic in poem:
        lmnr_context.event("topic_alignment")  # send an event with a pre-defined name
    
    # to trigger an automatic check for a possible event do:
    lmnr_context.check_span_event("excessive_wordiness")

    return poem

if __name__ == "__main__":
    print(poem_writer(topic="laminar flow"))
```

This gives an advantage of super quick instrumentation, but is somewhat limited in flexibility + doesn't really work as expected with threading.
This is due to the fact that we use `contextvars.ContextVar` for this and how Python manages them between threads.

If you want to instrument your code manually, follow on to the next section

## Manual instrumentation example

For manual instrumetation you will need to import the following:
- `trace` - this is a function to start a trace. It returns a `TraceContext`
- `TraceContext` - a pointer to the current trace that you can pass around functions as you want.
- `SpanContext` - a pointer to the current span that you can pass around functions as you want

Both `TraceContext` and `SpanContext` expose the following interfaces:
- `span(name: str, **kwargs)` - create a child span within the current context. Returns `SpanContext`
- `update(**kwargs)` - update the current trace or span and return it. Returns `TraceContext` or `SpanContext`. Useful when some metadata becomes known later during the program execution
- `end(**kwargs)` – update the current span, and terminate it

In addition, `SpanContext` allows you to:
- `event()` - emit a custom event at any point
- `check_span_event()` - register a possible event for automatic checking by Laminar.

Example:

```python
import os
from openai import OpenAI

from lmnr import trace, TraceContext, SpanContext
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def poem_writer(t: TraceContext, topic = "turbulence"):
    span: SpanContext = t.span(name="poem_writer", input=None)

    prompt = f"write a poem about {topic}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    # create a child span within the current `poem_writer` span.
    llm_span = span.span(name="OpenAI completion", input=messages, span_type="LLM")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello. What is the capital of France?"},
        ],
    )
    poem = response.choices[0].message.content
    if topic in poem:
        llm_span.event("topic_alignment")  # send an event with a pre-defined name

    # note that you can register possible events here as well, not only `llm_span.check_span_event()`
    llm_span.end(output=poem, check_event_names=["excessive_wordiness"])
    span.end(output=poem)
    return poem


t: TraceContext = trace(user_id="user", session_id="session", release="release")
main(t, topic="laminar flow")
t.end(success=True)

```

## Features

- Make Laminar endpoint calls from your Python code
- Make Laminar endpoint calls that can run your own functions as tools
- CLI to generate code from pipelines you build on Laminar or execute your own functions while you test your flows in workshop

## Making Laminar endpoint calls

After you are ready to use your pipeline in your code, deploy it in Laminar following the [docs](https://docs.lmnr.ai/pipeline/run-save-deploy#deploying-a-pipeline-version).

Once your pipeline is deployed, you can call it from Python in just a few lines.

Example use:

```python
from lmnr import Laminar

l = Laminar('<YOUR_PROJECT_API_KEY>')
result = l.run(
    endpoint = 'my_endpoint_name',
    inputs = {'input_node_name': 'some_value'},
    # all environment variables
    env = {'OPENAI_API_KEY': 'sk-some-key'},
    # any metadata to attach to this run's trace
    metadata = {'session_id': 'your_custom_session_id'}
)
```

Resulting in:

```python
>>> result
EndpointRunResponse(
    outputs={'output': {'value': [ChatMessage(role='user', content='hello')]}},
    # useful to locate your trace
    run_id='53b012d5-5759-48a6-a9c5-0011610e3669'
)
```

## Making calls to pipelines that run your own logic

If your pipeline contains tool call nodes, they will be able to call your local code.
The only difference is that you need to pass references
to the functions you want to call right into our SDK.

Example use:

```python
from lmnr import Laminar, NodeInput

# adding **kwargs is safer, in case an LLM produces more arguments than needed
def my_tool(arg1: string, arg2: string, **kwargs) -> NodeInput {
    return f'{arg1}&{arg2}'
}

l = Laminar('<YOUR_PROJECT_API_KEY>')
result = l.run(
    endpoint = 'my_endpoint_name',
    inputs = {'input_node_name': 'some_value'},
    # all environment variables
    env = {'OPENAI_API_KEY': '<YOUR_MODEL_PROVIDER_KEY>'},
    # any metadata to attach to this run's trace
    metadata = {'session_id': 'your_custom_session_id'},
    # specify as many tools as needed.
    # Each tool name must match tool node name in the pipeline
    tools=[my_tool]
)
```

## LaminarRemoteDebugger

If your pipeline contains local call nodes, they will be able to call code right on your machine.

### Step by step instructions to connect to Laminar:

#### 1. Create your pipeline with function call nodes

Add function calls to your pipeline; these are signature definitions of your functions

#### 2. Implement the functions

At the root level, create a file: `pipeline.py`

Annotate functions with the same name.

Example:

```python
from lmnr import Pipeline

lmnr = Pipeline()

@lmnr.func("foo") # the node in the pipeline is called foo and has one parameter arg
def custom_logic(arg: str) -> str:
    return arg * 10
```

#### 3. Link lmnr.ai workshop to your machine

1. At the root level, create a `.env` file if not already
1. In project settings, create or copy a project api key.
1. Add an entry in `.env` with: `LMNR_PROJECT_API_KEY=s0meKey...`
1. In project settings create or copy a dev session. These are your individual sessions.
1. Add an entry in `.env` with: `LMNR_DEV_SESSION_ID=01234567-89ab-cdef-0123-4567890ab`

#### 4. Run the dev environment

```sh
lmnr dev
```

This will start a session, try to persist it, and reload the session on files change.

## CLI for code generation

### Basic usage

```
lmnr pull <pipeline_name> <pipeline_version_name> --project-api-key <PROJECT_API_KEY>
```

Note that `lmnr` CLI command will only be available from within the virtual environment
where you have installed the package.

To import your pipeline
```python
# submodule with the name of your pipeline will be generated in lmnr_engine.pipelines
from lmnr_engine.pipelines.my_custom_pipeline import MyCustomPipeline


pipeline = MyCustomPipeline()
res = pipeline.run(
    inputs={
        "instruction": "Write me a short linkedin post about a dev tool for LLM developers"
    },
    env={
        "OPENAI_API_KEY": <OPENAI_API_KEY>,
    }
)
print(f"Pipeline run result:\n{res}")
```

### Current functionality
- Supports graph generation for graphs with the following nodes: Input, Output, LLM, Router, Code.
- For LLM nodes, it only supports OpenAI and Anthropic models. Structured output in LLM nodes will be supported soon.

## PROJECT_API_KEY

Read more [here](https://docs.lmnr.ai/api-reference/introduction#authentication) on how to get `PROJECT_API_KEY`.
