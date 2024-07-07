# Python SDK for Laminar AI

## Quickstart
```sh
python3 -m venv .myenv
source .myenv/bin/activate  # or use your favorite env management tool

pip install lmnr
```

## Features

- Make Laminar endpoint calls from your Python code
- Make Laminar endpoint calls that can run your own functions as tools
- CLI to generate code from pipelines you build on Laminar
- `LaminarRemoteDebugger` to execute your own functions while you test your flows in workshop

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

If your pipeline contains tool call nodes, they will be able to call your local code.
If you want to test them from the Laminar workshop in your browser, you can attach to your
locally running debugger.

### Step by step instructions to use `LaminarRemoteDebugger`:

#### 1. Create your pipeline with tool call nodes

Add tool calls to your pipeline; node names must match the functions you want to call.

#### 2. Start LaminarRemoteDebugger in your code

Example:

```python
from lmnr import LaminarRemoteDebugger, NodeInput

# adding **kwargs is safer, in case an LLM produces more arguments than needed
def my_tool(arg1: string, arg2: string, **kwargs) -> NodeInput {
    return f'{arg1}&{arg2}'
}

debugger = LaminarRemoteDebugger('<YOUR_PROJECT_API_KEY>', [my_tool])
session_id = debugger.start()  # the session id will also be printed to console
```

This will establish a connection with Laminar API and allow for the pipeline execution
to call your local functions.

#### 3. Link lmnr.ai workshop to your debugger

Set up `DEBUGGER_SESSION_ID` environment variable in your pipeline.

#### 4. Run and experiment

You can run as many sessions as you need, experimenting with your flows.

#### 5. Stop the debugger

In order to stop the session, do

```python
debugger.stop()
```

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
print(f"RESULT:\n{res}")
```

### Current functionality
- Supports graph generation for graphs with Input, Output, and LLM nodes only
- For LLM nodes, it only supports OpenAI and Anthropic models and doesn't support structured output

## PROJECT_API_KEY

Read more [here](https://docs.lmnr.ai/api-reference/introduction#authentication) on how to get `PROJECT_API_KEY`.