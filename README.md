# Python SDK for Laminar AI

Example use:

```python
from lmnr import Laminar

l = Laminar('<YOUR_PROJECT_API_KEY>')
result = l.run(
    endpoint = 'my_endpoint_name',
    inputs = {'input_node_name': 'some_value'},
    env = {'OPENAI_API_KEY': 'sk-some-key'},
    metadata = {'session_id': 'your_custom_session_id'}
)
```

Resulting in:

```python
>>> result
EndpointRunResponse(outputs={'output': {'value': [ChatMessage(role='user', content='hello')]}}, run_id='53b012d5-5759-48a6-a9c5-0011610e3669')
```

## CLI for code generation

### Basic usage

```
lmnr pull <pipeline_name> <pipeline_version_name> --project-api-key <PROJECT_API_KEY>
```

Read more [here](https://docs.lmnr.ai/api-reference/introduction#authentication) on how to get `PROJECT_API_KEY`.

To import your pipeline
```python
# submodule with the name of your pipeline will be generated in lmnr_engine.pipelines
from lmnr_engine.pipelines.my_custom_pipeline import MyCustomPipeline


pipeline = MyCustomPipeline()
res = pipeline.run(
    inputs={
        "instruction": "Write me a short linked post about dev tool for LLM developers which they'll love"
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
