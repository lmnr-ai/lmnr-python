# Laminar Python

OpenTelemetry log sender for [Laminar](https://github.com/lmnr-ai/lmnr) for Python code.

 <a href="https://pypi.org/project/lmnr/"> ![PyPI - Version](https://img.shields.io/pypi/v/lmnr?label=lmnr&logo=pypi&logoColor=3775A9) </a>
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmnr)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lmnr)



## Quickstart
```sh
python3 -m venv .myenv
source .myenv/bin/activate  # or use your favorite env management tool

pip install lmnr
```

And the in your main Python file

```python
from lmnr import Laminar as L

L.initialize(project_api_key="<LMNR_PROJECT_API_KEY>")
```

This will automatically instrument most of the LLM, Vector DB, and related
calls with OpenTelemetry-compatible instrumentation.

We rely on the amazing [OpenLLMetry](https://github.com/traceloop/openllmetry), open-source package
by TraceLoop, to achieve that.

### Project API key

Get the key from the settings page of your Laminar project ([Learn more](https://docs.lmnr.ai/api-reference/introduction#authentication)).
You can either pass it to `.initialize()` or set it to `.env` at the root of your package with the key `LMNR_PROJECT_API_KEY`.

## Instrumentation

In addition to automatic instrumentation, we provide a simple `@observe()` decorator, if you want more fine-grained tracing
or to trace other functions.

### Example

```python
import os
from openai import OpenAI


from lmnr import observe, Laminar as L
L.initialize(project_api_key="<LMNR_PROJECT_API_KEY>")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

@observe()  # annotate all functions you want to trace
def poem_writer(topic="turbulence"):
    prompt = f"write a poem about {topic}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    poem = response.choices[0].message.content
    return poem

print(poem_writer(topic="laminar flow"))
```


## Sending events

You can send events in two ways:
- `.event(name, value)` – for a pre-defined event with one of possible values.
- `.evaluate_event(name, evaluator, data)` – for an event that is evaluated by evaluator pipeline based on the data.

Note that to run an evaluate event, you need to crate an evaluator pipeline and create a target version for it.

Read our [docs](https://docs.lmnr.ai) to learn more about event types and how they are created and evaluated.

### Example

```python
from lmnr import Laminar as L
# ...
poem = response.choices[0].message.content

# this will register True or False value with Laminar
L.event("topic alignment", topic in poem)

# this will run the pipeline `check_wordy` with `poem` set as the value
# of `text_input` node, and write the result as an event with name
# "excessive_wordiness"
L.evaluate_event("excessive_wordiness", "check_wordy", {"text_input": poem})
```

## Laminar pipelines as prompt chain managers

You can create Laminar pipelines in the UI and manage chains of LLM calls there.

After you are ready to use your pipeline in your code, deploy it in Laminar by selecting the target version for the pipeline.

Once your pipeline target is set, you can call it from Python in just a few lines.

Example use:

```python
from lmnr import Laminar as L

L.initialize('<YOUR_PROJECT_API_KEY>')

result = l.run(
    pipeline = 'my_pipeline_name',
    inputs = {'input_node_name': 'some_value'},
    # all environment variables
    env = {'OPENAI_API_KEY': 'sk-some-key'},
)
```

Resulting in:

```python
>>> result
PipelineRunResponse(
    outputs={'output': {'value': [ChatMessage(role='user', content='hello')]}},
    # useful to locate your trace
    run_id='53b012d5-5759-48a6-a9c5-0011610e3669'
)
```
