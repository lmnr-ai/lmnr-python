# Laminar AI

This repo provides core for code generation, Laminar CLI, and Laminar SDK.

## Quickstart
```sh
python3 -m venv .myenv
source .myenv/bin/activate  # or use your favorite env management tool

pip install lmnr
```

Create .env file at the root and add `LMNR_PROJECT_API_KEY` value to it.

Read more [here](https://docs.lmnr.ai/api-reference/introduction#authentication) on how to get `LMNR_PROJECT_API_KEY`.

## Sending events

You can send events in two ways:
- `.event(name, value)` – for a pre-defined event with one of possible values.
- `.evaluate_event(name, data)` – for an event that our agent checks for and assigns a value from possible values.

There are 3 types of events:
- Number - Numeric value.
- String - Arbitrary string.
- Boolean - Convenient to classify if something has took place or not.

Important notes:
- If event name does not match anything pre-defined in the UI, the event won't be saved.

## Instrumentation

We provide two ways to instrument your python code:
- With `@observe()` decorators and `wrap_llm_call` helpers
- Manually

It is important to not mix the two styles of instrumentation, this can lead to unpredictable results.

## Decorator instrumentation example

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
        # send an event with a pre-defined name
        lmnr_context.event("topic_alignment", "good")
    
    # to trigger an automatic check for a possible event do:
    lmnr_context.evaluate_event("excessive_wordiness", poem)

    return poem

if __name__ == "__main__":
    print(poem_writer(topic="laminar flow"))
```

This gives an advantage of quick instrumentation, but is somewhat limited in flexibility + doesn't really work as expected with threading.
This is due to the fact that we use `contextvars.ContextVar` for this, and how Python manages them between threads.

If you want to instrument your code manually, follow on to the next section

## Manual instrumentation example

For manual instrumetation you will need to import the following:
- `trace` - this is a function to start a trace. It returns a `TraceContext`
- `TraceContext` - a pointer to the current trace that you can pass around functions as you want.
- `SpanContext` - a pointer to the current span that you can pass around functions as you want

Both `TraceContext` and `SpanContext` expose the following interfaces:
- `span(name: str, **kwargs)` - create a child span within the current context. Returns `SpanContext`
- `update(**kwargs)` - update the current trace or span and return it. Returns `TraceContext` or `SpanContext`. Useful when some metadata becomes known later during the program execution

In addition, `SpanContext` allows you to:
- `event(name: str, value: str | int)` - emit a custom event at any point
- `evaluate_event(name: str, data: str)` - register a possible event for automatic checking by Laminar.
- `end(**kwargs)` – update the current span, and terminate it

Example:

```python
import os
from openai import OpenAI

from lmnr import trace, TraceContext, SpanContext, EvaluateEvent
from lmnr.semantic_conventions.gen_ai_spans import INPUT_TOKEN_COUNT, OUTPUT_TOKEN_COUNT, RESPONSE_MODEL, PROVIDER, STREAM
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def poem_writer(t: TraceContext, topic = "turbulence"):
    span: SpanContext = t.span(name="poem_writer", input=topic)

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
        llm_span.event("topic_alignment", "good")  # send an event with a pre-defined name

    # note that you can register possible events here as well,
    # not only `llm_span.evaluate_event()`
    llm_span.end(
        output=poem,
        evaluate_events=[EvaluateEvent(name="excessive_wordines", data=poem)],
        attributes={
            INPUT_TOKEN_COUNT: response.usage.prompt_tokens,
            OUTPUT_TOKEN_COUNT: response.usage.completion_tokens,
            RESPONSE_MODEL: response.model,
            PROVIDER: 'openai',
            STREAM: False
        }
    )
    span.end(output=poem)
    return poem


t: TraceContext = trace(user_id="user123", session_id="session123", release="release")
main(t, topic="laminar flow")
```

## Manual attributes

You can specify span attributes when creating/updating/ending spans.

If you use [decorator instrumentation](#decorator-instrumentation-example), `wrap_llm_call` handles all of this for you.

Example usage:

```python
from lmnr.semantic_conventions.gen_ai_spans import REQUEST_MODEL

# span_type = LLM is important for correct attribute semantics
llm_span = span.span(name="OpenAI completion", input=messages, span_type="LLM")
llm_span.update(
    attributes={REQUEST_MODEL: "gpt-4o-mini"}
)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello. What is the capital of France?"},
    ],
)
```

Semantics:

Check for available semantic conventions in `lmnr.semantic_conventions.gen_ai_spans`.

You can specify the cost with `COST`. Otherwise, the cost will be calculated 
on the Laminar servers, given the following are specified:

- span_type is `"LLM"`
- Model provider: `PROVIDER`, e.g. 'openai', 'anthropic'
- Output tokens: `OUTPUT_TOKEN_COUNT`
- Input tokens: `INPUT_TOKEN_COUNT`*
- Model. We look at `RESPONSE_MODEL` first, and then, if it is not present, we take the value of `REQUEST_MODEL`

\* Also, for the case when `PROVIDER` is `"openai"`, the `STREAM` is set to `True`, and `INPUT_TOKEN_COUNT` is not set, we will calculate
the number of input tokens, and the cost on the server using [tiktoken](https://github.com/zurawiki/tiktoken-rs) and 
use it in cost calculation.
This is done because OpenAI does not stream the usage back
when streaming is enabled. Output token count is (approximately) equal to the number of streaming
events sent by OpenAI, but there is no way to calculate the input token count, other than re-tokenizing.

## Making Laminar pipeline calls

After you are ready to use your pipeline in your code, deploy it in Laminar by selecting the target version for the pipeline.

Once your pipeline target is set, you can call it from Python in just a few lines.

Example use:

```python
from lmnr import Laminar 

# for decorator instrumentation, do: `from lmnr inport lmnr_context`

l = Laminar('<YOUR_PROJECT_API_KEY>')
result = l.run(  # lmnr_context.run( for decorator instrumentation
    pipeline = 'my_pipeline_name',
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
PipelineRunResponse(
    outputs={'output': {'value': [ChatMessage(role='user', content='hello')]}},
    # useful to locate your trace
    run_id='53b012d5-5759-48a6-a9c5-0011610e3669'
)
```
