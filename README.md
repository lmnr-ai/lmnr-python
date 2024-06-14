Python SDK for Laminar AI

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