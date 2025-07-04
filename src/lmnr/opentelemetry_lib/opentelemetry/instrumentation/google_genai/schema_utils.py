from typing import Any
from google.genai._api_client import BaseApiClient
from google.genai._transformers import t_schema
from google.genai.types import JSONSchemaType

import json


def process_schema(schema: Any) -> dict[str, Any]:
    # The only thing we need from the client is the t_schema function
    client = BaseApiClient(api_key="dummy")
    json_schema = t_schema(client, schema).json_schema.model_dump(
        exclude_unset=True, exclude_none=True
    )
    return json_schema


class SchemaJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, JSONSchemaType):
            return o.value
        return super().default(o)
