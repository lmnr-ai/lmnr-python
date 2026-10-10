import json
from typing import Any

from google.genai._api_client import BaseApiClient
from google.genai._transformers import t_schema
from google.genai.types import JSONSchemaType
from typing_extensions import override

DUMMY_CLIENT = BaseApiClient(api_key="dummy")


def process_schema(schema: Any) -> dict[str, Any]:
    # The only thing we need from the client is the t_schema function
    try:
        type_schema = t_schema(DUMMY_CLIENT, schema)
        if type_schema is None:
            return {}
        json_schema = type_schema.json_schema.model_dump(
            exclude_unset=True, exclude_none=True
        )
    except Exception:
        json_schema = {}
    return json_schema


class SchemaJSONEncoder(json.JSONEncoder):
    @override
    def default(self, o: Any) -> Any:
        if isinstance(o, JSONSchemaType):
            return o.value
        return super().default(o)
