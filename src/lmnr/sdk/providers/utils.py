import logging
import json
import pydantic
import typing

logger = logging.getLogger("lmnr.sdk.tracing.providers.utils")


def parse_or_dump_to_dict(
    obj: typing.Union[pydantic.BaseModel, dict[str, typing.Any], str]
) -> dict[str, typing.Any]:
    if isinstance(obj, pydantic.BaseModel):
        return obj.model_dump()
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                return parsed
            else:
                logging.warning(
                    f"Expected a dict, but got: {type(parsed)}. Returning empty dict."
                )
                return {}
        except Exception as e:
            logging.error(f"Error parsing string: {e}")
            return {}

    if isinstance(obj, dict):
        return obj
    logging.warning(
        f"Expected a dict, BaseModel, or str, but got {type(obj)}. Returning empty dict."
    )
    return {}
