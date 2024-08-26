import abc
import pydantic
import typing


class Provider(abc.ABC):
    def display_name(self) -> str:
        raise NotImplementedError("display_name not implemented")

    def stream_list_to_dict(self, response: list[typing.Any]) -> dict[str, typing.Any]:
        raise NotImplementedError("stream_list_to_dict not implemented")

    def extract_llm_attributes_from_response(
        self, response: typing.Union[str, dict[str, typing.Any], pydantic.BaseModel]
    ) -> dict[str, typing.Any]:
        raise NotImplementedError(
            "extract_llm_attributes_from_response not implemented"
        )

    def extract_llm_output(
        self, response: typing.Union[str, dict[str, typing.Any], pydantic.BaseModel]
    ) -> typing.Any:
        raise NotImplementedError("extract_llm_output not implemented")

    def extract_llm_attributes_from_args(
        self, func_args: list[typing.Any], func_kwargs: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        raise NotImplementedError("_extract_llm_attributes_from_args not implemented")
