from typing import Callable, Coroutine


class Config:
    exception_logger = None
    upload_base64_image: (
        Callable[[str, str, str, str], Coroutine[None, None, str]] | None
    ) = None
    convert_image_to_openai_format: bool = True
