from typing import Callable


class Config:
    exception_logger: Callable[[Exception], None] | None = None

