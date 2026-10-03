from collections.abc import Callable


class Config:
    enrich_token_usage: bool = False
    exception_logger: Callable[..., None] | None = None
    use_legacy_attributes: bool = True
