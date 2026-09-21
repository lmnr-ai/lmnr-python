import logging
import os

import dotenv
from typing_extensions import override

GREY = "\x1b[38;20m"
GREEN = "\x1b[32;20m"
YELLOW = "\x1b[33;20m"
RED = "\x1b[31;20m"
BOLD_RED = "\x1b[31;1m"
RESET = "\x1b[0m"


_nameToLevel = {
    'CRITICAL': logging.CRITICAL,
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
}


class CustomFormatter(logging.Formatter):
    fmt: str = "%(asctime)s::%(name)s::%(levelname)s: %(message)s (%(filename)s:%(lineno)d)"

    def _format_definition(self, level: int) -> str | None:
        if level == logging.DEBUG:
            return GREY + self.fmt + RESET
        if level == logging.INFO:
            return GREEN + self.fmt + RESET
        if level == logging.WARNING:
            return YELLOW + self.fmt + RESET
        if level == logging.ERROR:
            return RED + self.fmt + RESET
        if level == logging.CRITICAL:
            return BOLD_RED + self.fmt + RESET
        return None

    @override
    def format(self, record: logging.LogRecord):
        log_fmt = self._format_definition(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ColorfulFormatter(logging.Formatter):
    fmt: str = "Laminar %(levelname)s: %(message)s"

    def _format_definition(self, level: int) -> str | None:
        if level == logging.DEBUG:
            return GREY + self.fmt + RESET
        if level == logging.INFO:
            return GREEN + self.fmt + RESET
        if level == logging.WARNING:
            return YELLOW + self.fmt + RESET
        if level == logging.ERROR:
            return RED + self.fmt + RESET
        if level == logging.CRITICAL:
            return BOLD_RED + self.fmt + RESET
        return None

    @override
    def format(self, record: logging.LogRecord):
        log_fmt = self._format_definition(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# For StreamHandlers / console
class VerboseColorfulFormatter(CustomFormatter):
    @override
    def format(self, record: logging.LogRecord):
        return super().format(record)


# For Verbose FileHandlers / files
class VerboseFormatter(CustomFormatter):
    fmt: str = "%(asctime)s::%(name)s::%(levelname)s: %(message)s (%(filename)s:%(lineno)d)"

    @override
    def format(self, record: logging.LogRecord):
        formatter = logging.Formatter(self.fmt)
        return formatter.format(record)


def get_level_from_env() -> int:
    env_level = None
    if val := os.getenv("LMNR_LOG_LEVEL"):
        env_level = val.upper().strip()
    else:
        dotenv_path = dotenv.find_dotenv(usecwd=True)
        # use DotEnv directly so we can set verbose to False
        env_level = (
            (
                dotenv.main.DotEnv(dotenv_path, verbose=False, encoding="utf-8").get(
                    "LMNR_LOG_LEVEL"
                )
                or "INFO"
            )
            .upper()
            .strip()
        )
    if env_level:
        return _nameToLevel.get(env_level, logging.INFO)  # pyright
    return logging.INFO


_LMNR_HANDLER_FLAG = "_lmnr_default_handler"


def get_default_logger(
    name: str, level: int | None = None, propagate: bool = False, verbose: bool = True
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level or get_level_from_env())
    # `logging.getLogger(name)` returns the SAME logger every call, so adding a handler
    # unconditionally stacks one per call and emits the message once per handler. Callers
    # legitimately call this repeatedly (module import, per-span in LaminarSpan.__init__,
    # inside helpers), so attach at most one handler of our own per logger.
    if not any(getattr(h, _LMNR_HANDLER_FLAG, False) for h in logger.handlers):
        console_log_handler = logging.StreamHandler()
        if verbose:
            console_log_handler.setFormatter(VerboseColorfulFormatter())
        else:
            console_log_handler.setFormatter(ColorfulFormatter())
        setattr(console_log_handler, _LMNR_HANDLER_FLAG, True)
        logger.addHandler(console_log_handler)
    logger.propagate = propagate
    return logger
