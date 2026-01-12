import logging
import os

import dotenv


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s::%(name)s::%(levelname)s: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: green + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset,
    }

    def format(self, record: logging.LogRecord):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ColorfulFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "Laminar %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: green + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset,
    }

    def format(self, record: logging.LogRecord):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# For StreamHandlers / console
class VerboseColorfulFormatter(CustomFormatter):
    def format(self, record):
        return super().format(record)


# For Verbose FileHandlers / files
class VerboseFormatter(CustomFormatter):
    fmt = "%(asctime)s::%(name)s::%(levelname)s: %(message)s (%(filename)s:%(lineno)d)"

    def format(self, record):
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
        return logging._nameToLevel.get(env_level, logging.INFO)
    return logging.INFO


def get_default_logger(
    name: str, level: int | None = None, propagate: bool = False, verbose: bool = True
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level or get_level_from_env())
    console_log_handler = logging.StreamHandler()
    if verbose:
        console_log_handler.setFormatter(VerboseColorfulFormatter())
    else:
        console_log_handler.setFormatter(ColorfulFormatter())
    logger.addHandler(console_log_handler)
    logger.propagate = propagate
    return logger
