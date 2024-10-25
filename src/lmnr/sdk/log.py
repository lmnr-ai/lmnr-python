import logging


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


# For StreamHandlers / console
class VerboseColorfulFormatter(CustomFormatter):
    def format(self, record):
        return super().format(record)


# For Verbose FileHandlers / files
class VerboseFormatter(CustomFormatter):
    fmt = "%(asctime)s::%(name)s::%(levelname)s| %(message)s (%(filename)s:%(lineno)d)"

    def format(self, record):
        formatter = logging.Formatter(self.fmt)
        return formatter.format(record)


def get_default_logger(name: str, level: int = logging.INFO, propagate: bool = False):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_log_handler = logging.StreamHandler()
    console_log_handler.setFormatter(VerboseColorfulFormatter())
    logger.addHandler(console_log_handler)
    logger.propagate = propagate
    return logger
