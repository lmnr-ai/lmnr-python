"""get_default_logger must attach at most one handler per logger.

`logging.getLogger(name)` returns the same logger object every call, so an unguarded
`addHandler` stacks a handler per call and emits every message once per handler. Callers
legitimately call it repeatedly — notably `LaminarSpan.__init__`, so the duplicate count grew
with the number of spans a run created.
"""

import logging

from lmnr.sdk.log import get_default_logger


def _handler_count(name: str) -> int:
    return len(logging.getLogger(name).handlers)


def test_repeated_calls_attach_only_one_handler():
    name = "lmnr.test.repeated"
    logging.getLogger(name).handlers.clear()

    for _ in range(25):
        get_default_logger(name)

    assert _handler_count(name) == 1


def test_a_message_is_emitted_once(caplog):
    name = "lmnr.test.emit_once"
    logging.getLogger(name).handlers.clear()

    logger = get_default_logger(name)
    for _ in range(5):
        get_default_logger(name)

    # propagate=False by default, so caplog needs the records routed to it.
    with caplog.at_level(logging.WARNING, logger=name):
        logging.getLogger(name).propagate = True
        logger.warning("only once")

    assert [r.message for r in caplog.records].count("only once") == 1


def test_a_foreign_handler_is_left_alone():
    """We only dedupe OUR handler, so a user's own handler must survive."""
    name = "lmnr.test.foreign"
    logger = logging.getLogger(name)
    logger.handlers.clear()
    foreign = logging.StreamHandler()
    logger.addHandler(foreign)

    get_default_logger(name)
    get_default_logger(name)

    assert foreign in logger.handlers
    # The user's handler plus exactly one of ours.
    assert _handler_count(name) == 2
