"""Wrappers for opencode-ai ``SessionResource.chat`` and its async variant.

The wrappers append a synthetic text part with Laminar span context metadata to
the ``parts`` kwarg. The Opencode server forwards that metadata so the LLM
call span can be re-parented onto the caller's Laminar trace.

The synthetic part format matches the TS instrumentation and the Opencode
server's expectations:

    {
        "type": "text",
        "text": "",
        "synthetic": True,
        "ignored": True,
        "metadata": {"lmnrSpanContext": "<serialized LaminarSpanContext>"},
    }
"""

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


def _inject_span_context(kwargs: dict) -> dict:
    """Return a copy of ``kwargs`` with a synthetic laminar-span-context part
    appended to ``parts`` when a Laminar span context is active and ``parts``
    is a list.

    Leaves ``kwargs`` unchanged when:
    - Laminar is not initialised or there is no active span context,
    - ``parts`` is missing, not a list, or not provided as a keyword argument.
    """
    # Local import to avoid a circular import at module load time.
    from lmnr.sdk.laminar import Laminar

    try:
        serialized = Laminar.serialize_span_context()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to serialize Laminar span context: %s", exc)
        return kwargs

    if serialized is None:
        return kwargs

    parts = kwargs.get("parts")
    if not isinstance(parts, list):
        return kwargs

    synthetic_part = {
        "type": "text",
        "text": "",
        "synthetic": True,
        "ignored": True,
        "metadata": {"lmnrSpanContext": serialized},
    }
    new_kwargs = dict(kwargs)
    new_kwargs["parts"] = [*parts, synthetic_part]
    return new_kwargs


def wrap_chat(wrapped, instance, args, kwargs):
    new_kwargs = _inject_span_context(kwargs)
    return wrapped(*args, **new_kwargs)


async def wrap_chat_async(wrapped, instance, args, kwargs):
    new_kwargs = _inject_span_context(kwargs)
    return await wrapped(*args, **new_kwargs)
