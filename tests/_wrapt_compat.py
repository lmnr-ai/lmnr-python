"""wrapt 1.x / 2.x compatibility helpers for tests.

In wrapt 2.x `FunctionWrapper` derives from `BaseObjectProxy`, not
`ObjectProxy` (which became a subclass adding `__iter__`), so
`isinstance(fn, wrapt.ObjectProxy)` is False for wrapped functions. On 1.x
`ObjectProxy` is the base of `FunctionWrapper`.
"""

import wrapt

WRAPT_PROXY: type = getattr(wrapt, "BaseObjectProxy", wrapt.ObjectProxy)


def is_wrapt_proxy(value: object) -> bool:
    """True when `value` is a wrapt proxy (e.g. a function wrapped by us)."""
    return isinstance(value, WRAPT_PROXY)
