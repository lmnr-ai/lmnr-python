"""Deterministic seeded permutation — a cross-language parity surface.

This mirrors, byte-for-byte, the TypeScript SDK's ``seededPerm``. No off-the-shelf
RNG is identical across JS and Python, so both SDKs own this tiny generator
(mulberry32) plus a downward Fisher-Yates shuffle. The shared vector fixture
``tests/data/dataset/seeded_perm_cases.json`` guards against drift.

The exact algorithm is pinned in ``.agent-team/spec.md``. Do NOT "improve" it —
any change that alters the output breaks reproducibility across languages and
across SDK versions.
"""

_UINT32_MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """32-bit integer multiply, matching JS ``Math.imul`` (low 32 bits)."""
    return ((a & _UINT32_MASK) * (b & _UINT32_MASK)) & _UINT32_MASK


def seeded_perm(n: int, seed: int) -> list[int]:
    """Return a deterministic permutation of ``range(n)`` for the given seed.

    Pure function of ``(n, seed)``: the same arguments always yield the same
    order. ``t`` is kept masked to a non-negative uint32 throughout so Python's
    ``>>`` behaves as JS's logical ``>>>``.
    """
    if n <= 0:
        return []

    state = seed & _UINT32_MASK

    def next_float() -> float:
        nonlocal state
        # mulberry32 -> float in [0, 1)
        state = (state + 0x6D2B79F5) & _UINT32_MASK
        t = state
        t = _imul(t ^ (t >> 15), t | 1)
        t = (t ^ (t + _imul(t ^ (t >> 7), t | 61))) & _UINT32_MASK
        return ((t ^ (t >> 14)) & _UINT32_MASK) / 0x100000000

    ix = list(range(n))
    # Fisher-Yates, downward.
    for i in range(n - 1, 0, -1):
        j = int(next_float() * (i + 1))
        ix[i], ix[j] = ix[j], ix[i]
    return ix
