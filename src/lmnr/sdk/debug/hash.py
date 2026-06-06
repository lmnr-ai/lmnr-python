"""Canonical input hash for the v2 debugger cache (shared spec §5).

Each replayed LLM call is addressed by a single 32-byte blake3 digest of its
input messages, with the system message excluded:

    input_hash = hex( blake3( canonical_json( messages_without_system ) ) )

This module reproduces app-server's primitives byte-for-byte so the hash the SDK
sends matches the hash app-server computes from the stored span:

- `canonical_json` mirrors `app-server/src/traces/input_dedup.rs::canonical_json`
  (object keys sorted lexicographically and recursively, array order preserved,
  scalars serialized like `serde_json::to_string`).
- system-message exclusion mirrors
  `app-server/src/traces/prompt_hash.rs::extract_system_message`.

Part of the cross-language parity surface — keep line-comparable with the TS
`hash.ts` and pinned by the shared `tests/data/debug/input_hash_cases.json`
vector. No AI-SDK reshape lives here: Python provider messages already match the
stored shape (shared spec §9; the reshape is TS-only).

Number canonicalization is deferred (shared spec §5.1): `1.0` and `1` are NOT
normalized to the same form. Most inputs are strings; revisit only if needed,
defining it in the shared spec and applying it on both SDK and app-server in
lockstep.
"""

import json
from typing import Any

from blake3 import blake3


def _canonical_json(value: Any) -> str:
    """Reproduce app-server's `canonical_json` (input_dedup.rs).

    Objects → keys sorted lexicographically (recursive); arrays → order
    preserved; scalars → `json.dumps` (the `serde_json::to_string` equivalent for
    strings/numbers/bools/null). No whitespace, `","` / `":"` separators.
    """
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: kv[0])
        return (
            "{"
            + ",".join(
                json.dumps(k, ensure_ascii=False) + ":" + _canonical_json(v)
                for k, v in items
            )
            + "}"
        )
    if isinstance(value, list):
        return "[" + ",".join(_canonical_json(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def _extract_system_remaining(messages: Any) -> list[Any] | None:
    """Return the message array without its system message, or None.

    Mirrors `extract_system_message` (prompt_hash.rs): find the first
    `role == "system"` entry, extract its text from the string / content-array /
    parts shapes, and — only when that text is non-empty — return the remaining
    messages. Returns None (caller hashes the whole input unchanged) when the
    input isn't an array, has no system message, or the system text is empty.
    """
    if not isinstance(messages, list):
        return None

    sys_idx = next(
        (
            i
            for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "system"
        ),
        None,
    )
    if sys_idx is None:
        return None

    sys_text = _system_text(messages[sys_idx])
    if not sys_text:
        return None

    return [m for i, m in enumerate(messages) if i != sys_idx]


def _system_text(sys_msg: dict[str, Any]) -> str:
    """Extract the system prompt text (priority order matches prompt_hash.rs)."""
    content = sys_msg.get("content")
    # "content": "plain string" (OpenAI format)
    if isinstance(content, str):
        return content
    # "content": [{"text": "...", "type": "text"}, ...] (Anthropic format)
    if isinstance(content, list):
        joined = " ".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
        if joined:
            return joined
    # "parts" shapes — first part only (Gemini {"text"}, OTel {"content"}).
    parts = sys_msg.get("parts")
    if isinstance(parts, list) and parts:
        first = parts[0]
        if isinstance(first, dict):
            text = first.get("text")
            if not isinstance(text, str):
                text = first.get("content")
            if isinstance(text, str):
                return text
    return ""


def debug_input_hash(messages: Any) -> str:
    """Hex blake3 of the canonical, system-excluded input messages (shared §5)."""
    remaining = _extract_system_remaining(messages)
    target = remaining if remaining is not None else messages
    canonical = _canonical_json(target)
    return blake3(canonical.encode("utf-8")).hexdigest()
