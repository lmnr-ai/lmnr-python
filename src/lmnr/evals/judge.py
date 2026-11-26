"""`ctx.judge(...)`: a small LLM-as-judge on LiteLLM (a Harbor dependency)."""

from __future__ import annotations

import json
from typing import Any

from lmnr.evals.errors import EvalConfigError
from lmnr.sdk.laminar import Laminar

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"pass": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["pass", "reason"],
    "additionalProperties": False,
}

PROMPT = """You grade the output of an AI system against criteria.

<output>
{output}
</output>

<criteria>
{criteria}
</criteria>

Decide whether the output meets the criteria. Answer with JSON: "pass" (true or false) \
and a one-sentence "reason"."""


async def judge(
    output: Any, criteria: str, *, model: str, api_key: str | None = None
) -> tuple[bool, str]:
    """`(passed, reason)` for `output` against `criteria`, asked of `model`."""
    try:
        import litellm
    except ImportError as e:
        raise EvalConfigError(
            "ctx.judge() needs litellm. Harbor installs it; elsewhere, `pip install litellm`."
        ) from e

    text = output if isinstance(output, str) else json.dumps(output, indent=2, default=str)
    with Laminar.start_as_current_span(
        "judge", input={"output": output, "criteria": criteria, "model": model}
    ) as span:
        response: Any = await litellm.acompletion(
            model=model,
            api_key=api_key,
            messages=[{"role": "user", "content": PROMPT.format(output=text, criteria=criteria)}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA, "strict": True},
            },
        )
        content = response.choices[0].message.content or ""
        try:
            verdict = json.loads(content)
            passed, reason = verdict["pass"], str(verdict.get("reason", ""))
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"The judge's answer is not a verdict: {content!r}") from e
        if not isinstance(passed, bool):
            raise TypeError(f"The judge's answer is not a verdict: {content!r}")
        span.set_output({"pass": passed, "reason": reason})
    return passed, reason
