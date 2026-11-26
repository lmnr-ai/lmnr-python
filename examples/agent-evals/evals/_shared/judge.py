"""An LLM judge shared by the datasets that grade replies. It runs on the host,
in verify(), so it uses the Anthropic SDK from evals/pyproject.toml and a key
the container never sees."""

from anthropic import AsyncAnthropic
from lmnr import observe
from pydantic import BaseModel


class Verdict(BaseModel):
    passed: bool
    reason: str


# Traced under the trial's verifier span; the key is left out of the span's input.
@observe(name="judge", ignore_inputs=["api_key"])
async def judge(ticket: str, reply: str, criteria: str, api_key: str) -> Verdict:
    client = AsyncAnthropic(api_key=api_key)
    response = await client.messages.parse(
        model="claude-haiku-5-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "You grade a support agent's reply to a customer.\n\n"
                    f"Ticket:\n{ticket}\n\nCriteria: {criteria}\n\nReply:\n{reply}\n\n"
                    "Does the reply meet the criteria?"
                ),
            }
        ],
        output_format=Verdict,
    )
    if response.parsed_output is None:
        raise ValueError(f"The judge gave no verdict (stop reason {response.stop_reason})")
    return response.parsed_output
