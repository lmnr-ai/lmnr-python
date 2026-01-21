import re


def infer_provider(model: str | None) -> str:
    if model is None:
        return "litellm"

    if "/" in model:
        return model.split("/")[0]
    if "gemini" in model:
        return "gemini"
    if "claude" in model:
        return "anthropic"
    if "gpt" in model or re.match(r"^o[0-9]*$", model):
        return "openai"
    return "litellm"
