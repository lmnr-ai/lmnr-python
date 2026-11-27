from __future__ import annotations

from dataclasses import dataclass

# How much of a failed command's output goes into its error.
OUTPUT_TAIL_CHARS = 2000


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    return_code: int


class EvalConfigError(Exception):
    """An `EvalRunner` (or the `eval.py` that defines it) is set up wrong."""


class CommandError(RuntimeError):
    """A command run with `check=True` exited with a non-zero code."""

    def __init__(self, command: str, result: ExecResult, what: str = "Command"):
        self.command: str = command
        self.result: ExecResult = result
        message = f"{what} failed with exit code {result.return_code}: {command}"
        super().__init__(message + output_tails(result))


class CommandTimeout(TimeoutError):
    """A command ran longer than its `timeout` and was killed, with everything
    it started (`check` doesn't matter). `result` has the output it printed
    until then, and exit code 124."""

    def __init__(self, command: str, result: ExecResult, timeout: float, what: str = "Command"):
        self.command: str = command
        self.result: ExecResult = result
        self.timeout: float = timeout
        message = f"{what} timed out after {timeout:g}s: {command}"
        super().__init__(message + output_tails(result))


def output_tails(result: ExecResult) -> str:
    """The end of a command's stdout and stderr, for an error message."""
    tails = ""
    for name, output in (("stdout", result.stdout), ("stderr", result.stderr)):
        if output.strip():
            tails += f"\n--- {name} (tail) ---\n{output[-OUTPUT_TAIL_CHARS:].rstrip()}"
    return tails
