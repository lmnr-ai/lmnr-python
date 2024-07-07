import threading
from dataclasses import dataclass
from typing import Union

from lmnr_engine.types import Message


@dataclass
class State:
    status: str  # "Success", "Empty", "Termination"  # TODO: Turn into Enum
    message: Union[Message, None]

    @classmethod
    def new(cls, val: Message) -> "State":
        return cls(
            status="Success",
            message=val,
        )

    @classmethod
    def empty(cls) -> "State":
        return cls(
            status="Empty",
            message=Message.empty(),
        )

    @classmethod
    def termination(cls) -> "State":
        return cls(
            status="Termination",
            message=None,
        )

    def is_success(self) -> bool:
        return self.status == "Success"

    def is_termination(self) -> bool:
        return self.status == "Termination"

    def get_out(self) -> Message:
        if self.message is None:
            raise ValueError("Cannot get message from a termination state")

        return self.message


@dataclass
class ExecState:
    output: State
    semaphore: threading.Semaphore

    @classmethod
    def new(cls) -> "ExecState":
        return cls(
            output=State.empty(),
            semaphore=threading.Semaphore(0),
        )

    # Assume this is called by the caller who doesn't need to acquire semaphore
    def set_state(self, output: State):
        self.output = output

    # Assume the caller is smart to call this after acquiring the semaphore
    def get_state(self) -> State:
        return self.output

    def set_state_and_permits(self, output: State, permits: int):
        self.output = output
        self.semaphore.release(permits)
