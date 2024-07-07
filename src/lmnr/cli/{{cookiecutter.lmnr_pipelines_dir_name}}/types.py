from dataclasses import dataclass
from typing import Union
import uuid
import datetime
from lmnr.types import NodeInput, ChatMessage


@dataclass
class Message:
    id: uuid.UUID
    # output value of producing node in form of NodeInput
    # for the following consumer
    value: NodeInput
    # all input messages to this node; accumulates previous messages too
    # input_messages: list["Message"]
    start_time: datetime.datetime
    end_time: datetime.datetime
    # node_id: uuid.UUID
    # node_name: str
    # node_type: str
    # all node per-run metadata that needs to be logged at the end of execution
    # meta_log: MetaLog | None

    @classmethod
    def empty(cls) -> "Message":
        return cls(
            id=uuid.uuid4(),
            value="",
            # input_messages=[],
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now(),
            # node_id=uuid.uuid4(),
            # node_name="",
            # node_type="",
        )
