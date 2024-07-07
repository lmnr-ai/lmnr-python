from dataclasses import dataclass
from typing import Union

from lmnr_engine.types import NodeInput


@dataclass
class RunOutput:
    status: str  # "Success" | "Termination"  TODO: Turn into Enum
    output: Union[NodeInput, None]


class NodeRunError(Exception):
    pass
