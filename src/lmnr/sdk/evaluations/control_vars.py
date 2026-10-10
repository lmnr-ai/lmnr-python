from contextvars import ContextVar

from lmnr.sdk.evaluations.evaluation import Evaluation

PREPARE_ONLY: ContextVar[bool] = ContextVar("__lmnr_prepare_only", default=False)
EVALUATION_INSTANCES: ContextVar[list[Evaluation] | None]= ContextVar("__lmnr_evaluation_instances")
