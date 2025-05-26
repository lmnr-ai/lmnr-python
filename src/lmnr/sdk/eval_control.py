from contextvars import ContextVar


PREPARE_ONLY: ContextVar[bool] = ContextVar("__lmnr_prepare_only", default=False)
EVALUATION_INSTANCES = ContextVar("__lmnr_evaluation_instances")
