from contextvars import ContextVar

PREPARE_ONLY = ContextVar("__lmnr_prepare_only", default=False)
EVALUATION_INSTANCE = ContextVar("__lmnr_evaluation_instance", default=None)
