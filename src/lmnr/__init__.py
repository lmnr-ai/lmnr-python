from .sdk.client.synchronous.sync_client import LaminarClient
from .sdk.client.asynchronous.async_client import AsyncLaminarClient
from .sdk.datasets import EvaluationDataset, LaminarDataset
from .sdk.evaluations import evaluate
from .sdk.laminar import Laminar
from .sdk.types import (
    AgentOutput,
    FinalOutputChunkContent,
    HumanEvaluator,
    RunAgentResponseChunk,
    StepChunkContent,
    TracingLevel,
)
from .sdk.decorators import observe
from .sdk.types import LaminarSpanContext
from .opentelemetry_lib.tracing.attributes import Attributes
from .opentelemetry_lib.tracing.instruments import Instruments
from .opentelemetry_lib.tracing.processor import LaminarSpanProcessor
from .opentelemetry_lib.tracing.tracer import get_laminar_tracer_provider, get_tracer
from opentelemetry.trace import use_span

# Remote evaluation decorators
def executor(func=None):
    """
    Decorator to mark a function as an executor for remote evaluation.
    
    Usage:
        @executor()
        async def my_executor(data):
            return {"result": "..."}
    """
    def decorator(f):
        f._lmnr_executor = True
        return f
    
    if func is None:
        # Called with parentheses: @executor()
        return decorator
    else:
        # Called without parentheses: @executor (backward compatibility)
        return decorator(func)

def evaluator(name=None):
    """
    Decorator to mark a function as an evaluator for remote evaluation.
    
    Args:
        name: Optional custom name for the evaluator. If not provided, uses function name.
    
    Usage:
        @evaluator()
        def accuracy_check(output, target):
            return 1.0 if output == target else 0.0
            
        @evaluator("custom_name")  
        def my_evaluator(output, target):
            return score
    """
    def decorator(func):
        func._lmnr_evaluator = True
        func._lmnr_evaluator_name = name if name is not None else func.__name__
        return func
    
    if callable(name):
        # Called without parentheses: @evaluator (backward compatibility)
        func = name
        func._lmnr_evaluator = True
        func._lmnr_evaluator_name = func.__name__
        return func
    else:
        # Called with parentheses: @evaluator() or @evaluator("name")
        return decorator

__all__ = [
    "AgentOutput",
    "AsyncLaminarClient",
    "Attributes",
    "EvaluationDataset",
    "FinalOutputChunkContent",
    "HumanEvaluator",
    "Instruments",
    "Laminar",
    "LaminarClient",
    "LaminarDataset",
    "LaminarSpanContext",
    "LaminarSpanProcessor",
    "RunAgentResponseChunk",
    "StepChunkContent",
    "TracingLevel",
    "get_laminar_tracer_provider",
    "get_tracer",
    "evaluate",
    "observe",
    "use_span",
    "executor",
    "evaluator",
]
