"""
Test evaluation for Laminar instrumentation
"""
from lmnr import executor, evaluator

@executor()
async def my_instrumented_executor(data):
    """Test executor with instrumentation"""
    input_text = data.get("input", "")
    return {
        "result": input_text.upper(),
        "char_count": len(input_text),
        "word_count": len(input_text.split())
    }

@evaluator("accuracy")
async def accuracy_check(output, target):
    """Check if the result matches expected"""
    return output.get("result") == target.get("expected")

@evaluator("length_validation")
async def length_validation(output, target):
    """Validate the character count is correct"""
    result = output.get("result", "")
    expected_length = target.get("expected_length", 0)
    actual_length = output.get("char_count", 0)
    return actual_length == expected_length and len(result) == expected_length 