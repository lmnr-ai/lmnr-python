"""
Test evaluation for improved workflow (directory-based bundles)
"""
from lmnr import executor, evaluator


@executor()
async def simple_executor(data):
    """Simple executor that processes input"""
    return {
        "result": data.get("input", "").upper(),
        "length": len(data.get("input", ""))
    }


@evaluator("accuracy")
async def accuracy_check(output, target):
    """Check if result matches expected"""
    return output.get("result") == target.get("expected")


@evaluator()  # Will use function name
async def length_check(output, target):
    """Check if length is reasonable"""
    return output.get("length", 0) > 0 