#!/usr/bin/env python3
"""
CLI entry point for evaluation bundle.
This script can be executed as a subprocess and communicates via stdout/stdin.
"""
import sys
import json
import asyncio
import argparse
from pathlib import Path

# Add site-packages to Python path for dependencies
bundle_dir = Path(__file__).parent
site_packages = bundle_dir / "site-packages"
if site_packages.exists():
    sys.path.insert(0, str(site_packages))

# Add bundle directory to path for evaluation imports
sys.path.insert(0, str(bundle_dir))

try:
    from evaluation import *  # Import all evaluation code
    from lmnr.sdk.types import Datapoint
    from lmnr.sdk.log import get_default_logger
except ImportError as e:
    # Fallback if lmnr is not in site-packages
    print(json.dumps({"error": f"Import error: {e}", "success": False}))
    sys.exit(1)

LOG = get_default_logger(__name__)


async def execute_datapoint(
    datapoint_data: dict,
    executor_func_name: str,
    evaluators_config: dict
) -> dict:
    """Execute evaluation for a single datapoint"""
    try:
        # Convert to Datapoint object
        datapoint = Datapoint.model_validate(datapoint_data)
        
        # Get executor function from globals
        executor_func = globals().get(executor_func_name)
        if not executor_func:
            return {
                "success": False,
                "error": f"Executor function '{executor_func_name}' not found"
            }
        
        # Execute the function
        if asyncio.iscoroutinefunction(executor_func):
            output = await executor_func(datapoint.data)
        else:
            output = executor_func(datapoint.data)
        
        # Run evaluators
        scores = {}
        for evaluator_name, evaluator_func_name in evaluators_config.items():
            evaluator_func = globals().get(evaluator_func_name)
            if not evaluator_func:
                continue
            
            if asyncio.iscoroutinefunction(evaluator_func):
                score = await evaluator_func(output, datapoint.target)
            else:
                score = evaluator_func(output, datapoint.target)
            
            # Handle single number or dict scores
            if isinstance(score, (int, float)):
                scores[evaluator_name] = score
            elif isinstance(score, dict):
                scores.update(score)
        
        return {
            "success": True,
            "executor_output": output,
            "scores": scores,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "executor_output": None,
            "scores": {}
        }


def auto_detect_functions():
    """Auto-detect executor and evaluator functions from decorators"""
    executor_funcs = []
    evaluator_funcs = []
    evaluator_names = {}  # Map function name to custom evaluator name
    
    # Scan all globals for decorated functions
    for name, obj in globals().items():
        if callable(obj):
            # Check for executor decorator
            if hasattr(obj, '_lmnr_executor') and obj._lmnr_executor:
                executor_funcs.append(name)
            # Check for evaluator decorator  
            if hasattr(obj, '_lmnr_evaluator') and obj._lmnr_evaluator:
                evaluator_funcs.append(name)
                # Get custom evaluator name if provided
                custom_name = getattr(obj, '_lmnr_evaluator_name', name)
                evaluator_names[name] = custom_name
    
    return executor_funcs, evaluator_funcs, evaluator_names


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Evaluation bundle CLI")
    parser.add_argument("--datapoint", required=True, help="JSON string of datapoint data")
    parser.add_argument("--executor", help="Name of executor function (auto-detected if not provided)")
    parser.add_argument("--evaluators", help="JSON string of evaluators config (auto-detected if not provided)")
    
    args = parser.parse_args()
    
    try:
        # Parse inputs
        datapoint_data = json.loads(args.datapoint)
        
        # Auto-detect functions if not provided
        executor_funcs, evaluator_funcs, evaluator_names = auto_detect_functions()
        
        # Determine executor
        if args.executor:
            executor_name = args.executor
        elif executor_funcs:
            if len(executor_funcs) > 1:
                raise ValueError(f"Multiple executors found: {executor_funcs}. Please specify --executor.")
            executor_name = executor_funcs[0]  # Use the single executor found
        else:
            raise ValueError("No executor function found. Use --executor or add @executor() decorator.")
        
        # Determine evaluators
        if args.evaluators:
            evaluators_config = json.loads(args.evaluators)
        else:
            # Auto-create evaluators config using custom names if provided
            evaluators_config = {}
            for func_name in evaluator_funcs:
                evaluator_name = evaluator_names.get(func_name, func_name)
                evaluators_config[evaluator_name] = func_name
            
            if not evaluators_config:
                raise ValueError("No evaluator functions found. Add @evaluator() decorators or use --evaluators.")
        
        # Execute evaluation
        result = asyncio.run(execute_datapoint(
            datapoint_data,
            executor_name,
            evaluators_config
        ))
        
        # Output result as JSON to stdout
        print(json.dumps(result))
        
    except Exception as e:
        # Output error as JSON to stdout
        error_result = {
            "success": False,
            "error": str(e),
            "executor_output": None,
            "scores": {}
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()
