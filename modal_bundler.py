import os
import json
import tempfile
import zipfile
import subprocess
import uuid
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, Any

from pydantic import BaseModel
import modal
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# Create Modal app for bundling
app = modal.App("laminar-evaluation-bundler")

# Define the Modal image with UV and Python tools
bundler_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "boto3"
    ])
    .run_commands([
        "apt-get update && apt-get install -y curl",  # Install curl first
        "curl -LsSf https://astral.sh/uv/install.sh | sh",  # Install UV
        "ln -s /root/.local/bin/uv /usr/local/bin/uv",  # Create symlink to make UV globally available
        "/usr/local/bin/uv --version",  # Verify UV is working using the symlink
    ])
)

class BundleRequest(BaseModel):
    """Request to create an evaluation bundle"""
    source_zip_data: str  # Base64 encoded zip of source files
    dependencies: list[str] = []  # Optional explicit dependencies
    python_requirement: str = ">=3.10"
    bundle_name: str = None  # Optional custom bundle name

class BundleResponse(BaseModel):
    """Response from bundle creation"""
    success: bool
    bundle_id: str = None
    error: str = None
    s3_key: str = None
    bundle_size: int = None

def upload_to_s3(bundle_zip_path: Path, bundle_id: str) -> str:
    """Upload bundle to S3 and return the S3 key"""
    try:
        s3_client = boto3.client('s3')
        bucket_name = os.environ.get('EVAL_BUNDLES_S3_BUCKET', 'hosted-evals-zip')
        s3_key = f"{bundle_id}.zip"
        
        # Upload the bundle
        s3_client.upload_file(str(bundle_zip_path), bucket_name, s3_key)
        print(f"✅ Uploaded bundle to S3: s3://{bucket_name}/{s3_key}")
        return s3_key
        
    except Exception as e:
        raise Exception(f"Failed to upload to S3: {e}")

def extract_source_files(source_zip_data: str, extract_dir: Path):
    """Extract source files from base64 encoded zip data"""
    import base64
    
    # Decode base64 zip data
    zip_bytes = base64.b64decode(source_zip_data)
    zip_path = extract_dir / "source.zip"
    
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)
    
    # Extract source files
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Remove the zip file
    os.remove(zip_path)
    print(f"📁 Extracted source files to {extract_dir}")

async def create_evaluation_bundle(
    source_dir: Path,
    bundle_dir: Path,
    dependencies: list[str],
    python_requirement: str
) -> bool:
    """Create evaluation bundle using the same logic as local bundler"""
    
    try:
        print(f"🔨 Creating evaluation bundle...")
        
        # Find evaluation file
        eval_files = list(source_dir.glob("*.py"))
        if not eval_files:
            raise ValueError("No Python evaluation files found in source")
        
        eval_file = eval_files[0]  # Take the first Python file
        print(f"📝 Using evaluation file: {eval_file.name}")
        
        # Create bundle directory
        bundle_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Copy evaluation file
        eval_dest = bundle_dir / "evaluation.py"
        eval_dest.write_text(eval_file.read_text())
        print(f"📋 Copied evaluation file")
        
        # 2. Create pyproject.toml
        pyproject_content = f'''[project]
name = "eval-bundle"
version = "0.1.0"
description = "Evaluation bundle for Laminar remote execution"
dependencies = {json.dumps(dependencies, indent=2)}
requires-python = "{python_requirement}"

[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"
'''
        pyproject_path = bundle_dir / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        print(f"📦 Created pyproject.toml with {len(dependencies)} dependencies")
        
        # 3. Create virtual environment
        print("🔨 Creating virtual environment...")
        
        # Debug: Check if UV is available
        try:
            uv_check = await asyncio.create_subprocess_exec(
                "which", "uv",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await uv_check.communicate()
            if uv_check.returncode == 0:
                print(f"✅ UV found at: {stdout.decode().strip()}")
            else:
                print(f"❌ UV not found in PATH")
                
            # Also try UV version
            version_check = await asyncio.create_subprocess_exec(
                "uv", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await version_check.communicate()
            if version_check.returncode == 0:
                print(f"✅ UV version: {stdout.decode().strip()}")
            else:
                print(f"❌ UV version check failed: {stderr.decode()}")
        except Exception as e:
            print(f"❌ UV check failed: {e}")
        
        result = await asyncio.create_subprocess_exec(
            "uv", "venv", ".venv",
            cwd=bundle_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        
        if result.returncode != 0:
            print(f"❌ UV venv stdout: {stdout.decode()}")
            print(f"❌ UV venv stderr: {stderr.decode()}")
            raise Exception(f"UV venv failed: {stderr.decode()}")
        
        print(f"✅ Virtual environment created")
        
        # 4. Install dependencies natively on Linux
        print(f"📥 Installing {len(dependencies)} dependencies natively on Linux...")
        cmd_args = [
            "uv", "pip", "install",
            "--python", ".venv/bin/python",
        ] + dependencies
        
        print(f"🔧 Running command: {' '.join(cmd_args)}")
        
        result = await asyncio.create_subprocess_exec(
            *cmd_args,
            cwd=bundle_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        
        if result.returncode != 0:
            print(f"❌ Dependencies install failed: {stderr.decode()}")
            print(f"❌ Dependencies install stdout: {stdout.decode()}")
            return False
        
        print(f"✅ Dependencies installed successfully")
        print(f"📦 Installation output: {stdout.decode()[:500]}...")  # Show first 500 chars
        
        # 5. Create CLI entry point
        await create_cli_entry_point(bundle_dir)
        
        # 6. Create executable script
        await create_bundle_executable(bundle_dir)
        
        # 7. Create bundle metadata
        await create_bundle_metadata(bundle_dir, eval_file)
        
        print(f"🎉 Bundle created successfully at {bundle_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Bundle creation failed: {e}")
        return False

async def create_cli_entry_point(bundle_dir: Path):
    """Create CLI entry point for subprocess execution"""
    cli_content = '''#!/usr/bin/env python3
"""
CLI entry point for evaluation bundle.
This script can be executed as a subprocess and communicates via stdout/stdin.
"""
import sys
import json
import asyncio
import argparse
from pathlib import Path
import os

# Add dependencies to Python path - check for UV-style .venv first
bundle_dir = Path(__file__).parent

# Check for UV-style .venv structure first (preferred)
uv_site_packages_found = False
venv_lib_dir = bundle_dir / ".venv" / "lib"
if venv_lib_dir.exists():
    for python_dir in venv_lib_dir.glob("python*"):
        uv_site_packages = python_dir / "site-packages"
        if uv_site_packages.exists():
            sys.path.insert(0, str(uv_site_packages))
            uv_site_packages_found = True
            break

# Fallback to direct site-packages directory
if not uv_site_packages_found:
    site_packages = bundle_dir / "site-packages"
    if site_packages.exists():
        sys.path.insert(0, str(site_packages))

# Add bundle directory to path for evaluation imports
sys.path.insert(0, str(bundle_dir))

try:
    from evaluation import *  # Import all evaluation code
    from lmnr.sdk.types import Datapoint, LaminarSpanContext
    from lmnr.sdk.log import get_default_logger
    from lmnr.sdk.utils import from_env
    from lmnr import Laminar
except ImportError as e:
    # Fallback if lmnr is not in site-packages
    print(json.dumps({"error": f"Import error: {e}", "success": False}))
    sys.exit(1)

LOG = get_default_logger(__name__)

async def execute_datapoint(
    datapoint_data: dict,
    executor_func_name: str,
    evaluators_config: dict,
    base_url: str = None,
    span_context: dict | str = None,
    project_api_key: str | None = None
) -> dict:
    """Execute evaluation for a single datapoint with Laminar instrumentation"""
    
    try:
        # Initialize Laminar if API key is available
        if project_api_key:
            Laminar.initialize(
                project_api_key=project_api_key,
                base_url=base_url or "https://api.lmnr.ai"
            )
            LOG.debug("Laminar initialized successfully")
        else:
            LOG.warning("No project API key found in environment. Laminar instrumentation disabled.")
        
        # Parse span context if provided
        parent_span_context = None
        if span_context:
            try:
                parent_span_context = LaminarSpanContext.deserialize(span_context)
                LOG.debug("Span context parsed successfully")
            except Exception as e:
                LOG.warning(f"Failed to parse span context: {e}")
        
        # Convert to Datapoint object
        datapoint = Datapoint.model_validate(datapoint_data)
        
        # Get executor function from globals
        executor_func = globals().get(executor_func_name)
        if not executor_func:
            return {
                "success": False,
                "error": f"Executor function '{executor_func_name}' not found"
            }
                    
        # Execute the function with Laminar tracing
        with Laminar.start_as_current_span(
            name=f"executor.{executor_func_name}",
            input=datapoint.data,
            span_type="EXECUTOR",
            parent_span_context=parent_span_context
        ):
            
            # Execute the function
            if asyncio.iscoroutinefunction(executor_func):
                output = await executor_func(datapoint.data)
            else:
                output = executor_func(datapoint.data)
            
            # Set output using Laminar SDK
            Laminar.set_span_output(output)
        
        # Run evaluators with tracing
        scores = {}
        for evaluator_name, evaluator_func_name in evaluators_config.items():
            evaluator_func = globals().get(evaluator_func_name)
            if not evaluator_func:
                LOG.warning(f"Evaluator function '{evaluator_func_name}' not found")
                continue
            
            with Laminar.start_as_current_span(
                name=f"evaluator.{evaluator_name}",
                input={"output": output, "target": datapoint.target},
                span_type="EVALUATOR",
                parent_span_context=parent_span_context
            ):
                
                try:
                    if asyncio.iscoroutinefunction(evaluator_func):
                        score = await evaluator_func(output, datapoint.target)
                    else:
                        score = evaluator_func(output, datapoint.target)
                    
                    # Set the score as span output
                    Laminar.set_span_output(score)
                    
                    # Handle single number or dict scores
                    if isinstance(score, (int, float, bool)):
                        scores[evaluator_name] = score
                    elif isinstance(score, dict):
                        scores.update(score)
                    else:
                        # Convert other types to string
                        scores[evaluator_name] = str(score)
                    
                except Exception as eval_error:
                    LOG.error(f"Error in evaluator {evaluator_name}: {eval_error}")
                    # Set error as span output and continue with other evaluators
                    Laminar.set_span_output({"error": str(eval_error)})
        
        return {
            "success": True,
            "executor_output": output,
            "scores": scores,
            "error": None
        }
        
    except Exception as e:
        LOG.error(f"Error executing datapoint: {e}")
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
    parser.add_argument("--base-url", default="https://api.lmnr.ai", help="Laminar base URL")
    parser.add_argument("--span-context", help="JSON string of Laminar span context for tracing continuity")
    parser.add_argument("--project-api-key", help="Laminar project API key")

    args = parser.parse_args()
    
    try:
        # Parse inputs
        datapoint_data = json.loads(args.datapoint)
        span_context = args.span_context
        
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
        
        if not args.project_api_key:
            project_api_key = from_env("LMNR_PROJECT_API_KEY")
        else:
            project_api_key = args.project_api_key

        # Execute evaluation
        result = asyncio.run(execute_datapoint(
            datapoint_data,
            executor_name,
            evaluators_config,
            base_url=args.base_url,
            span_context=span_context,
            project_api_key=project_api_key
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
'''
    
    cli_path = bundle_dir / "run_evaluation.py"
    cli_path.write_text(cli_content)
    cli_path.chmod(0o755)
    print("📝 Created CLI entry point")

async def create_bundle_executable(bundle_dir: Path):
    """Create main executable script for the bundle"""
    executable_content = '''#!/bin/bash
# Bundle executable script
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE_DIR"

# Find and set up Python path to include bundled dependencies
export PYTHONPATH="$BUNDLE_DIR"

# Check for UV-style .venv structure first (preferred)
if [ -d "$BUNDLE_DIR/.venv/lib" ]; then
    # Find the correct Python version directory
    for python_dir in "$BUNDLE_DIR/.venv/lib"/python*; do
        if [ -d "$python_dir/site-packages" ]; then
            export PYTHONPATH="$python_dir/site-packages:$PYTHONPATH"
            echo "Using UV site-packages: $python_dir/site-packages"
            break
        fi
    done
fi

echo "Final PYTHONPATH: $PYTHONPATH"

# Execute the CLI with all arguments
exec python3 run_evaluation.py "$@"
'''
    
    executable_path = bundle_dir / "run"
    executable_path.write_text(executable_content)
    executable_path.chmod(0o755)
    print("🔧 Created bundle executable script")

async def create_bundle_metadata(bundle_dir: Path, eval_file: Path):
    """Create metadata file for the bundle"""
    import sys
    
    # Calculate file hash for integrity checking
    file_hash = hashlib.sha256(eval_file.read_bytes()).hexdigest()
    
    metadata = {
        "bundle_version": "1.0",
        "eval_file_name": eval_file.name,
        "eval_file_hash": file_hash,
        "created_at": str(uuid.uuid4()),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "bundled_on": "linux_x86_64",  # Native bundling platform
    }
    
    metadata_path = bundle_dir / "bundle_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print("📋 Created bundle metadata")

def create_zip_from_bundle(bundle_dir: Path, output_zip: Path):
    """Create zip file from bundle directory"""
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in bundle_dir.rglob('*'):
            if file_path.is_file():
                arc_name = file_path.relative_to(bundle_dir)
                
                # Create ZipInfo to set file permissions
                zip_info = zipfile.ZipInfo(str(arc_name))
                
                # Set executable permissions for run scripts
                if arc_name.name in ['run', 'run_evaluation.py']:
                    zip_info.external_attr = 0o755 << 16  # Unix permissions
                else:
                    zip_info.external_attr = 0o644 << 16  # Regular file permissions
                
                # Read file content and add to zip
                with open(file_path, 'rb') as src_file:
                    zipf.writestr(zip_info, src_file.read())
    
    print(f"📦 Created zip bundle: {output_zip}")

@app.function(
    image=bundler_image,
    secrets=[
        modal.Secret.from_name("aws-secret"),  # AWS credentials for S3 upload
    ],
    timeout=1200,  # 20 minutes timeout for bundling
    memory=4096,   # 4GB memory for large bundles
)
@modal.fastapi_endpoint(method="POST")
async def create_remote_bundle(request_data: BundleRequest) -> BundleResponse:
    """
    Laminar's internal evaluation bundling service.
    
    This runs on Modal to create Linux-native evaluation bundles without 
    cross-compilation issues. Called by Laminar API endpoints.
    """
    
    try:
        print(f"🚀 Starting remote bundle creation...")
        
        # Generate bundle ID
        bundle_id = request_data.bundle_name or f"bundle_{uuid.uuid4().hex[:12]}"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            bundle_dir = temp_path / "bundle"
            
            # 1. Extract source files
            print(f"📂 Extracting source files...")
            source_dir.mkdir()
            extract_source_files(request_data.source_zip_data, source_dir)
            
            # 2. Ensure we have dependencies (fallback to basic set)
            dependencies = request_data.dependencies
            if not dependencies:
                dependencies = ["lmnr"]  # Minimum required
                print(f"⚠️  No dependencies specified, using default: {dependencies}")
            
            # 3. Create the bundle
            print(f"🔨 Creating evaluation bundle...")
            success = await create_evaluation_bundle(
                source_dir=source_dir,
                bundle_dir=bundle_dir,
                dependencies=dependencies,
                python_requirement=request_data.python_requirement
            )
            
            if not success:
                return BundleResponse(
                    success=False,
                    error="Bundle creation failed"
                )
            
            # 4. Create zip file
            bundle_zip = temp_path / f"{bundle_id}.zip"
            create_zip_from_bundle(bundle_dir, bundle_zip)
            
            # 5. Upload to S3
            print(f"☁️  Uploading bundle to S3...")
            s3_key = upload_to_s3(bundle_zip, bundle_id)
            bundle_size = bundle_zip.stat().st_size
            
            print(f"✅ Remote bundle creation completed!")
            print(f"   Bundle ID: {bundle_id}")
            print(f"   S3 Key: {s3_key}")
            print(f"   Size: {bundle_size:,} bytes")
            
            return BundleResponse(
                success=True,
                bundle_id=bundle_id,
                s3_key=s3_key,
                bundle_size=bundle_size
            )
            
    except Exception as e:
        print(f"❌ Remote bundle creation failed: {e}")
        return BundleResponse(
            success=False,
            error=str(e)
        )
