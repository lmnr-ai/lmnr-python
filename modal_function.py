import json
import tempfile
import subprocess
import zipfile
import os
from pathlib import Path
from typing import Dict, Any

from pydantic import BaseModel

import modal
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# Create Modal app
app = modal.App("evaluation-bundle-runner")

# Define the Modal image with required dependencies
# Use Python 3.13 to match the bundle dependencies
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install([
        "boto3",
        "requests",
        "lmnr",  # Add lmnr as backup
    ])
)

def extract_bundle(zip_data: bytes, extract_to: str):
    """Extract zip bundle to specified directory"""
    zip_path = Path(extract_to) / "bundle.zip"
    
    # Write zip data to file
    with open(zip_path, "wb") as f:
        f.write(zip_data)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Remove the zip file
    os.remove(zip_path)

def download_bundle_from_s3(bundle_id: str, bucket_name: str = None) -> bytes:
    """Download evaluation bundle from S3 based on bundle_id"""
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Default bucket name if not provided
        if bucket_name is None:
            bucket_name = os.environ.get('EVAL_BUNDLES_S3_BUCKET', 'eval-bundles')
        
        # Construct S3 key - assuming bundles are stored as {bundle_id}.zip
        s3_key = f"{bundle_id}.zip"
        
        # Download the file
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        return response['Body'].read()
        
    except NoCredentialsError:
        raise Exception("AWS credentials not found. Please configure AWS credentials.")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            raise Exception(f"Bundle {bundle_id} not found in S3 bucket {bucket_name}")
        elif error_code == 'NoSuchBucket':
            raise Exception(f"S3 bucket {bucket_name} not found")
        else:
            raise Exception(f"Error downloading bundle from S3: {e}")

class RequestData(BaseModel):
    bundle_id: str
    datapoint: dict
    parent_span_context: str = None
    bucket_name: str = None
    project_api_key: str = None

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("aws-secret"),  # AWS credentials for S3 access
    ],
    timeout=600,  # 10 minutes timeout
)
@modal.fastapi_endpoint(method="POST")
def run_evaluation_bundle(request_data: RequestData) -> Dict[str, Any]:
    """
    Modal web endpoint function that downloads and runs evaluation bundles.
    
    Expected request body:
    {
        "bundle_id": "string",
        "datapoint": {...},
        "parent_span_context": "string" (optional),
        "bucket_name": "string" (optional)
    }
    """
    try:
        # Extract parameters from request
        bundle_id = request_data.bundle_id
        datapoint = request_data.datapoint
        parent_span_context = request_data.parent_span_context
        bucket_name = request_data.bucket_name
        project_api_key = request_data.project_api_key
        
        # Validate required parameters
        if not bundle_id:
            return {
                "success": False,
                "error": "bundle_id is required",
                "executor_output": None,
                "scores": {}
            }
        
        if not datapoint:
            return {
                "success": False,
                "error": "datapoint is required",
                "executor_output": None,
                "scores": {}
            }
        
        # Download bundle from S3
        try:
            print(f"Downloading bundle from S3: {bundle_id} from bucket: {bucket_name}")
            bundle_zip_data = download_bundle_from_s3(bundle_id, bucket_name)
            print(f"Downloaded bundle from S3: {bundle_id} from bucket: {bucket_name}")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to download bundle: {str(e)}",
                "executor_output": None,
                "scores": {}
            }
        
        # Extract bundle to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_bundle(bundle_zip_data, temp_dir)
            
            # Look for the bundle executable script first (preferred method)
            run_script = Path(temp_dir) / "run"
            use_executable = False
            
            if run_script.exists():
                # Make sure it's executable
                run_script.chmod(0o755)
                use_executable = True
                print("Using bundle executable script")
            else:
                # Fallback to run_evaluation.py
                run_script = Path(temp_dir) / "run_evaluation.py"
                if not run_script.exists():
                    return {
                        "success": False,
                        "error": "Neither 'run' script nor 'run_evaluation.py' found in bundle",
                        "executor_output": None,
                        "scores": {}
                    }
                print("Using direct run_evaluation.py")
            
            # Set up environment for bundled dependencies
            env = os.environ.copy()
            
            # Debug: Check Python version
            python_version_result = subprocess.run(
                ["python3", "--version"], 
                capture_output=True, 
                text=True
            )
            print(f"Modal Python version: {python_version_result.stdout.strip()}")
            
            # Add bundled site-packages to Python path
            site_packages_paths = []
            
            # Check for UV-style .venv structure
            venv_lib_dir = Path(temp_dir) / ".venv" / "lib"
            if venv_lib_dir.exists():
                print(f"Found .venv/lib directory: {venv_lib_dir}")
                python_dirs = list(venv_lib_dir.glob("python*"))
                print(f"Available Python directories: {[d.name for d in python_dirs]}")
                
                for python_dir in python_dirs:
                    site_packages = python_dir / "site-packages"
                    if site_packages.exists():
                        site_packages_paths.append(str(site_packages))
                        print(f"Found UV site-packages: {site_packages}")
                        
                        # Debug: Check if pydantic is installed in this site-packages
                        pydantic_path = site_packages / "pydantic"
                        pydantic_core_path = site_packages / "pydantic_core"
                        print(f"Pydantic found: {pydantic_path.exists()}")
                        print(f"Pydantic-core found: {pydantic_core_path.exists()}")
                        
                        if pydantic_core_path.exists():
                            # List contents of pydantic_core
                            pydantic_core_files = list(pydantic_core_path.iterdir())
                            print(f"Pydantic-core files: {[f.name for f in pydantic_core_files[:10]]}")  # Limit output
            
            # Check for direct site-packages directory
            direct_site_packages = Path(temp_dir) / "site-packages"
            if direct_site_packages.exists():
                site_packages_paths.append(str(direct_site_packages))
                print(f"Found direct site-packages: {direct_site_packages}")
            
            # Update PYTHONPATH to include bundled dependencies
            if site_packages_paths:
                current_pythonpath = env.get("PYTHONPATH", "")
                all_paths = site_packages_paths + [temp_dir]
                if current_pythonpath:
                    all_paths.append(current_pythonpath)
                env["PYTHONPATH"] = ":".join(all_paths)
                print(f"Set PYTHONPATH: {env['PYTHONPATH']}")
                
                # Test if critical imports work with bundled dependencies
                try:
                    test_result = subprocess.run(
                        ["python3", "-c", "import lmnr; import pydantic; print('Dependencies OK')"],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=10
                    )
                    if test_result.returncode == 0:
                        print("✅ Bundled dependencies test passed")
                    else:
                        print(f"⚠️ Bundled dependencies test failed: {test_result.stderr}")
                        print("Will try without bundled site-packages...")
                        # Fallback: remove bundled site-packages from PYTHONPATH
                        env["PYTHONPATH"] = f"{temp_dir}:{current_pythonpath}"
                except subprocess.TimeoutExpired:
                    print("⚠️ Dependency test timed out, proceeding anyway...")
                except Exception as e:
                    print(f"⚠️ Could not test dependencies: {e}")
            else:
                # Fallback: just add the bundle directory
                env["PYTHONPATH"] = f"{temp_dir}:{env.get('PYTHONPATH', '')}"
                print("No site-packages found, using bundle directory only")
            
            # Prepare command arguments
            if use_executable:
                # Use the bundle executable which handles the environment setup
                cmd_args = [
                    str(run_script),
                    "--datapoint", json.dumps(datapoint),
                ]
            else:
                # Direct python execution
                cmd_args = [
                    "python3", str(run_script),
                    "--datapoint", json.dumps(datapoint),
                ]
            
            # Add project API key if provided
            if project_api_key:
                cmd_args.extend(["--project-api-key", project_api_key])
            
            # Add span context if provided
            if parent_span_context:
                cmd_args.extend(["--span-context", parent_span_context])
            
            print(f"Executing command: {' '.join(cmd_args)}")
            print(f"Working directory: {temp_dir}")
            
            # Run the evaluation as subprocess
            try:
                result = subprocess.run(
                    cmd_args,
                    capture_output=True,
                    text=True,
                    cwd=temp_dir,  # Set working directory to bundle directory
                    env=env,  # Use updated environment with PYTHONPATH
                    timeout=60 * 60,   # 1 hour timeout for individual evaluation
                )
                
                print(f"Evaluation result: return code {result.returncode}")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Evaluation failed with return code {result.returncode}. stderr: {result.stderr}. stdout: {result.stdout}",
                        "executor_output": None,
                        "scores": {}
                    }
                
                # Parse the output
                try:
                    evaluation_result = json.loads(result.stdout)
                    return evaluation_result
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"Failed to parse evaluation output as JSON: {str(e)}. Output: {result.stdout[:500]}",
                        "executor_output": None,
                        "scores": {}
                    }
                
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": "Evaluation timed out after 1 hour",
                    "executor_output": None,
                    "scores": {}
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error running evaluation subprocess: {str(e)}",
                    "executor_output": None,
                    "scores": {}
                }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "executor_output": None,
            "scores": {}
        } 