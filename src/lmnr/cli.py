from argparse import ArgumentParser
import asyncio
import importlib.util
import os
import re
import sys
import shutil
import subprocess
import json
from pathlib import Path
import uuid

from lmnr.sdk.evaluations import Evaluation
from lmnr.sdk.remote_eval import RemoteEvaluationBundler, RemoteEvaluationRunner

from .sdk.eval_control import PREPARE_ONLY, EVALUATION_INSTANCES
from .sdk.log import get_default_logger

LOG = get_default_logger(__name__)


EVAL_DIR = "evals"


async def run_evaluation(args):
    sys.path.append(os.getcwd())

    if args.file is None:
        files = [
            os.path.join(EVAL_DIR, f)
            for f in os.listdir(EVAL_DIR)
            if re.match(r".*_eval\.py$", f) or re.match(r"eval_.*\.py$", f)
        ]
        if len(files) == 0:
            LOG.error("No evaluation files found in `evals` directory")
            LOG.info(
                "Eval files must be located in the `evals` directory and must be named *_eval.py or eval_*.py"
            )
            return
        files.sort()
        LOG.info(f"Located {len(files)} evaluation files in {EVAL_DIR}")

    else:
        files = [args.file]

    prep_token = PREPARE_ONLY.set(True)
    try:
        for file in files:
            LOG.info(f"Running evaluation from {file}")
            file = os.path.abspath(file)
            name = "user_module" + file

            spec = importlib.util.spec_from_file_location(name, file)
            if spec is None or spec.loader is None:
                LOG.error(f"Could not load module specification from {file}")
                if args.fail_on_error:
                    return
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod

            spec.loader.exec_module(mod)
            evaluations: list[Evaluation] | None = EVALUATION_INSTANCES.get()
            if evaluations is None:
                LOG.warning("Evaluation instance not found")
                if args.fail_on_error:
                    return
                continue

            LOG.info(f"Loaded {len(evaluations)} evaluations from {file}")

            for evaluation in evaluations:
                try:
                    await evaluation.run()
                except Exception as e:
                    LOG.error(f"Error running evaluation: {e}")
                    if args.fail_on_error:
                        raise
    finally:
        PREPARE_ONLY.reset(prep_token)


async def bundle_evaluation(args):
    """Bundle evaluation code (creates directory, no zip until upload)"""
    bundler = RemoteEvaluationBundler()
    
    try:
        bundle_path = await bundler.create_bundle(
            eval_file=args.file,
            output_dir=args.output_dir,
            include_current_env=args.include_current_env,
            pyproject_file=getattr(args, 'pyproject', None)
        )
        LOG.info(f"Successfully created evaluation bundle directory: {bundle_path}")
        
        # Inspect bundle contents if requested
        if args.inspect:
            await bundler.inspect_bundle_contents(bundle_path)
        
        # Note about next steps
        LOG.info("💡 Next steps:")
        LOG.info(f"  • Test locally: lmnr eval dry-run {bundle_path}")
        LOG.info(f"  • Submit for remote execution: lmnr eval submit {bundle_path} <dataset_name>")
        LOG.info(f"  • Create zip for sharing: lmnr eval push {bundle_path}")
        
        return bundle_path
        
    except Exception as e:
        LOG.error(f"Failed to create bundle: {e}")
        if getattr(args, 'fail_on_error', False):
            raise
        return None


async def dry_run_evaluation(args):
    """Test bundled evaluation locally before upload"""
    bundler = RemoteEvaluationBundler()
    
    try:
        bundle_path = Path(args.bundle_path)
        if not bundle_path.exists():
            LOG.error(f"Bundle not found: {args.bundle_path}")
            return False
        
        # Check if it's a directory or zip file
        if bundle_path.is_dir():
            # It's already a directory, use it directly
            test_dir = bundle_path
            LOG.info(f"Testing bundle directory: {test_dir}")
        else:
            # It's a zip file, extract it for testing
            test_dir = Path(args.output_dir) / f"dry_run_{bundle_path.stem}_{uuid.uuid4().hex[:8]}"
            await bundler.extract_bundle_for_inspection(str(bundle_path), test_dir)
            LOG.info(f"Bundle extracted for testing: {test_dir}")
        
        # Show bundle analysis
        await bundler.inspect_bundle_contents(str(test_dir))
        
        # Test execution scenarios
        success = True
        
        # Scenario 1: Test with dataset data (if dataset name provided)
        if args.dataset_name:
            success = await _test_with_dataset(test_dir, args.dataset_name, args.num_points or 5, args)
        
        # Scenario 2: Test with manual datapoint (if provided)
        elif args.test_datapoint:
            success = await _test_with_manual_datapoint(test_dir, args)
        
        else:
            LOG.info("💡 No test data provided. Use --dataset-name or --test-datapoint to run execution tests")
        
        # Clean up extracted files only if we extracted them and not requested to keep
        if not bundle_path.is_dir() and not args.keep_extracted:
            import shutil
            shutil.rmtree(test_dir)
            LOG.info("Cleaned up extracted test files")
        elif not bundle_path.is_dir() and args.keep_extracted:
            LOG.info(f"Test files kept at: {test_dir}")
        
        if success:
            LOG.info("✅ Dry run completed successfully")
        else:
            LOG.error("❌ Dry run failed")
        return success
        
    except Exception as e:
        LOG.error(f"Dry run failed: {e}")
        if args.fail_on_error:
            raise
        return False


async def _test_with_dataset(extract_dir: Path, dataset_name: str, num_points: int, args=None) -> bool:
    """Test bundle with first n datapoints from LaminarDataset"""
    from .sdk.datasets import LaminarDataset
    from .sdk.client.synchronous.sync_client import LaminarClient
    from .sdk.utils import from_env
    import subprocess
    import json
    
    LOG.info(f"📊 Testing with dataset '{dataset_name}' (first {num_points} datapoints)")
    
    try:
        # Set up dataset and client
        api_key = from_env("LMNR_PROJECT_API_KEY")
        if not api_key:
            LOG.error("Project API key not found. Set LMNR_PROJECT_API_KEY environment variable")
            return False
        
        base_url = from_env("LMNR_BASE_URL") or "https://api.lmnr.ai"
        client = LaminarClient(base_url, api_key)
        
        dataset = LaminarDataset(dataset_name)
        dataset.set_client(client)
        
        # Get dataset size and validate
        dataset_size = len(dataset)
        LOG.info(f"📈 Dataset '{dataset_name}' contains {dataset_size} total datapoints")
        
        if dataset_size == 0:
            LOG.error(f"Dataset '{dataset_name}' is empty")
            return False
        
        # Get first n datapoints
        test_points = min(num_points, dataset_size)
        datapoints = dataset.slice(0, test_points)
        LOG.info(f"🧪 Testing with {test_points} datapoints")
        
        # Make run script executable
        run_script = extract_dir / "run"
        run_script.chmod(0o755)
        
        # Test each datapoint
        success_count = 0
        for i, datapoint in enumerate(datapoints):
            LOG.info(f"  Testing datapoint {i+1}/{test_points}...")
            
            # Convert datapoint to dict format expected by CLI
            datapoint_data = {
                "data": datapoint.data,
                "target": datapoint.target
            }
            
            # Prepare CLI arguments
            cmd_args = [
                "./run",  # Use relative path since we're setting cwd=extract_dir
                "--datapoint", json.dumps(datapoint_data)
                # executor and evaluators auto-detected from decorators
            ]
            
            # Execute the test
            result = subprocess.run(cmd_args, capture_output=True, text=True, cwd=extract_dir)
            
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout)
                    if output.get("success", False):
                        success_count += 1
                        LOG.info(f"    ✅ Success - Scores: {output.get('scores', {})}")
                    else:
                        LOG.error(f"    ❌ Execution failed: {output.get('error', 'Unknown error')}")
                except json.JSONDecodeError:
                    LOG.error(f"    ❌ Invalid JSON output: {result.stdout}")
            else:
                LOG.error(f"    ❌ Process failed: {result.stderr}")
        
        # Report results
        LOG.info(f"🎯 Test Results: {success_count}/{test_points} datapoints passed")
        
        if success_count == test_points:
            LOG.info("🎉 All test datapoints passed!")
            return True
        elif success_count > 0:
            LOG.warning(f"⚠️  {test_points - success_count} datapoints failed")
            return False
        else:
            LOG.error("💥 All test datapoints failed")
            return False
            
    except Exception as e:
        LOG.error(f"Dataset testing failed: {e}")
        return False
    finally:
        # Clean up client connection
        try:
            if 'client' in locals():
                client.close()
        except:
            pass


async def _test_with_manual_datapoint(extract_dir: Path, args) -> bool:
    """Test bundle with manually provided datapoint"""
    import json
    import subprocess
    
    LOG.info(f"🧪 Testing with manual datapoint...")
    
    try:
        datapoint = json.loads(args.test_datapoint)
        
        # Make run script executable
        run_script = extract_dir / "run"
        run_script.chmod(0o755)
        
        # Execute the test
        cmd_args = [
            "./run",  # Use relative path since we're setting cwd=extract_dir
            "--datapoint", json.dumps(datapoint)
        ]
        
        result = subprocess.run(cmd_args, capture_output=True, text=True, cwd=extract_dir)
        
        if result.returncode == 0:
            LOG.info("✅ Test execution successful!")
            LOG.info(f"Output: {result.stdout}")
            return True
        else:
            LOG.error("❌ Test execution failed!")
            LOG.error(f"Error: {result.stderr}")
            return False
            
    except json.JSONDecodeError as e:
        LOG.error(f"Invalid JSON in test parameters: {e}")
        return False
    except Exception as e:
        LOG.error(f"Test execution failed: {e}")
        return False


async def submit_evaluation(args):
    """Upload bundle and run remote evaluation"""
    try:
        bundler = RemoteEvaluationBundler()
        runner = RemoteEvaluationRunner()
        
        bundle_path = Path(args.bundle_path)
        if not bundle_path.exists():
            LOG.error(f"Bundle not found: {args.bundle_path}")
            return None
        
        # If it's a directory, create a zip file for upload
        if bundle_path.is_dir():
            LOG.info(f"Creating zip file from bundle directory: {bundle_path}")
            zip_file = await bundler.create_zip_from_bundle(str(bundle_path))
            upload_file = zip_file
        else:
            # It's already a zip file
            upload_file = str(bundle_path)
        
        LOG.info(f"Uploading bundle: {upload_file}")
        bundle_id = await runner.upload_bundle(upload_file)
        LOG.info(f"Bundle uploaded with ID: {bundle_id}")
        
        # Then run the remote evaluation
        LOG.info(f"Starting remote evaluation on dataset: {args.dataset_name}")
        result = await runner.run_remote_evaluation(
            bundle_id=bundle_id,
            dataset_name=args.dataset_name,
            evaluation_name=args.name,
            group_name=args.group_name,
            concurrency_limit=args.concurrency_limit
        )
        
        LOG.info("✅ Remote evaluation submitted successfully!")
        LOG.info(f"Evaluation ID: {result['evaluation_id']}")
        if 'evaluation_url' in result:
            LOG.info(f"View results: {result['evaluation_url']}")
        
        await runner.close()
        return result
        
    except Exception as e:
        LOG.error(f"Failed to submit evaluation: {e}")
        if args.fail_on_error:
            raise
        return None


async def inspect_bundle(args):
    """Inspect the contents of an existing bundle"""
    bundler = RemoteEvaluationBundler()
    
    try:
        bundle_path = Path(args.bundle_path)
        if not bundle_path.exists():
            LOG.error(f"Bundle not found: {args.bundle_path}")
            return
        
        LOG.info(f"Inspecting bundle: {bundle_path}")
        
        if bundle_path.is_dir():
            # It's a directory, inspect directly
            await bundler.inspect_bundle_contents(str(bundle_path))
            LOG.info(f"Bundle directory location: {bundle_path}")
        else:
            # It's a zip file, inspect the zip
            await bundler.inspect_bundle_contents(str(bundle_path))
            
            if args.extract:
                extract_dir = bundle_path.parent / f"extracted_{bundle_path.stem}"
                await bundler.extract_bundle_for_inspection(str(bundle_path), extract_dir)
                LOG.info(f"Bundle extracted to: {extract_dir}")
            
    except Exception as e:
        LOG.error(f"Failed to inspect bundle: {e}")
        if args.fail_on_error:
            raise


async def push_bundle(args):
    """Create zip from bundle directory and upload to Laminar servers"""
    bundler = RemoteEvaluationBundler()
    runner = RemoteEvaluationRunner()
    
    try:
        bundle_path = Path(args.bundle_path)
        if not bundle_path.exists():
            LOG.error(f"Bundle not found: {args.bundle_path}")
            return None
        
        # If it's a directory, create a zip file for upload
        if bundle_path.is_dir():
            LOG.info(f"Creating zip file from bundle directory: {bundle_path}")
            zip_file = await bundler.create_zip_from_bundle(str(bundle_path))
            upload_file = zip_file
        else:
            # It's already a zip file
            upload_file = str(bundle_path)
        
        LOG.info(f"Uploading bundle: {upload_file}")
        bundle_id = await runner.upload_bundle(upload_file)
        LOG.info(f"✅ Bundle uploaded successfully!")
        LOG.info(f"Bundle ID: {bundle_id}")
        LOG.info("💡 Next steps:")
        LOG.info(f"  • Run evaluation: lmnr eval run-remote {bundle_id} <dataset_name>")
        
        await runner.close()
        return bundle_id
        
    except Exception as e:
        LOG.error(f"Failed to push bundle: {e}")
        if args.fail_on_error:
            raise
        return None


async def zip_bundle(args):
    """Create zip file from bundle directory"""
    bundler = RemoteEvaluationBundler()
    
    try:
        bundle_path = Path(args.bundle_path)
        if not bundle_path.exists():
            LOG.error(f"Bundle not found: {args.bundle_path}")
            return None
        
        if not bundle_path.is_dir():
            LOG.error(f"Bundle path must be a directory: {args.bundle_path}")
            return None
        
        LOG.info(f"Creating zip file from bundle directory: {bundle_path}")
        zip_file = await bundler.create_zip_from_bundle(str(bundle_path))
        LOG.info(f"✅ Successfully created zip file: {zip_file}")
        LOG.info("💡 Next steps:")
        LOG.info(f"  • Upload and run: lmnr eval submit {zip_file} <dataset_name>")
        LOG.info(f"  • Test locally: lmnr eval dry-run {zip_file}")
        LOG.info(f"  • Inspect contents: lmnr eval inspect {zip_file}")
        
        return zip_file
        
    except Exception as e:
        LOG.error(f"Failed to create zip file: {e}")
        if args.fail_on_error:
            raise
        return None


async def upload_bundle(args):
    """Upload evaluation bundle to Laminar servers (legacy - expects zip file)"""
    runner = RemoteEvaluationRunner()
    
    try:
        bundle_id = await runner.upload_bundle(args.bundle_file)
        LOG.info(f"Successfully uploaded bundle with ID: {bundle_id}")
    except Exception as e:
        LOG.error(f"Failed to upload bundle: {e}")
        if args.fail_on_error:
            raise


async def run_remote_evaluation(args):
    """Run evaluation remotely on Laminar servers"""
    runner = RemoteEvaluationRunner()
    
    try:
        result = await runner.run_remote_evaluation(
            bundle_id=args.bundle_id,
            dataset_name=args.dataset_name,
            evaluation_name=args.name,
            group_name=args.group_name,
            concurrency_limit=args.concurrency_limit
        )
        LOG.info(f"Remote evaluation completed. Evaluation ID: {result['evaluation_id']}")
        LOG.info(f"View results at: {result['evaluation_url']}")
    except Exception as e:
        LOG.error(f"Failed to run remote evaluation: {e}")
        if args.fail_on_error:
            raise


async def bundle_evaluation_remote(args):
    """Send source files to Laminar for optimized remote bundling"""
    bundler = RemoteEvaluationBundler()
    
    try:
        result = await bundler.create_remote_bundle(
            eval_file=args.file,
            pyproject_file=getattr(args, 'pyproject', None),
            base_url=getattr(args, 'base_url', None)
        )
        
        if not result:
            LOG.error("Remote bundling failed")
            return None
        
        bundle_id = result.get("bundle_id")
        bundle_size = result.get("bundle_size", 0)
        
        LOG.info("💡 Next steps:")
        LOG.info(f"  • Run evaluation: lmnr eval run-remote {bundle_id} <dataset_name>")
        LOG.info(f"  • View in dashboard: Open your Laminar project")
        
        return result
        
    except Exception as e:
        LOG.error(f"Failed to create remote bundle: {e}")
        if args.fail_on_error:
            raise
        return None


def cli():
    parser = ArgumentParser(
        prog="lmnr",
        description="CLI for Laminar. Call `lmnr [subcommand] --help` for more information on each subcommand.",
    )

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # Main eval parser
    parser_eval = subparsers.add_parser(
        "eval",
        description="Run, bundle, or manage evaluations",
        help="Run, bundle, or manage evaluations",
    )
    
    eval_subparsers = parser_eval.add_subparsers(title="eval subcommands", dest="eval_subcommand")

    # Original eval run command (now as a subcommand)
    parser_eval_run = eval_subparsers.add_parser(
        "run",
        description="Run an evaluation locally",
        help="Run an evaluation locally",
    )
    parser_eval_run.add_argument(
        "file",
        nargs="?",
        help="A file containing the evaluation to run."
        + "If no file name is provided, all evaluation files in the `evals` directory are run as long"
        + "as they match *_eval.py or eval_*.py",
        default=None,
    )
    parser_eval_run.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Fail on error",
    )

    # Bundle command
    parser_eval_bundle = eval_subparsers.add_parser(
        "bundle",
        description="Bundle evaluation code for remote execution (creates directory)",
        help="Bundle evaluation code for remote execution",
    )
    parser_eval_bundle.add_argument(
        "file",
        help="Path to the evaluation file",
    )
    parser_eval_bundle.add_argument(
        "--output-dir",
        default="./bundles",
        help="Output directory for the bundle (default: ./bundles)",
    )
    parser_eval_bundle.add_argument(
        "--pyproject",
        help="Path to pyproject.toml file for dependencies",
    )
    parser_eval_bundle.add_argument(
        "--include-current-env",
        action="store_true",
        help="Include current environment dependencies",
    )
    parser_eval_bundle.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect bundle contents after creation",
    )
    parser_eval_bundle.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Fail on error",
    )

    # Bundle-remote command (NEW!)
    parser_eval_bundle_remote = eval_subparsers.add_parser(
        "bundle-remote",
        description="Send source files to Laminar for optimized remote bundling",
        help="Bundle evaluation code remotely (optimized for production environments)",
    )
    parser_eval_bundle_remote.add_argument(
        "file",
        help="Path to the evaluation file",
    )
    parser_eval_bundle_remote.add_argument(
        "--pyproject",
        help="Path to pyproject.toml file for dependencies",
    )
    parser_eval_bundle_remote.add_argument(
        "--base-url",
        help="Laminar API base URL",
    )
    parser_eval_bundle_remote.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Fail on error",
    )

    # Dry-run command
    parser_eval_dry_run = eval_subparsers.add_parser(
        "dry-run",
        description="Test bundled evaluation locally before upload",
        help="Test bundled evaluation locally",
    )
    parser_eval_dry_run.add_argument(
        "bundle_path",
        help="Path to the bundle directory or zip file to test",
    )
    parser_eval_dry_run.add_argument(
        "--test-datapoint",
        help="JSON string of test datapoint (alternative to --dataset-name)",
    )
    parser_eval_dry_run.add_argument(
        "--dataset-name",
        help="Name of LaminarDataset to test against (requires --num-points)",
    )
    parser_eval_dry_run.add_argument(
        "--num-points",
        type=int,
        default=5,
        help="Number of datapoints to test when using --dataset-name (default: 5)",
    )
    parser_eval_dry_run.add_argument(
        "--output-dir",
        default="./temp",
        help="Output directory for extracted files (default: ./temp)",
    )
    parser_eval_dry_run.add_argument(
        "--keep-extracted",
        action="store_true",
        default=False,
        help="Keep extracted files after dry run (for debugging)",
    )
    parser_eval_dry_run.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Fail on error",
    )

    # Submit command (upload and run)
    parser_eval_submit = eval_subparsers.add_parser(
        "submit",
        description="Upload bundle and run remote evaluation",
        help="Upload bundle and run remote evaluation",
    )
    parser_eval_submit.add_argument(
        "bundle_path",
        help="Path to the bundle directory or zip file to upload and run",
    )
    parser_eval_submit.add_argument(
        "dataset_name",
        help="Name of the LaminarDataset to evaluate",
    )
    parser_eval_submit.add_argument(
        "--name",
        help="Name for the evaluation run",
    )
    parser_eval_submit.add_argument(
        "--group-name",
        help="Group name for the evaluation",
    )
    parser_eval_submit.add_argument(
        "--concurrency-limit",
        type=int,
        default=50,
        help="Maximum concurrent evaluations (default: 50)",
    )
    parser_eval_submit.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Fail if any error occurs during submission",
    )

    # Push command (create zip and upload)
    parser_eval_push = eval_subparsers.add_parser(
        "push",
        description="Create zip from bundle directory and upload to Laminar servers",
        help="Create zip from bundle directory and upload to Laminar servers",
    )
    parser_eval_push.add_argument(
        "bundle_path",
        help="Path to the bundle directory or zip file to upload",
    )
    parser_eval_push.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Fail on error",
    )

    # Zip command (create zip from bundle directory)
    parser_eval_zip = eval_subparsers.add_parser(
        "zip",
        description="Create zip file from bundle directory",
        help="Create zip file from bundle directory",
    )
    parser_eval_zip.add_argument(
        "bundle_path",
        help="Path to the bundle directory to zip",
    )
    parser_eval_zip.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Fail on error",
    )

    # Upload command (kept for manual workflow)
    parser_eval_upload = eval_subparsers.add_parser(
        "upload",
        description="Upload evaluation bundle to Laminar servers",
        help="Upload evaluation bundle to Laminar servers",
    )
    parser_eval_upload.add_argument(
        "bundle_file",
        help="Path to the bundle file to upload",
    )
    parser_eval_upload.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Fail on error",
    )

    # Run remote command (kept for manual workflow)  
    parser_eval_remote = eval_subparsers.add_parser(
        "run-remote",
        description="Run evaluation remotely on Laminar servers",
        help="Run evaluation remotely on Laminar servers",
    )
    parser_eval_remote.add_argument(
        "bundle_id",
        help="ID of the uploaded bundle to run",
    )
    parser_eval_remote.add_argument(
        "dataset_name",
        help="Name of the LaminarDataset to evaluate",
    )
    parser_eval_remote.add_argument(
        "--name",
        help="Name for the evaluation run",
    )
    parser_eval_remote.add_argument(
        "--group-name",
        help="Group name for the evaluation",
    )
    parser_eval_remote.add_argument(
        "--concurrency-limit",
        type=int,
        default=50,
        help="Maximum number of concurrent serverless function calls (default: 50)",
    )
    parser_eval_remote.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Fail on error",
    )

    # Inspect command
    parser_eval_inspect = eval_subparsers.add_parser(
        "inspect",
        description="Inspect the contents of an evaluation bundle",
        help="Inspect the contents of an evaluation bundle",
    )
    parser_eval_inspect.add_argument(
        "bundle_path",
        help="Path to the bundle directory or zip file to inspect",
    )
    parser_eval_inspect.add_argument(
        "--extract",
        action="store_true",
        default=False,
        help="Extract bundle contents for detailed inspection",
    )
    parser_eval_inspect.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Fail on error",
    )

    # For backward compatibility, also support the old direct eval command
    parser_eval_legacy = subparsers.add_parser(
        "eval-legacy",
        description="Run an evaluation (legacy command)",
        help="Run an evaluation (legacy command)",
    )
    parser_eval_legacy.add_argument(
        "file",
        nargs="?",
        help="A file containing the evaluation to run.",
        default=None,
    )
    parser_eval_legacy.add_argument(
        "--fail-on-error",
        action="store_true",
        default=False,
        help="Fail on error",
    )

    parsed = parser.parse_args()
    
    if parsed.subcommand == "eval":
        if parsed.eval_subcommand == "run":
            asyncio.run(run_evaluation(parsed))
        elif parsed.eval_subcommand == "bundle":
            asyncio.run(bundle_evaluation(parsed))
        elif parsed.eval_subcommand == "bundle-remote":
            asyncio.run(bundle_evaluation_remote(parsed))
        elif parsed.eval_subcommand == "dry-run":
            asyncio.run(dry_run_evaluation(parsed))
        elif parsed.eval_subcommand == "submit":
            asyncio.run(submit_evaluation(parsed))
        elif parsed.eval_subcommand == "push":
            asyncio.run(push_bundle(parsed))
        elif parsed.eval_subcommand == "zip":
            asyncio.run(zip_bundle(parsed))
        elif parsed.eval_subcommand == "upload":
            asyncio.run(upload_bundle(parsed))
        elif parsed.eval_subcommand == "run-remote":
            asyncio.run(run_remote_evaluation(parsed))
        elif parsed.eval_subcommand == "inspect":
            asyncio.run(inspect_bundle(parsed))
        else:
            # If no subcommand, show help
            parser_eval.print_help()
    elif parsed.subcommand == "eval-legacy":
        asyncio.run(run_evaluation(parsed))
    else:
        parser.print_help()
