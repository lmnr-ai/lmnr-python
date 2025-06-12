import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from lmnr.cli import run_evaluation
from lmnr.sdk.evaluations import Evaluation


@pytest.fixture
def mock_args():
    """Create a mock args object with default values."""
    return Namespace(
        file=[],
        continue_on_error=False,
        output_file=None,
    )


@pytest.fixture
def mock_evaluation():
    """Create a mock evaluation that returns sample scores."""
    mock_eval = AsyncMock(spec=Evaluation)
    mock_eval.run.return_value = {"accuracy": 0.85, "precision": 0.92}
    return mock_eval


@pytest.fixture
def sample_eval_files():
    """Create temporary evaluation files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create evals directory
        evals_dir = Path(temp_dir) / "evals"
        evals_dir.mkdir()

        # Create sample eval files
        eval_files = [
            evals_dir / "test_eval.py",
            evals_dir / "eval_accuracy.py",
            evals_dir / "other_file.py",  # Should be ignored
        ]

        for file in eval_files:
            file.write_text("# Sample evaluation file")

        yield temp_dir, eval_files


@patch("lmnr.cli.os.getcwd")
@patch("lmnr.cli.os.listdir")
@patch("lmnr.cli.importlib.util.spec_from_file_location")
@patch("lmnr.cli.importlib.util.module_from_spec")
@patch("lmnr.cli.EVALUATION_INSTANCES")
@patch("lmnr.cli.PREPARE_ONLY")
@pytest.mark.asyncio
async def test_run_evaluation_auto_discovery(
    mock_prepare_only,
    mock_eval_instances,
    mock_module_from_spec,
    mock_spec_from_file_location,
    mock_listdir,
    mock_getcwd,
    mock_args,
    mock_evaluation,
):
    """Test auto-discovery of evaluation files in evals directory."""
    # Setup mocks
    mock_getcwd.return_value = "/test/path"
    mock_listdir.return_value = [
        "test_eval.py",
        "eval_accuracy.py",
        "other_file.py",
    ]

    # Mock spec and module loading
    mock_spec = Mock()
    mock_spec.loader = Mock()
    mock_spec_from_file_location.return_value = mock_spec
    mock_module = Mock()
    mock_module_from_spec.return_value = mock_module

    # Mock evaluation instances
    mock_eval_instances.get.return_value = [mock_evaluation]
    mock_token = Mock()
    mock_prepare_only.set.return_value = mock_token

    # Run the function
    await run_evaluation(mock_args)

    # Verify expected behavior
    assert mock_listdir.called
    assert mock_spec_from_file_location.call_count == 2  # Only eval files
    assert mock_evaluation.run.call_count == 2  # One for each eval file
    mock_prepare_only.reset.assert_called_once_with(mock_token)


@patch("lmnr.cli.glob.glob")
@patch("lmnr.cli.importlib.util.spec_from_file_location")
@patch("lmnr.cli.importlib.util.module_from_spec")
@patch("lmnr.cli.EVALUATION_INSTANCES")
@patch("lmnr.cli.PREPARE_ONLY")
@pytest.mark.asyncio
async def test_run_evaluation_explicit_files(
    mock_prepare_only,
    mock_eval_instances,
    mock_module_from_spec,
    mock_spec_from_file_location,
    mock_glob,
    mock_args,
    mock_evaluation,
):
    """Test running evaluation with explicitly specified files."""
    # Setup
    mock_args.file = ["test_eval.py", "another_eval.py"]
    mock_glob.side_effect = lambda pattern: (
        [pattern] if pattern.endswith(".py") else []
    )

    # Mock spec and module loading
    mock_spec = Mock()
    mock_spec.loader = Mock()
    mock_spec_from_file_location.return_value = mock_spec
    mock_module = Mock()
    mock_module_from_spec.return_value = mock_module

    # Mock evaluation instances
    mock_eval_instances.get.return_value = [mock_evaluation]
    mock_token = Mock()
    mock_prepare_only.set.return_value = mock_token

    # Run the function
    await run_evaluation(mock_args)

    # Verify expected behavior
    assert mock_spec_from_file_location.call_count == 2
    assert mock_evaluation.run.call_count == 2
    mock_prepare_only.reset.assert_called_once_with(mock_token)


@patch("lmnr.cli.os.listdir")
@patch("lmnr.cli.LOG")
@pytest.mark.asyncio
async def test_run_evaluation_no_eval_files_found(
    mock_log,
    mock_listdir,
    mock_args,
):
    """Test behavior when no evaluation files are found."""
    # Setup
    mock_listdir.return_value = ["other_file.py", "not_eval.txt"]

    # Run the function
    await run_evaluation(mock_args)

    # Verify logging
    mock_log.error.assert_called_with("No evaluation files found in `evals` directory")
    mock_log.info.assert_called_with(
        "Eval files must be located in the `evals` directory and must be named *_eval.py or eval_*.py"
    )


@patch("lmnr.cli.importlib.util.spec_from_file_location")
@patch("lmnr.cli.LOG")
@pytest.mark.asyncio
async def test_run_evaluation_module_load_error(
    mock_log,
    mock_spec_from_file_location,
    mock_args,
):
    """Test error handling when module specification cannot be loaded."""
    # Setup
    mock_args.file = ["test_eval.py"]
    mock_spec_from_file_location.return_value = None

    # Run the function
    await run_evaluation(mock_args)

    # Verify error logging
    mock_log.error.assert_called()
    assert "Could not load module specification" in str(mock_log.error.call_args)


@patch("lmnr.cli.importlib.util.spec_from_file_location")
@patch("lmnr.cli.importlib.util.module_from_spec")
@patch("lmnr.cli.EVALUATION_INSTANCES")
@patch("lmnr.cli.PREPARE_ONLY")
@patch("lmnr.cli.LOG")
@pytest.mark.asyncio
async def test_run_evaluation_no_evaluation_instances(
    mock_log,
    mock_prepare_only,
    mock_eval_instances,
    mock_module_from_spec,
    mock_spec_from_file_location,
    mock_args,
):
    """Test behavior when no evaluation instances are found."""
    # Setup
    mock_args.file = ["test_eval.py"]

    # Mock spec and module loading
    mock_spec = Mock()
    mock_spec.loader = Mock()
    mock_spec_from_file_location.return_value = mock_spec
    mock_module = Mock()
    mock_module_from_spec.return_value = mock_module

    # Mock no evaluation instances
    mock_eval_instances.get.return_value = None
    mock_token = Mock()
    mock_prepare_only.set.return_value = mock_token

    # Run the function
    await run_evaluation(mock_args)

    # Verify warning
    mock_log.warning.assert_called()
    assert "Evaluation instance not found" in str(mock_log.warning.call_args)


@patch("lmnr.cli.importlib.util.spec_from_file_location")
@patch("lmnr.cli.importlib.util.module_from_spec")
@patch("lmnr.cli.EVALUATION_INSTANCES")
@patch("lmnr.cli.PREPARE_ONLY")
@patch("lmnr.cli.LOG")
@pytest.mark.asyncio
async def test_run_evaluation_continue_on_error(
    mock_log,
    mock_prepare_only,
    mock_eval_instances,
    mock_module_from_spec,
    mock_spec_from_file_location,
    mock_args,
):
    """Test continue_on_error flag behavior."""
    # Setup
    mock_args.file = ["test_eval.py", "bad_eval.py"]
    mock_args.continue_on_error = True

    # Mock spec loading - first succeeds, second fails
    mock_spec = Mock()
    mock_spec.loader = Mock()
    mock_spec_from_file_location.side_effect = [mock_spec, None]
    mock_module = Mock()
    mock_module_from_spec.return_value = mock_module

    # Mock evaluation instances
    mock_evaluation = AsyncMock(spec=Evaluation)
    mock_evaluation.run.return_value = {"accuracy": 0.85}
    mock_eval_instances.get.return_value = [mock_evaluation]
    mock_token = Mock()
    mock_prepare_only.set.return_value = mock_token

    # Run the function
    await run_evaluation(mock_args)

    # Verify it continues and processes the first file
    assert mock_evaluation.run.call_count == 1
    mock_log.error.assert_called()


@patch("lmnr.cli.importlib.util.spec_from_file_location")
@patch("lmnr.cli.importlib.util.module_from_spec")
@patch("lmnr.cli.EVALUATION_INSTANCES")
@patch("lmnr.cli.PREPARE_ONLY")
@patch("lmnr.cli.LOG")
@pytest.mark.asyncio
async def test_run_evaluation_evaluation_error_no_continue(
    mock_log,
    mock_prepare_only,
    mock_eval_instances,
    mock_module_from_spec,
    mock_spec_from_file_location,
    mock_args,
):
    """Test error handling when evaluation fails and continue_on_error is False."""
    # Setup
    mock_args.file = ["test_eval.py"]
    mock_args.continue_on_error = False

    # Mock spec and module loading
    mock_spec = Mock()
    mock_spec.loader = Mock()
    mock_spec_from_file_location.return_value = mock_spec
    mock_module = Mock()
    mock_module_from_spec.return_value = mock_module

    # Mock evaluation that raises an error
    mock_evaluation = AsyncMock(spec=Evaluation)
    mock_evaluation.run.side_effect = Exception("Evaluation failed")
    mock_eval_instances.get.return_value = [mock_evaluation]
    mock_token = Mock()
    mock_prepare_only.set.return_value = mock_token

    # Run the function and expect it to raise
    with pytest.raises(Exception, match="Evaluation failed"):
        await run_evaluation(mock_args)

    # Verify error logging
    mock_log.error.assert_called()


@patch("lmnr.cli.importlib.util.spec_from_file_location")
@patch("lmnr.cli.importlib.util.module_from_spec")
@patch("lmnr.cli.EVALUATION_INSTANCES")
@patch("lmnr.cli.PREPARE_ONLY")
@patch("lmnr.cli.LOG")
@pytest.mark.asyncio
async def test_run_evaluation_evaluation_error_with_continue(
    mock_log,
    mock_prepare_only,
    mock_eval_instances,
    mock_module_from_spec,
    mock_spec_from_file_location,
    mock_args,
):
    """Test error handling when evaluation fails and continue_on_error is True."""
    # Setup
    mock_args.file = ["test_eval.py"]
    mock_args.continue_on_error = True

    # Mock spec and module loading
    mock_spec = Mock()
    mock_spec.loader = Mock()
    mock_spec_from_file_location.return_value = mock_spec
    mock_module = Mock()
    mock_module_from_spec.return_value = mock_module

    # Mock evaluation that raises an error
    mock_evaluation = AsyncMock(spec=Evaluation)
    mock_evaluation.run.side_effect = Exception("Evaluation failed")
    mock_eval_instances.get.return_value = [mock_evaluation]
    mock_token = Mock()
    mock_prepare_only.set.return_value = mock_token

    # Run the function - should not raise
    await run_evaluation(mock_args)

    # Verify error logging but function continues
    mock_log.error.assert_called()
    mock_prepare_only.reset.assert_called_once_with(mock_token)


@patch("lmnr.cli.sys.path")
@patch("lmnr.cli.os.getcwd")
@pytest.mark.asyncio
async def test_run_evaluation_adds_cwd_to_path(
    mock_getcwd,
    mock_sys_path,
    mock_args,
):
    """Test that current working directory is added to sys.path."""
    # Setup
    mock_getcwd.return_value = "/test/cwd"
    mock_sys_path.append = Mock()

    # Mock empty file list to avoid further processing
    with patch("lmnr.cli.os.listdir", return_value=[]):
        await run_evaluation(mock_args)

    # Verify sys.path.append was called with cwd
    mock_sys_path.append.assert_called_with("/test/cwd")
