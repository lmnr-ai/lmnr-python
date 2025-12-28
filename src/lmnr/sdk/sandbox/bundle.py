import base64
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Bundle:
    """
    Represents a bundled evaluation code package for remote execution.
    
    Contains all information needed to execute an evaluation remotely:
    source files, function names, and dependencies.
    """
    
    # Source file containing the executor (may be None for installed packages)
    entry_file: Path | None
    
    # Project root directory (where pyproject.toml is located)
    project_root: Path
    
    # Name of the executor function to call
    executor_name: str
    
    # Module path for the executor (e.g., "agent.core" or "lmnr.decorators")
    executor_module: str
    
    # Mapping of evaluator display names to (function_name, module_path)
    # e.g., {"accuracy": ("check_accuracy", "myevals.accuracy")}
    # Evaluators can be in different files from the executor
    evaluator_names: dict[str, tuple[str, str]]

    @classmethod
    def from_functions(
        cls,
        executor: Callable[..., Any],
        evaluators: dict[str, Callable[..., Any]],
    ) -> "Bundle":
        """
        Create a Bundle by introspecting executor and evaluator functions.
        
        Args:
            executor: The executor function
            evaluators: Dict of evaluator name to evaluator function/HumanEvaluator
            
        Returns:
            A Bundle instance with source file and function info extracted
        """
        # Get source file from executor
        source_file = inspect.getsourcefile(executor)
        if source_file is None:
            raise ValueError("Cannot determine source file for executor")
        entry_file = Path(source_file).resolve()
        
        # Find project root (walk up to find pyproject.toml)
        # Start from current working directory if executor is from installed package
        entry_file_str = str(entry_file)
        executor_is_installed = (
            ".venv" in entry_file_str or 
            "site-packages" in entry_file_str
        )
        
        if executor_is_installed:
            # Use current working directory to find project root
            project_root = cls._find_project_root(Path.cwd())
            # Use the actual module name for installed packages
            executor_module = executor.__module__
            # Don't track entry_file for installed packages
            entry_file_resolved = None
        else:
            project_root = cls._find_project_root(entry_file)
            # Calculate module path from file path for project files
            relative_path = entry_file.relative_to(project_root)
            executor_module = str(relative_path.with_suffix("")).replace("/", ".")
            entry_file_resolved = entry_file
        
        # Get function names
        executor_name = executor.__name__
        
        # Extract evaluator function names AND their source files
        # (evaluators can be in different files from executor)
        evaluator_names = {}
        for name, func in evaluators.items():
            if callable(func) and hasattr(func, "__name__"):
                eval_source = inspect.getsourcefile(func)
                if eval_source:
                    eval_path = Path(eval_source).resolve()
                    
                    # Check if evaluator is from an installed package (in .venv or site-packages)
                    # If so, use its __module__ directly instead of calculating from file path
                    eval_path_str = str(eval_path)
                    is_installed_package = (
                        ".venv" in eval_path_str or 
                        "site-packages" in eval_path_str or
                        not eval_path_str.startswith(str(project_root))
                    )
                    
                    if is_installed_package:
                        # Use the actual module name for installed packages
                        eval_module = func.__module__
                    else:
                        # For project files, calculate relative module path
                        eval_relative = eval_path.relative_to(project_root)
                        eval_module = str(eval_relative.with_suffix("")).replace("/", ".")
                    
                    evaluator_names[name] = (func.__name__, eval_module)
        
        return cls(
            entry_file=entry_file_resolved,
            project_root=project_root,
            executor_name=executor_name,
            executor_module=executor_module,
            evaluator_names=evaluator_names,
        )

    @staticmethod
    def _find_project_root(start_path: Path) -> Path:
        """Walk up directory tree to find pyproject.toml"""
        # Handle both files and directories
        current = start_path if start_path.is_dir() else start_path.parent
        
        for _ in range(10):  # Max 10 levels up
            if (current / "pyproject.toml").exists():
                return current
            if current.parent == current:  # Reached filesystem root
                break
            current = current.parent
        
        # Fallback: use the starting directory
        return start_path if start_path.is_dir() else start_path.parent

    def to_files(self) -> dict[str, bytes]:
        """
        Get files to upload to sandbox as a dict of path -> content.
        
        Includes all Python source files and project configuration files
        from the project root, excluding common non-essential directories.
        
        Returns:
            Dict mapping relative file paths to file contents
        """
        files = {}
        
        # Directories to exclude
        exclude_dirs = {
            ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
            "node_modules", ".mypy_cache", ".ruff_cache", "dist", "build",
            ".eggs", "*.egg-info", ".tox", ".nox",
        }
        
        # File extensions to include
        include_extensions = {".py", ".toml", ".txt", ".json", ".yaml", ".yml", ".lock"}
        
        # Walk the project directory
        for path in self.project_root.rglob("*"):
            if not path.is_file():
                continue
            
            # Skip excluded directories
            parts = path.relative_to(self.project_root).parts
            if any(part in exclude_dirs or part.endswith(".egg-info") for part in parts):
                continue
            
            # Include files with matching extensions
            if path.suffix in include_extensions:
                relative_path = path.relative_to(self.project_root)
                content = path.read_bytes()
                
                # Sanitize pyproject.toml to remove local path dependencies
                if path.name == "pyproject.toml":
                    content = self._sanitize_pyproject(content)
                
                files[str(relative_path)] = content
        
        return files
    
    def _sanitize_pyproject(self, content: bytes) -> bytes:
        """
        Remove local path dependencies from pyproject.toml.
        
        Handles two formats:
        1. Inline: lmnr @ file:///Users/foo/lmnr-python -> lmnr
        2. UV sources section: lmnr = { path = "/path" } -> (removed)
        
        This allows the sandbox to install packages from PyPI instead
        of trying to use non-existent local paths.
        """
        text = content.decode("utf-8")
        
        # Pattern to match: "package @ file:///path/to/local"
        # Replace with just the package name
        # Handles both quoted and unquoted versions
        patterns = [
            # "package @ file:///path" -> "package"
            r'"([a-zA-Z0-9_-]+)\s*@\s*file://[^"]*"',
            # 'package @ file:///path' -> 'package'  
            r"'([a-zA-Z0-9_-]+)\s*@\s*file://[^']*'",
            # package @ file:///path (unquoted, in arrays)
            r'([a-zA-Z0-9_-]+)\s*@\s*file://[^\s,\]]+',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, r'"\1"', text)
        
        # Remove [tool.uv.sources] entries with local paths
        # Matches: lmnr = { path = "/some/path" } or lmnr = { path = "/some/path", ... }
        text = re.sub(
            r'^[a-zA-Z0-9_-]+\s*=\s*\{[^}]*path\s*=\s*["\'][^"\']*["\'][^}]*\}\s*$',
            '',
            text,
            flags=re.MULTILINE
        )
        
        # Remove empty [tool.uv.sources] section if all entries were removed
        # This cleans up the section header if it's now empty
        text = re.sub(
            r'\[tool\.uv\.sources\]\s*(?=\[|\Z)',
            '',
            text
        )
        
        return text.encode("utf-8")

    def get_execution_command(
        self,
        func_name: str,
        args: tuple[Any, ...],
    ) -> list[str]:
        """
        Generate command to execute a function in the sandbox.
        
        Args:
            func_name: Name of function to call
            args: Arguments to pass to function
            
        Returns:
            Command as list of strings
        """
        # Serialize args to JSON and encode as base64 to avoid escaping issues
        args_json = json.dumps(args[0]) if args else "{}"
        args_b64 = base64.b64encode(args_json.encode()).decode()
        
        # Build Python code to execute
        python_code = f"""
import base64
import json
import sys
sys.path.insert(0, '.')
from {self.executor_module} import {func_name}
data = json.loads(base64.b64decode('{args_b64}').decode())
result = {func_name}(data)
print(json.dumps(result))
"""
        
        # Use uv run to execute within the virtual environment
        return ["uv", "run", "python", "-c", python_code]

    def get_full_execution_command(
        self,
        executor_args: tuple[Any, ...],
        target: Any,
    ) -> list[str]:
        """
        Generate command to execute executor AND evaluators in the sandbox.
        
        This runs both in the same sandbox session, allowing evaluators to
        access any files created/modified by the executor.
        
        Args:
            executor_args: Arguments to pass to executor
            target: Target value for evaluators
            
        Returns:
            Command that outputs JSON: {"output": <executor_result>, "scores": {...}}
        """
        # Serialize args to JSON and encode as base64 to avoid escaping issues
        data_json = json.dumps(executor_args[0]) if executor_args else "{}"
        target_json = json.dumps(target)
        data_b64 = base64.b64encode(data_json.encode()).decode()
        target_b64 = base64.b64encode(target_json.encode()).decode()
        
        # Build import statements for each evaluator (may be from different modules)
        evaluator_imports = []
        for name, (func_name, eval_module) in self.evaluator_names.items():
            evaluator_imports.append(f"from {eval_module} import {func_name}")
        evaluator_imports_str = "\n".join(evaluator_imports)
        
        # Build evaluator mapping
        evaluator_mapping = ", ".join(
            f'"{name}": {func_name}' 
            for name, (func_name, _) in self.evaluator_names.items()
        )
        
        # Build Python code to execute both executor and evaluators
        python_code = f"""
import base64
import json
import sys
sys.path.insert(0, '.')

# Import executor
from {self.executor_module} import {self.executor_name}

# Import evaluators (may be from different modules)
{evaluator_imports_str}

# Map evaluator names to functions
evaluators = {{{evaluator_mapping}}}

# Run executor (decode base64 to avoid escaping issues with special chars)
data = json.loads(base64.b64decode('{data_b64}').decode())
output = {self.executor_name}(data)

# Run evaluators
target = json.loads(base64.b64decode('{target_b64}').decode())
scores = {{}}
for name, evaluator in evaluators.items():
    try:
        score = evaluator(output, target)
        if isinstance(score, (int, float, bool)):
            scores[name] = score
        elif isinstance(score, dict):
            scores.update(score)
        else:
            scores[name] = float(score)
    except Exception as e:
        print(f"Evaluator {{name}} failed: {{e}}", file=sys.stderr)
        scores[name] = None

# Write result to file (avoids mixing with user's stdout prints)
with open('/tmp/__lmnr_result__.json', 'w') as f:
    json.dump({{"output": output, "scores": scores}}, f)
"""
        
        # Use uv run to execute within the virtual environment
        return ["uv", "run", "python", "-c", python_code]
    
    # Path where the result file is written in the sandbox
    RESULT_FILE_PATH = "/tmp/__lmnr_result__.json"
