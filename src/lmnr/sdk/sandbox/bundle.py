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
    
    # Source file containing the executor
    entry_file: Path
    
    # Project root directory (where pyproject.toml is located)
    project_root: Path
    
    # Name of the executor function to call
    executor_name: str
    
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
        project_root = cls._find_project_root(entry_file)
        
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
                    eval_relative = eval_path.relative_to(project_root)
                    eval_module = str(eval_relative.with_suffix("")).replace("/", ".")
                    evaluator_names[name] = (func.__name__, eval_module)
        
        return cls(
            entry_file=entry_file,
            project_root=project_root,
            executor_name=executor_name,
            evaluator_names=evaluator_names,
        )

    @staticmethod
    def _find_project_root(start_file: Path) -> Path:
        """Walk up directory tree to find pyproject.toml"""
        current = start_file.parent
        for _ in range(10):  # Max 10 levels up
            if (current / "pyproject.toml").exists():
                return current
            if current.parent == current:  # Reached filesystem root
                break
            current = current.parent
        
        # Fallback: use the file's directory
        return start_file.parent

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
        # Get the module name from entry file
        relative_path = self.entry_file.relative_to(self.project_root)
        module_path = str(relative_path.with_suffix("")).replace("/", ".")
        
        # Serialize args to JSON
        args_json = json.dumps(args[0]) if args else "{}"
        
        # Build Python code to execute
        python_code = f"""
import json
import sys
sys.path.insert(0, '.')
from {module_path} import {func_name}
data = json.loads('''{args_json}''')
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
        # Get the module name from entry file (for executor)
        relative_path = self.entry_file.relative_to(self.project_root)
        executor_module = str(relative_path.with_suffix("")).replace("/", ".")
        
        # Serialize args to JSON
        data_json = json.dumps(executor_args[0]) if executor_args else "{}"
        target_json = json.dumps(target)
        
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
import json
import sys
sys.path.insert(0, '.')

# Import executor
from {executor_module} import {self.executor_name}

# Import evaluators (may be from different modules)
{evaluator_imports_str}

# Map evaluator names to functions
evaluators = {{{evaluator_mapping}}}

# Run executor
data = json.loads('''{data_json}''')
output = {self.executor_name}(data)

# Run evaluators
target = json.loads('''{target_json}''')
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

# Output combined result
print(json.dumps({{"output": output, "scores": scores}}))
"""
        
        # Use uv run to execute within the virtual environment
        return ["uv", "run", "python", "-c", python_code]
