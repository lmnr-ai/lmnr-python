import asyncio
import os
import sys
import tempfile
import zipfile
import shutil
import subprocess
import json
import uuid
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, Optional

from .client.asynchronous.async_client import AsyncLaminarClient
from .datasets import LaminarDataset
from .log import get_default_logger
from .utils import from_env

LOG = get_default_logger(__name__)


class RemoteEvaluationBundler:
    """Handles bundling of evaluation code for remote execution"""
    
    def __init__(self):
        self._logger = LOG
    
    async def create_bundle(
        self,
        eval_file: str,
        output_dir: str = "./bundles",
        include_current_env: bool = False,
        pyproject_file: str | None = None
    ) -> str:
        """
        Create a self-contained bundle directory of evaluation code for remote execution.
        
        Args:
            eval_file: Path to the evaluation file
            output_dir: Output directory for the bundle
            include_current_env: Whether to include current environment dependencies
            pyproject_file: Optional path to pyproject.toml file for dependencies
            
        Returns:
            Path to the created bundle directory
        """
        eval_path = Path(eval_file).resolve()
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create bundle directory directly in output path
        bundle_name = f"eval_bundle_{eval_path.stem}_{uuid.uuid4().hex[:8]}"
        bundle_dir = output_path / bundle_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize UV project in bundle directory
        await self._init_uv_project(bundle_dir)
        
        # Copy evaluation file with evaluate() calls removed
        eval_dest = bundle_dir / "evaluation.py"
        await self._copy_evaluation_file_cleaned(eval_path, eval_dest)
        
        # Analyze dependencies using pyproject.toml if available, then fallback to imports
        dependencies, python_requirement = await self._analyze_dependencies_enhanced(
            eval_path, include_current_env, pyproject_file
        )
        await self._create_pyproject_toml(bundle_dir, dependencies, python_requirement)
        
        # Install dependencies with UV to create self-contained bundle
        installation_success = await self._install_dependencies_with_uv(bundle_dir, dependencies, python_requirement)
        if not installation_success:
            self._logger.error("Failed to install dependencies. Bundle creation aborted.")
            raise RuntimeError("Dependency installation failed")
        
        # Create CLI entry point for subprocess execution
        await self._create_cli_entry_point(bundle_dir)
        
        # Create bundle metadata
        await self._create_bundle_metadata(bundle_dir, eval_path)
        
        # Create executable script
        await self._create_bundle_executable(bundle_dir)
        
        self._logger.info(f"Self-contained bundle created: {bundle_dir}")
        return str(bundle_dir)
    
    async def create_zip_from_bundle(self, bundle_dir: str, output_file: str | None = None) -> str:
        """
        Create a zip file from a bundle directory for uploading.
        
        Args:
            bundle_dir: Path to the bundle directory
            output_file: Optional output file path. If not provided, creates next to bundle directory.
            
        Returns:
            Path to the created zip file
        """
        bundle_path = Path(bundle_dir)
        if not bundle_path.exists() or not bundle_path.is_dir():
            raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
        
        # Determine output file path
        if output_file:
            zip_path = Path(output_file)
        else:
            zip_path = bundle_path.parent / f"{bundle_path.name}.zip"
        
        # Create zip bundle
        await self._create_zip_bundle(bundle_path, zip_path)
        
        self._logger.info(f"Created zip bundle: {zip_path}")
        return str(zip_path)
    
    async def _init_uv_project(self, bundle_dir: Path):
        """Initialize a UV project in the bundle directory"""
        # Check if UV is available
        try:
            result = subprocess.run(
                ["uv", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            self._logger.debug(f"UV version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "UV is not available. Please install UV to use remote evaluations. "
                "Visit https://docs.astral.sh/uv/getting-started/installation/"
            )
        
        # Don't run uv init since it creates workspace conflicts
        # Just ensure UV can work in this directory
        self._logger.debug("UV available, proceeding with direct installation")
        
        # UV will work without init when we provide pyproject.toml directly
    
    async def _analyze_dependencies_enhanced(
        self, 
        eval_file: Path, 
        include_current_env: bool,
        pyproject_file: str | None = None
    ) -> tuple[list[str], str]:
        """Enhanced dependency analysis using pyproject.toml when available"""
        dependencies = ["lmnr"]  # Always include lmnr
        python_requirement = ">=3.10"  # Default
        
        # Try to find and use pyproject.toml file
        pyproject_deps, python_req = await self._get_dependencies_from_pyproject(eval_file, pyproject_file)
        if pyproject_deps:
            dependencies.extend(pyproject_deps)
            if python_req:
                python_requirement = python_req
            self._logger.info(f"Using dependencies from pyproject.toml: {len(pyproject_deps)} packages")
        else:
            # Fallback to import-based analysis
            self._logger.info("No pyproject.toml found, analyzing imports...")
            import_deps = await self._analyze_dependencies_from_imports(eval_file)
            dependencies.extend(import_deps)
        
        # Include current environment if requested
        if include_current_env:
            current_deps = await self._get_current_environment_deps()
            dependencies.extend(current_deps)
        
        # Remove duplicates and sort
        return sorted(list(set(dependencies))), python_requirement

    async def _get_dependencies_from_pyproject(
        self, 
        eval_file: Path, 
        pyproject_file: str | None = None
    ) -> tuple[list[str], str | None]:
        """Extract dependencies from pyproject.toml file"""
        pyproject_path = None
        
        if pyproject_file:
            # User-specified pyproject.toml file
            pyproject_path = Path(pyproject_file).resolve()
            if not pyproject_path.exists():
                self._logger.warning(f"Specified pyproject.toml not found: {pyproject_file}")
                return [], None
            self._logger.info(f"Using user-specified pyproject.toml: {pyproject_path}")
        else:
            # Auto-detect pyproject.toml
            pyproject_path = self._find_pyproject_toml(eval_file)
            if pyproject_path:
                self._logger.info(f"Auto-detected pyproject.toml: {pyproject_path}")
        
        if not pyproject_path:
            self._logger.info("No pyproject.toml found")
            return [], None
        
        try:
            content = pyproject_path.read_text()
            self._logger.debug(f"Pyproject.toml content preview: {content[:500]}...")
            
            dependencies = []
            
            # Parse dependencies using a more robust method that handles nested brackets
            import re
            
            # Look for [project] dependencies with better bracket handling
            project_deps_match = self._extract_dependencies_array(content, "project")
            if project_deps_match:
                dependencies.extend(project_deps_match)
                self._logger.debug(f"Found project dependencies: {project_deps_match}")
            else:
                self._logger.debug("No project dependencies found in pyproject.toml")
            
            # Look for [tool.poetry.dependencies] if it's a poetry project
            poetry_deps = self._extract_poetry_dependencies(content)
            if poetry_deps:
                dependencies.extend(poetry_deps)
                self._logger.debug(f"Found poetry dependencies: {poetry_deps}")
            
            # Remove duplicates and filter out problematic packages
            cleaned_deps = []
            for dep in dependencies:
                if dep and not self._is_problematic_dependency(dep):
                    cleaned_deps.append(dep)
                else:
                    self._logger.debug(f"Filtered out problematic dependency: {dep}")
            
            # Extract python requirement
            python_req = None
            python_req_match = re.search(r'requires-python\s*=\s*"([^"]+)"', content)
            if python_req_match:
                python_req = python_req_match.group(1)
                self._logger.debug(f"Found python requirement: {python_req}")
            
            self._logger.debug(f"Final cleaned dependencies: {cleaned_deps}")
            return cleaned_deps, python_req
            
        except Exception as e:
            self._logger.warning(f"Failed to parse pyproject.toml: {e}")
            return [], None

    def _extract_dependencies_array(self, content: str, section: str) -> list[str]:
        """Extract dependencies array from pyproject.toml, handling nested brackets properly"""
        import re
        
        # Find the start of the dependencies section
        if section == "project":
            pattern = r'\[project\].*?dependencies\s*=\s*\['
        else:
            return []
        
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            return []
        
        # Find the position after the opening bracket
        start_pos = match.end() - 1  # Position of the opening bracket
        
        # Now manually find the matching closing bracket
        bracket_count = 0
        end_pos = None
        
        for i in range(start_pos, len(content)):
            char = content[i]
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_pos = i
                    break
        
        if end_pos is None:
            self._logger.warning("Could not find closing bracket for dependencies array")
            return []
        
        # Extract the content between brackets
        deps_content = content[start_pos + 1:end_pos]
        
        # Parse dependencies using a simpler approach focused on quoted strings
        dependencies = []
        in_quotes = False
        quote_char = None
        current_dep = ""
        
        for i, char in enumerate(deps_content):
            if not in_quotes:
                # Looking for start of quoted dependency
                if char in ['"', "'"]:
                    in_quotes = True
                    quote_char = char
                    current_dep = ""
                # Skip everything else (whitespace, commas, etc.)
            else:
                # Inside quotes
                if char == quote_char and (i == 0 or deps_content[i-1] != '\\'):
                    # End of quoted dependency
                    in_quotes = False
                    quote_char = None
                    dep = current_dep.strip()
                    if dep and not dep.startswith('#'):
                        dependencies.append(dep)
                        self._logger.debug(f"Parsed dependency: {dep}")
                    current_dep = ""
                else:
                    # Add character to current dependency
                    current_dep += char
        
        # Handle case where quotes weren't closed properly
        if in_quotes and current_dep.strip():
            dep = current_dep.strip()
            if dep and not dep.startswith('#'):
                dependencies.append(dep)
                self._logger.debug(f"Parsed dependency (unclosed quote): {dep}")
        
        self._logger.debug(f"Total dependencies parsed: {len(dependencies)}")
        return dependencies

    def _extract_poetry_dependencies(self, content: str) -> list[str]:
        """Extract dependencies from poetry format"""
        import re
        
        poetry_deps_match = re.search(
            r'\[tool\.poetry\.dependencies\](.*?)(?=\[|\Z)', 
            content, 
            re.DOTALL | re.IGNORECASE
        )
        if poetry_deps_match:
            deps_section = poetry_deps_match.group(1)
            # Parse poetry-style dependencies (key = "version" format)
            poetry_deps = re.findall(r'^([a-zA-Z0-9_-]+)\s*=', deps_section, re.MULTILINE)
            # Filter out python itself
            poetry_deps = [dep for dep in poetry_deps if dep.lower() != 'python']
            return poetry_deps
        
        return []

    def _find_pyproject_toml(self, eval_file: Path) -> Path | None:
        """Find pyproject.toml file by traversing up the directory tree"""
        current_dir = eval_file.parent
        
        # Search up to 5 levels up the directory tree
        for _ in range(5):
            pyproject_path = current_dir / "pyproject.toml"
            if pyproject_path.exists():
                return pyproject_path
            
            parent = current_dir.parent
            if parent == current_dir:  # Reached filesystem root
                break
            current_dir = parent
        
        return None

    def _is_problematic_dependency(self, dep: str) -> bool:
        """Check if a dependency is known to be problematic"""
        # Remove version specifiers to get package name
        package_name = re.split(r'[<>=!]', dep)[0].strip()
        
        problematic_packages = {
            'pyside', 'pyside2', 'pyside6',  # Qt bindings often cause issues
        }
        
        return package_name.lower() in problematic_packages

    async def _analyze_dependencies_from_imports(self, eval_file: Path) -> list[str]:
        """Fallback: Analyze dependencies from import statements (old method)"""
        dependencies = []
        
        # Read the evaluation file and extract imports
        content = eval_file.read_text()
        import_pattern = r'^(?:from\s+(\S+)\s+import|import\s+(\S+))'
        
        # Check if this looks like a local project by looking for common indicators
        is_local_project = self._detect_local_project_context(eval_file)
        
        for line in content.split('\n'):
            line = line.strip()
            if match := re.match(import_pattern, line):
                module = match.group(1) or match.group(2)
                if module and not module.startswith('.') and module not in ['lmnr']:
                    # Map common modules to package names
                    package = self._map_import_to_package(module.split('.')[0], is_local_project)
                    if package:
                        dependencies.append(package)
        
        return dependencies
    
    def _detect_local_project_context(self, eval_file: Path) -> bool:
        """Detect if this evaluation file is part of a local project"""
        # Look for common local project indicators
        eval_dir = eval_file.parent
        
        # Check for common project files in the directory tree
        project_indicators = [
            'pyproject.toml', 'setup.py', 'requirements.txt', 
            '.git', 'src/', 'lib/', 'package.json'
        ]
        
        # Check current directory and parents (up to 3 levels)
        for level in range(4):
            check_dir = eval_dir
            for _ in range(level):
                check_dir = check_dir.parent
                if check_dir == check_dir.parent:  # Reached root
                    break
            
            for indicator in project_indicators:
                if (check_dir / indicator).exists():
                    self._logger.debug(f"Detected local project context: found {indicator} in {check_dir}")
                    return True
        
        return False
    
    async def _get_current_environment_deps(self) -> list[str]:
        """Get dependencies from current environment"""
        try:
            process = await asyncio.create_subprocess_exec(
                "uv", "pip", "freeze",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                current_deps = []
                for dep in stdout.decode().strip().split('\n'):
                    if '==' in dep and dep.strip():
                        pkg_name = dep.split('==')[0]
                        # Only include well-known packages, not local/editable ones
                        if not dep.startswith('-e ') and not pkg_name.startswith('.'):
                            current_deps.append(dep)
                return current_deps
        except Exception as e:
            self._logger.warning(f"Could not export current environment: {e}")
        
        return []
    
    def _map_import_to_package(self, import_name: str, is_local_project: bool = False) -> Optional[str]:
        """Map import names to PyPI package names"""
        # Built-in modules (never install these)
        builtin_modules = {
            'json', 'os', 'sys', 're', 'typing', 'asyncio', 'uuid', 'hashlib', 
            'pathlib', 'tempfile', 'shutil', 'subprocess', 'datetime', 'time',
            'collections', 'itertools', 'functools', 'math', 'random', 'string',
            'io', 'logging', 'warnings', 'copy', 'pickle', 'base64', 'urllib',
            'http', 'email', 'xml', 'html', 'csv', 'sqlite3', 'gzip', 'zipfile'
        }
        
        if import_name in builtin_modules:
            return None
        
        # Common local/project package names that shouldn't be installed from PyPI
        local_package_patterns = [
            'index',  # Common local package name
            'src', 'lib', 'utils', 'helpers', 'config', 'core', 'models',
            'services', 'handlers', 'middleware', 'database', 'api'
        ]
        
        # If it looks like a local project and the import matches common local patterns
        if is_local_project and import_name in local_package_patterns:
            self._logger.info(f"Skipping local package import: {import_name}")
            return None
        
        # Common mappings where import name differs from package name
        mappings = {
            'pydantic': 'pydantic',
            'openai': 'openai',
            'anthropic': 'anthropic',
            'httpx': 'httpx',
            'requests': 'requests',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'scipy': 'scipy',
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'bs4': 'beautifulsoup4',
            'yaml': 'PyYAML',
            'tqdm': 'tqdm',
            'click': 'click',
            'flask': 'flask',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'sqlalchemy': 'sqlalchemy',
            'alembic': 'alembic',
            'pytest': 'pytest',
            'black': 'black',
            'mypy': 'mypy',
            'flake8': 'flake8',
        }
        
        mapped_package = mappings.get(import_name, import_name)
        
        # Log what we're including
        if mapped_package != import_name:
            self._logger.debug(f"Mapped import '{import_name}' to package '{mapped_package}'")
        elif mapped_package not in builtin_modules:
            self._logger.debug(f"Including package: {mapped_package}")
        
        return mapped_package
    
    async def _create_pyproject_toml(self, bundle_dir: Path, dependencies: list[str], python_requirement: str):
        """Create pyproject.toml for the bundle"""
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
        self._logger.debug(f"Created pyproject.toml with {len(dependencies)} dependencies")
    
    def _generate_pyproject_toml(self, dependencies: list[str], python_requirement: str | None) -> str:
        """Generate pyproject.toml content for UV"""
        python_req = python_requirement or ">=3.10"
        
        content = f'''[project]
name = "eval-bundle"
version = "0.1.0"
description = "Evaluation bundle for Laminar remote execution"
dependencies = {json.dumps(dependencies, indent=2)}
requires-python = "{python_req}"

[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"
'''
        return content
    
    async def _install_dependencies_with_uv(self, bundle_dir: Path, dependencies: list[str], python_requirement: str | None) -> bool:
        """Install dependencies using UV package manager for local bundling"""
        try:
            self._logger.info(f"Installing {len(dependencies)} dependencies with UV")
            self._logger.debug(f"Dependencies to install: {dependencies}")
            
            # Generate pyproject.toml for UV
            pyproject_content = self._generate_pyproject_toml(dependencies, python_requirement)
            pyproject_path = bundle_dir / "pyproject.toml"
            pyproject_path.write_text(pyproject_content)
            self._logger.debug(f"Generated pyproject.toml at {pyproject_path}")
            
            # Create virtual environment for dependencies
            self._logger.debug("Creating virtual environment...")
            result = await asyncio.create_subprocess_exec(
                "uv", "venv", ".venv",
                cwd=bundle_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                self._logger.error(f"UV venv creation failed: {stderr.decode()}")
                return False
            
            # Install dependencies using UV pip
            self._logger.debug(f"Installing {len(dependencies)} dependencies...")
            cmd_args = [
                "uv", "pip", "install",
                "--python", ".venv/bin/python",  # Use the venv Python
            ] + dependencies  # Add all dependencies as arguments
            
            result = await asyncio.create_subprocess_exec(
                *cmd_args,
                cwd=bundle_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            self._logger.debug(f"uv pip install - Return code: {result.returncode}")
            if stdout:
                self._logger.debug(f"uv pip install stdout: {stdout.decode()}")
            if stderr:
                self._logger.debug(f"uv pip install stderr: {stderr.decode()}")
            
            if result.returncode != 0:
                self._logger.error(f"UV pip install failed: {stderr.decode()}")
                return False
            
            # Check if site-packages was created
            venv_lib = bundle_dir / ".venv" / "lib"
            site_packages = None
            
            if venv_lib.exists():
                python_dirs = [d for d in venv_lib.iterdir() if d.name.startswith('python')]
                if python_dirs:
                    site_packages = python_dirs[0] / "site-packages"
            
            if site_packages and site_packages.exists():
                # Count installed packages
                installed_packages = [d for d in site_packages.iterdir() if d.is_dir() and not d.name.startswith('_')]
                self._logger.info(f"Successfully installed {len(installed_packages)} packages in site-packages")
                self._logger.debug(f"Installed packages: {[p.name for p in installed_packages[:10]]}...")  # Show first 10
                return True
            else:
                self._logger.error(f"site-packages directory not found")
                return False
            
        except Exception as e:
            self._logger.error(f"Failed to install dependencies with UV: {e}")
            return False

    async def _create_cli_entry_point(self, bundle_dir: Path):
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
            )   :
                
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
        # Make it executable
        cli_path.chmod(0o755)
        self._logger.debug("Created CLI entry point with Laminar instrumentation")

    async def _create_bundle_executable(self, bundle_dir: Path):
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

# Check for direct site-packages directory (fallback)
if [ -d "$BUNDLE_DIR/site-packages" ]; then
    export PYTHONPATH="$BUNDLE_DIR/site-packages:$PYTHONPATH"
    echo "Using direct site-packages: $BUNDLE_DIR/site-packages"
fi

# Add any existing PYTHONPATH from environment
if [ -n "$PYTHONPATH_ORIG" ]; then
    export PYTHONPATH="$PYTHONPATH:$PYTHONPATH_ORIG"
fi

echo "Final PYTHONPATH: $PYTHONPATH"

# Execute the CLI with all arguments
exec python3 run_evaluation.py "$@"
'''
        
        executable_path = bundle_dir / "run"
        executable_path.write_text(executable_content)
        executable_path.chmod(0o755)
        self._logger.debug("Created bundle executable script with UV environment support")
    
    async def _create_bundle_metadata(self, bundle_dir: Path, eval_file: Path):
        """Create metadata file for the bundle"""
        # Calculate file hash for integrity checking
        file_hash = hashlib.sha256(eval_file.read_bytes()).hexdigest()
        
        # Analyze the evaluation file
        analysis = self._analyze_evaluation_file(eval_file)
        
        metadata = {
            "bundle_version": "1.0",
            "eval_file_name": eval_file.name,
            "eval_file_hash": file_hash,
            "created_at": str(uuid.uuid4()),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "analysis": analysis,
        }
        
        metadata_path = bundle_dir / "bundle_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        self._logger.debug("Created bundle metadata")
        self._logger.debug(f"Detected executor: {analysis.get('executor_func')}")
        self._logger.debug(f"Detected evaluators: {analysis.get('evaluators_config')}")
    
    async def _create_zip_bundle(self, bundle_dir: Path, output_file: Path):
        """Create zip file from bundle directory"""
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
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
        
        self._logger.debug(f"Created zip bundle: {output_file}")

    async def inspect_bundle_contents(self, bundle_path: str):
        """Inspect and display the contents of a bundle directory or zip file"""
        path = Path(bundle_path)
        if not path.exists():
            raise FileNotFoundError(f"Bundle not found: {bundle_path}")
        
        if path.is_dir():
            # It's a directory bundle
            await self._inspect_directory_bundle(path)
        else:
            # It's a zip file bundle
            await self._inspect_zip_bundle(path)
    
    async def _inspect_directory_bundle(self, bundle_dir: Path):
        """Inspect a bundle directory"""
        self._logger.info(f"Inspecting bundle directory: {bundle_dir.name}")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in bundle_dir.rglob('*') if f.is_file())
        self._logger.info(f"Total size: {total_size:,} bytes")
        
        # Display file list
        self._logger.info("\n📁 Bundle Contents:")
        all_files = []
        for file_path in bundle_dir.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(bundle_dir)
                size = file_path.stat().st_size
                all_files.append((str(rel_path), size))
        
        for file_name, size in sorted(all_files):
            self._logger.info(f"  {file_name:<30} ({size:,} bytes)")
        
        # Display metadata if available
        metadata_file = bundle_dir / 'bundle_metadata.json'
        if metadata_file.exists():
            self._logger.info("\n📋 Bundle Metadata:")
            metadata_content = metadata_file.read_text()
            metadata = json.loads(metadata_content)
            
            self._logger.info(f"  Bundle version: {metadata.get('bundle_version')}")
            self._logger.info(f"  Evaluation file: {metadata.get('eval_file_name')}")
            self._logger.info(f"  Python version: {metadata.get('python_version')}")
            
            analysis = metadata.get('analysis', {})
            if analysis:
                self._logger.info("\n🔍 Code Analysis:")
                executor = analysis.get('executor_func')
                if executor:
                    self._logger.info(f"  Executor function: {executor}")
                
                evaluators = analysis.get('evaluators_config', {})
                if evaluators:
                    self._logger.info("  Evaluators:")
                    for evaluator_name, func_name in evaluators.items():
                        if evaluator_name != func_name:
                            self._logger.info(f"    {evaluator_name}: {func_name} (custom name)")
                        else:
                            self._logger.info(f"    {evaluator_name}: {func_name}")
                
                functions = analysis.get('functions', [])
                if functions:
                    self._logger.info("  Functions found:")
                    for func in functions:
                        async_marker = "(async)" if func.get('is_async') else "(sync)"
                        decorator_info = ""
                        if func.get('decorator') == 'executor':
                            decorator_info = " [@executor]"
                        elif func.get('decorator') == 'evaluator':
                            evaluator_name = func.get('evaluator_name')
                            if evaluator_name and evaluator_name != func['name']:
                                decorator_info = f" [@evaluator(\"{evaluator_name}\")]"
                            else:
                                decorator_info = " [@evaluator]"
                        self._logger.info(f"    {func['name']} {async_marker} (line {func['line_number']}){decorator_info}")
        
        # Display dependencies if available
        pyproject_file = bundle_dir / 'pyproject.toml'
        if pyproject_file.exists():
            self._logger.info("\n📦 Dependencies:")
            pyproject_content = pyproject_file.read_text()
            
            # Parse dependencies from pyproject.toml
            import re
            deps_match = re.search(r'dependencies = \[(.*?)\]', pyproject_content, re.DOTALL)
            if deps_match:
                deps_content = deps_match.group(1)
                deps = [dep.strip().strip('"') for dep in deps_content.split(',') if dep.strip()]
                for dep in deps:
                    if dep:
                        self._logger.info(f"  {dep}")
        
        # Show site-packages info if it exists
        site_packages_dir = bundle_dir / "site-packages"
        venv_site_packages = None
        for venv_lib in bundle_dir.glob(".venv/lib/python*/site-packages"):
            if venv_lib.exists():
                venv_site_packages = venv_lib
                break
        
        if site_packages_dir.exists() or venv_site_packages:
            packages_dir = site_packages_dir if site_packages_dir.exists() else venv_site_packages
            package_count = len([p for p in packages_dir.iterdir() if p.is_dir()])
            self._logger.info(f"\n📦 Installed packages: {package_count}")
    
    async def _inspect_zip_bundle(self, bundle_path: Path):
        """Inspect a zip file bundle (original method)"""
        self._logger.info(f"Inspecting bundle: {bundle_path.name}")
        self._logger.info(f"File size: {bundle_path.stat().st_size:,} bytes")
        
        with zipfile.ZipFile(bundle_path, 'r') as zipf:
            # Display file list
            self._logger.info("\n📁 Bundle Contents:")
            file_list = zipf.namelist()
            for file_name in sorted(file_list):
                file_info = zipf.getinfo(file_name)
                size = file_info.file_size
                self._logger.info(f"  {file_name:<30} ({size:,} bytes)")
            
            # Display metadata if available
            if 'bundle_metadata.json' in file_list:
                self._logger.info("\n📋 Bundle Metadata:")
                metadata_content = zipf.read('bundle_metadata.json').decode('utf-8')
                metadata = json.loads(metadata_content)
                
                self._logger.info(f"  Bundle version: {metadata.get('bundle_version')}")
                self._logger.info(f"  Evaluation file: {metadata.get('eval_file_name')}")
                self._logger.info(f"  Python version: {metadata.get('python_version')}")
                
                analysis = metadata.get('analysis', {})
                if analysis:
                    self._logger.info("\n🔍 Code Analysis:")
                    executor = analysis.get('executor_func')
                    if executor:
                        self._logger.info(f"  Executor function: {executor}")
                    
                    evaluators = analysis.get('evaluators_config', {})
                    if evaluators:
                        self._logger.info("  Evaluators:")
                        for evaluator_name, func_name in evaluators.items():
                            if evaluator_name != func_name:
                                self._logger.info(f"    {evaluator_name}: {func_name} (custom name)")
                            else:
                                self._logger.info(f"    {evaluator_name}: {func_name}")
                    
                    functions = analysis.get('functions', [])
                    if functions:
                        self._logger.info("  Functions found:")
                        for func in functions:
                            async_marker = "(async)" if func.get('is_async') else "(sync)"
                            decorator_info = ""
                            if func.get('decorator') == 'executor':
                                decorator_info = " [@executor]"
                            elif func.get('decorator') == 'evaluator':
                                evaluator_name = func.get('evaluator_name')
                                if evaluator_name and evaluator_name != func['name']:
                                    decorator_info = f" [@evaluator(\"{evaluator_name}\")]"
                                else:
                                    decorator_info = " [@evaluator]"
                            self._logger.info(f"    {func['name']} {async_marker} (line {func['line_number']}){decorator_info}")
            
            # Display dependencies if available
            if 'pyproject.toml' in file_list:
                self._logger.info("\n📦 Dependencies:")
                pyproject_content = zipf.read('pyproject.toml').decode('utf-8')
                
                # Parse dependencies from pyproject.toml
                import re
                deps_match = re.search(r'dependencies = \[(.*?)\]', pyproject_content, re.DOTALL)
                if deps_match:
                    deps_content = deps_match.group(1)
                    deps = [dep.strip().strip('"') for dep in deps_content.split(',') if dep.strip()]
                    for dep in deps:
                        if dep:
                            self._logger.info(f"  {dep}")
    
    async def extract_bundle_for_inspection(self, bundle_file: str, extract_dir: Path):
        """Extract bundle contents to a directory for inspection"""
        bundle_path = Path(bundle_file)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle file not found: {bundle_file}")
        
        # Create extraction directory
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        self._logger.info(f"Extracting bundle to: {extract_dir}")
        
        with zipfile.ZipFile(bundle_path, 'r') as zipf:
            # Extract all files
            zipf.extractall(extract_dir)
            
            # Set proper permissions for executable files
            for file_info in zipf.filelist:
                file_path = extract_dir / file_info.filename
                if file_path.is_file():
                    # Check if this file should be executable
                    if file_info.filename.endswith('/run') or file_info.filename == 'run' or file_info.filename.endswith('/run_evaluation.py') or file_info.filename == 'run_evaluation.py':
                        # Set executable permissions
                        file_path.chmod(0o755)
                        self._logger.debug(f"Set executable permissions for: {file_info.filename}")
                    # Check if the file had executable permissions in the zip
                    elif hasattr(file_info, 'external_attr') and file_info.external_attr:
                        # Extract Unix permissions from external_attr
                        unix_permissions = (file_info.external_attr >> 16) & 0o777
                        if unix_permissions & 0o111:  # Check if any execute bit is set
                            file_path.chmod(unix_permissions)
                            self._logger.debug(f"Preserved executable permissions for: {file_info.filename}")
        
        # List extracted files
        extracted_files = list(extract_dir.rglob('*'))
        self._logger.info(f"Extracted {len(extracted_files)} files:")
        
        for file_path in sorted(extracted_files):
            if file_path.is_file():
                rel_path = file_path.relative_to(extract_dir)
                size = file_path.stat().st_size
                permissions = oct(file_path.stat().st_mode)[-3:]
                self._logger.info(f"  {rel_path} ({size:,} bytes) [{permissions}]")
        
        # Highlight key files for inspection
        key_files = ['evaluation.py', 'run_evaluation.py', 'run', 'pyproject.toml', 'bundle_metadata.json']
        found_key_files = []
        
        for key_file in key_files:
            key_path = extract_dir / key_file
            if key_path.exists():
                found_key_files.append(key_file)
        
        if found_key_files:
            self._logger.info(f"\n🔑 Key files for inspection:")
            for key_file in found_key_files:
                key_path = extract_dir / key_file
                permissions = oct(key_path.stat().st_mode)[-3:] if key_path.exists() else "???"
                self._logger.info(f"  {key_path} [{permissions}]")
        
        # Show site-packages info if it exists
        site_packages_dir = extract_dir / "site-packages"
        if site_packages_dir.exists():
            package_count = len([p for p in site_packages_dir.iterdir() if p.is_dir()])
            self._logger.info(f"\n📦 Installed packages in site-packages: {package_count}")

    def analyze_bundle_for_testing(self, bundle_file: str) -> Dict[str, Any]:
        """Analyze bundle and return structured information for testing"""
        bundle_path = Path(bundle_file)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle file not found: {bundle_file}")
        
        analysis = {
            "bundle_file": str(bundle_path),
            "file_size": bundle_path.stat().st_size,
            "contents": [],
            "metadata": None,
            "dependencies": [],
            "key_files": {},
            "has_site_packages": False,
            "site_packages_count": 0
        }
        
        with zipfile.ZipFile(bundle_path, 'r') as zipf:
            # Get file list
            analysis["contents"] = sorted(zipf.namelist())
            
            # Check for site-packages
            site_packages_files = [f for f in analysis["contents"] if f.startswith("site-packages/")]
            analysis["has_site_packages"] = len(site_packages_files) > 0
            if analysis["has_site_packages"]:
                # Count unique top-level packages
                packages = set()
                for f in site_packages_files:
                    parts = f.split('/')
                    if len(parts) >= 2:
                        packages.add(parts[1])
                analysis["site_packages_count"] = len(packages)
            
            # Extract metadata
            if 'bundle_metadata.json' in analysis["contents"]:
                metadata_content = zipf.read('bundle_metadata.json').decode('utf-8')
                analysis["metadata"] = json.loads(metadata_content)
            
            # Extract dependencies
            if 'pyproject.toml' in analysis["contents"]:
                pyproject_content = zipf.read('pyproject.toml').decode('utf-8')
                import re
                deps_match = re.search(r'dependencies = \[(.*?)\]', pyproject_content, re.DOTALL)
                if deps_match:
                    deps_content = deps_match.group(1)
                    analysis["dependencies"] = [
                        dep.strip().strip('"') 
                        for dep in deps_content.split(',') 
                        if dep.strip()
                    ]
            
            # Extract key file contents
            key_files = ['evaluation.py', 'run_evaluation.py', 'bundle_metadata.json']
            for key_file in key_files:
                if key_file in analysis["contents"]:
                    try:
                        content = zipf.read(key_file).decode('utf-8')
                        analysis["key_files"][key_file] = content
                    except UnicodeDecodeError:
                        analysis["key_files"][key_file] = "<binary content>"
        
        return analysis

    def _analyze_evaluation_file(self, eval_file: Path) -> Dict[str, Any]:
        """Analyze the evaluation file to extract function names and structure"""
        content = eval_file.read_text()
        
        # Extract function definitions with decorators
        functions = []
        executor_funcs = []
        evaluator_funcs = []
        evaluator_names = {}  # Maps function name to custom evaluator name
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for decorator patterns (both @decorator and @decorator() forms)
            if line.startswith('@executor') and ('()' in line or line == '@executor'):
                # Find the function definition on the next non-empty line
                func_line = self._find_next_function_def(lines, i + 1)
                if func_line:
                    func_info = self._parse_function_line(func_line, i + 1)
                    if func_info:
                        func_info['decorator'] = 'executor'
                        functions.append(func_info)
                        executor_funcs.append(func_info['name'])
            
            elif line.startswith('@evaluator'):
                # Parse evaluator decorator with optional name parameter
                evaluator_name = self._parse_evaluator_decorator(line)
                
                # Find the function definition on the next non-empty line  
                func_line = self._find_next_function_def(lines, i + 1)
                if func_line:
                    func_info = self._parse_function_line(func_line, i + 1)
                    if func_info:
                        func_info['decorator'] = 'evaluator'
                        func_info['evaluator_name'] = evaluator_name
                        functions.append(func_info)
                        evaluator_funcs.append(func_info['name'])
                        
                        # Map function name to custom evaluator name if provided
                        if evaluator_name and evaluator_name != func_info['name']:
                            evaluator_names[func_info['name']] = evaluator_name
                        else:
                            evaluator_names[func_info['name']] = func_info['name']
            
            # Also detect regular function definitions (without decorators)
            elif line.startswith('def ') or line.startswith('async def '):
                func_info = self._parse_function_line(line, i + 1)
                if func_info:
                    func_info['decorator'] = None
                    functions.append(func_info)
        
        # Validation: exactly one executor required
        if len(executor_funcs) == 0:
            raise ValueError(
                "No executor function found. Please add exactly one @executor() decorator to your execution function."
            )
        elif len(executor_funcs) > 1:
            raise ValueError(
                f"Multiple executor functions found: {executor_funcs}. "
                "Please use exactly one @executor() decorator."
            )
        
        # Validation: at least one evaluator required
        if len(evaluator_funcs) == 0:
            raise ValueError(
                "No evaluator functions found. Please add at least one @evaluator() decorator to your scoring functions."
            )
        
        # Create evaluators mapping using custom names if provided
        evaluators_config = {}
        for func_name in evaluator_funcs:
            evaluator_name = evaluator_names.get(func_name, func_name)
            evaluators_config[evaluator_name] = func_name
        
        # Select primary executor (should be exactly one now)
        primary_executor = executor_funcs[0]
        
        self._logger.info(f"✅ Validation passed: 1 executor, {len(evaluator_funcs)} evaluators")
        if evaluator_names:
            custom_names = [f"{func}→{name}" for func, name in evaluator_names.items() if func != name]
            if custom_names:
                self._logger.info(f"📝 Custom evaluator names: {', '.join(custom_names)}")
        
        return {
            "functions": functions,
            "executor_funcs": executor_funcs,
            "evaluator_funcs": evaluator_funcs,
            "evaluator_names": evaluator_names,
            "executor_func": primary_executor,
            "evaluators_config": evaluators_config,
            "has_decorators": True  # Always true now with validation
        }
    
    def _find_next_function_def(self, lines: list[str], start_idx: int) -> str | None:
        """Find the next function definition line after a decorator"""
        for i in range(start_idx, min(start_idx + 5, len(lines))):  # Look within next 5 lines
            line = lines[i].strip()
            if line.startswith('def ') or line.startswith('async def '):
                return line
        return None
    
    def _parse_function_line(self, line: str, line_number: int) -> Dict[str, Any] | None:
        """Parse a function definition line to extract function info"""
        # Extract function definitions
        function_pattern = r'^(async\s+)?def\s+(\w+)\s*\('
        match = re.match(function_pattern, line.strip())
        if match:
            is_async = bool(match.group(1))
            func_name = match.group(2)
            return {
                "name": func_name,
                "is_async": is_async,
                "line_number": line_number
            }
        return None

    def _parse_evaluator_decorator(self, line: str) -> str | None:
        """Parse @evaluator decorator line to extract custom name if provided"""
        import re
        
        # Handle different forms:
        # @evaluator
        # @evaluator()  
        # @evaluator("custom_name")
        # @evaluator('custom_name')
        
        if line.strip() == '@evaluator':
            return None  # Will use function name
        
        # Look for @evaluator() or @evaluator("name")
        match = re.match(r'@evaluator\s*\(\s*(?:"([^"]+)"|\'([^\']+)\'|)\s*\)', line.strip())
        if match:
            # Return the custom name if found, otherwise None
            return match.group(1) or match.group(2) or None
        
        # Fallback for malformed decorator
        return None

    async def _copy_evaluation_file_cleaned(self, source_path: Path, dest_path: Path):
        """Copy evaluation file (no cleaning needed with decorator approach)"""
        content = source_path.read_text()
        
        # With decorators, we can copy the file as-is since there are no evaluate() calls to remove
        # The decorators are just metadata and don't affect remote execution
        dest_path.write_text(content)
        self._logger.debug(f"Copied evaluation file to {dest_path}")



    async def create_remote_bundle(
        self,
        eval_file: str,
        pyproject_file: str | None = None,
        base_url: str | None = None
    ) -> Dict[str, Any] | None:
        """
        Create evaluation bundle remotely via Laminar API.
        
        Args:
            eval_file: Path to the evaluation file
            pyproject_file: Optional path to pyproject.toml file for dependencies
            base_url: Optional Laminar API base URL
            
        Returns:
            Dictionary with bundle information or None on failure
        """
        import base64
        import tempfile
        import requests
        
        try:
            eval_path = Path(eval_file).resolve()
            if not eval_path.exists():
                raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
            
            self._logger.info(f"📦 Preparing source files for remote bundling...")
            
            # Create temporary zip of source files
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                temp_zip_path = Path(temp_zip.name)
            
            try:
                with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add evaluation file
                    zipf.write(eval_path, eval_path.name)
                    
                    # Add pyproject.toml if it exists nearby
                    pyproject_path = None
                    if pyproject_file:
                        pyproject_path = Path(pyproject_file)
                    else:
                        # Auto-detect pyproject.toml
                        pyproject_path = self._find_pyproject_toml(eval_path)
                    
                    if pyproject_path and pyproject_path.exists():
                        zipf.write(pyproject_path, "pyproject.toml")
                        self._logger.info(f"📄 Including pyproject.toml from {pyproject_path}")
                
                # Read zip data and encode as base64
                zip_data = temp_zip_path.read_bytes()
                zip_base64 = base64.b64encode(zip_data).decode('utf-8')
                
                # Use Laminar API endpoint
                api_key = from_env("LMNR_PROJECT_API_KEY")
                if not api_key:
                    raise ValueError("Project API key not found. Set LMNR_PROJECT_API_KEY environment variable")
                
                base_url = base_url or from_env("LMNR_BASE_URL") or "https://api.lmnr.ai"
                # laminar_url = f"{base_url}/v1/evaluations/bundle"
                laminar_url = "https://lmnr-ai--laminar-evaluation-bundler-create-remote-bundle-dev.modal.run"
                
                # Extract dependencies from pyproject.toml if available
                dependencies = ["lmnr"]  # Default
                python_requirement = ">=3.10"
                
                if pyproject_path and pyproject_path.exists():
                    self._logger.info("🔍 Analyzing dependencies from pyproject.toml...")
                    pyproject_deps, python_req = await self._get_dependencies_from_pyproject(eval_path, str(pyproject_path))
                    if pyproject_deps:
                        dependencies.extend(pyproject_deps)
                        dependencies = sorted(list(set(dependencies)))  # Remove duplicates
                    if python_req:
                        python_requirement = python_req
                
                request_data = {
                    "source_zip_data": zip_base64,
                    "dependencies": dependencies,
                    "python_requirement": python_requirement,
                    "bundle_name": f"remote_{eval_path.stem}_{uuid.uuid4().hex[:8]}"
                }
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                self._logger.info(f"🚀 Sending to Laminar for optimized bundling...")
                self._logger.info(f"   Source size: {len(zip_data):,} bytes")
                self._logger.info(f"   Dependencies: {len(dependencies)} packages")
                self._logger.info(f"   Laminar endpoint: {laminar_url}")
                
                # Send to Laminar bundling service
                response = requests.post(
                    laminar_url,
                    json=request_data,
                    headers=headers,
                    timeout=1200  # 20 minutes timeout
                )
                
                if response.status_code != 200:
                    self._logger.error(f"Laminar bundling request failed: {response.status_code} - {response.text}")
                    return None
                
                result = response.json()
                
                if not result.get("success"):
                    self._logger.error(f"Remote bundling failed: {result.get('error')}")
                    return None
                
                # Success!
                bundle_id = result.get("bundle_id")
                bundle_size = result.get("bundle_size", 0)
                
                self._logger.info(f"✅ Remote bundle created successfully!")
                self._logger.info(f"   Bundle ID: {bundle_id}")
                self._logger.info(f"   Size: {bundle_size:,} bytes")
                self._logger.info(f"   Optimized for: Linux x86_64")
                
                return {
                    "bundle_id": bundle_id,
                    "bundle_size": bundle_size,
                    "bundled_remotely": True,
                    "dependencies": dependencies,
                    "python_requirement": python_requirement
                }
                
            finally:
                # Clean up temp file
                if temp_zip_path.exists():
                    temp_zip_path.unlink()
                    
        except Exception as e:
            self._logger.error(f"Failed to create remote bundle: {e}")
            return None


class RemoteEvaluationRunner:
    """Handles uploading bundles and running remote evaluations"""
    
    def __init__(
        self,
        project_api_key: str | None = None,
        base_url: str | None = None,
        http_port: int | None = None
    ):
        self._logger = LOG
        
        base_url = base_url or from_env("LMNR_BASE_URL") or "https://api.lmnr.ai"
        api_key = project_api_key or from_env("LMNR_PROJECT_API_KEY")
        
        if not api_key:
            raise ValueError(
                "Project API key is required. Set LMNR_PROJECT_API_KEY environment variable "
                "or pass project_api_key parameter."
            )
        
        self.client = AsyncLaminarClient(
            base_url=base_url,
            project_api_key=api_key,
            port=http_port
        )
        
    async def run_remote_evaluation(
        self,
        bundle_id: str,
        dataset_name: str,
        evaluation_name: str | None = None,
        group_name: str | None = None,
        concurrency_limit: int = 50
    ) -> Dict[str, Any]:
        """
        Run evaluation remotely using uploaded bundle.
        
        Args:
            bundle_id: ID of the uploaded bundle
            dataset_name: Name of the LaminarDataset to evaluate
            evaluation_name: Optional name for the evaluation
            group_name: Optional group name for the evaluation
            concurrency_limit: Maximum concurrent Modal function calls
            
        Returns:
            Dictionary with evaluation results and metadata
        """
        self._logger.info(f"Starting remote evaluation with bundle {bundle_id}")
        self._logger.info(f"Dataset: {dataset_name}, Concurrency: {concurrency_limit}")
        
        try:
            response = await self.client._remote_evals.start_remote_evaluation(
                bundle_id=bundle_id,
                dataset_name=dataset_name,
                evaluation_name=evaluation_name,
                group_name=group_name,
                concurrency_limit=concurrency_limit
            )
            
            evaluation_id = response.get("evaluation_id")
            if not evaluation_id:
                raise ValueError("Invalid response: missing evaluation_id")
            
            self._logger.info(f"Remote evaluation started with ID: {evaluation_id}")
            
            # Wait for completion or return immediately based on preference
            # For now, we'll return the initial response
            return {
                "evaluation_id": evaluation_id,
                "bundle_id": bundle_id,
                "dataset_name": dataset_name,
                "status": response.get("status", "started"),
                "evaluation_url": f"https://www.lmnr.ai/evaluations/{evaluation_id}"
            }
        except Exception as e:
            self._logger.error(f"Failed to start remote evaluation: {e}")
            raise
    
    async def get_evaluation_status(self, evaluation_id: str) -> Dict[str, Any]:
        """Get the status of a remote evaluation"""
        try:
            return await self.client._remote_evals.get_evaluation_status(evaluation_id)
        except Exception as e:
            self._logger.error(f"Failed to get evaluation status: {e}")
            raise
    
    async def close(self):
        """Close the client connection"""
        await self.client.close() 