from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os

from lmnr.sdk.laminar import Laminar


@dataclass
class SandboxConfig:
    """Configuration for creating sandboxes."""
    
    default_image: str = "python:3.11-slim"
    timeout: int = 5 * 60  # Default 5 minutes
    env: dict[str, str] = field(default_factory=dict)
    env_file: str | None = None  # Path to .env file to load
    
    def __post_init__(self):
        """Load env vars from file if specified."""
        if self.env_file:
            file_env = self._parse_env_file(self.env_file)
            # Merge: env_file first, then env overrides
            merged = dict(file_env)
            merged.update(self.env)
            self.env = merged
    
    @staticmethod
    def _parse_env_file(path: str) -> dict[str, str]:
        """Parse a .env file and return a dict of key-value pairs."""
        env = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    # Parse KEY=VALUE (handle quotes)
                    if '=' in line:
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip()
                        # Remove surrounding quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        env[key] = value
        except FileNotFoundError:
            pass  # Silently ignore missing file
        return env
    
    def create_sandbox(self, image: str | None = None, dockerfile: str | None = None) -> "Sandbox":
        """
        Create a new sandbox instance based on this config.
        
        Args:
            image: Docker image to use. Falls back to default_image if not specified.
                   Ignored if dockerfile is set.
            dockerfile: Path to Dockerfile (per-datapoint, takes precedence over image)
        """
        from lmnr.sdk.sandbox.modal import ModalSandbox
        
        actual_image = image or self.default_image
        env = dict(self.env)  # Copy to avoid mutating original
        env["LMNR_SPAN_CONTEXT"] = Laminar.serialize_span_context()
        env["LMNR_PROJECT_API_KEY"] = os.getenv("LMNR_PROJECT_API_KEY") or ""
        
        return ModalSandbox(
            image=actual_image,
            dockerfile=dockerfile,
            timeout=self.timeout,
            app_name="evals",
            env=env,
        )


@dataclass
class ExecutionResult:
    """Result of executing a command in a sandbox."""
    
    stdout: str
    stderr: str
    return_code: int
    
    @property
    def success(self) -> bool:
        return self.return_code == 0


class Sandbox(ABC):
    """
    Abstract base class for sandbox execution environments.
    
    A sandbox provides isolated execution of code, allowing commands
    to run in remote environments like Modal, Docker containers, 
    or cloud functions.
    
    Usage:
        sandbox = ModalSandbox(image="python:3.11-slim")
        await sandbox.start()
        await sandbox.upload_files({
            "pyproject.toml": b"...",
            "src/main.py": b"...",
        })
        result = await sandbox.execute(["uv", "run", "python", "src/main.py"])
        await sandbox.stop()
    """

    def __init__(self, image: str, env: dict[str, str] | None = None):
        """
        Initialize the sandbox.
        
        Args:
            image: Docker image to use for this sandbox
            env: Environment variables to set in the sandbox
        """
        self.image = image
        self.env = env or {}

    @abstractmethod
    async def start(self) -> None:
        """
        Start the sandbox container.
        """
        pass

    @abstractmethod
    async def upload_files(self, files: dict[str, bytes]) -> None:
        """
        Upload files to the sandbox, preserving directory structure.
        
        Args:
            files: Mapping of relative file paths to file contents.
                   Paths can include directories, e.g.:
                   {
                       "pyproject.toml": b"...",
                       "src/main.py": b"...",
                       "src/utils/helpers.py": b"...",
                   }
        """
        pass

    @abstractmethod
    async def execute(self, command: list[str]) -> ExecutionResult:
        """
        Execute a command in the sandbox.
        
        The command runs with the uploaded files available in the
        working directory, preserving the original directory structure.
        
        Args:
            command: Command to execute as list of arguments,
                     e.g., ["uv", "run", "python", "src/main.py"]
            
        Returns:
            ExecutionResult with stdout, stderr, and return code
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop and clean up the sandbox container.
        """
        pass
