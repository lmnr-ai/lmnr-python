from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SandboxType(Enum):
    """Supported sandbox types."""
    MODAL = "modal"


@dataclass
class SandboxConfig:
    """Configuration for creating sandboxes."""
    
    type: SandboxType
    default_image: str = "python:3.11-slim"
    timeout: int = 5 * 60  # Default 5 minutes
    env: dict[str, str] = field(default_factory=dict)
    
    def create_sandbox(self, image: str | None = None, dockerfile: str | None = None) -> "Sandbox":
        """
        Create a new sandbox instance based on this config.
        
        Args:
            image: Docker image to use. Falls back to default_image if not specified.
                   Ignored if dockerfile is set.
        """
        actual_image = image or self.default_image
        
        if self.type == SandboxType.MODAL:
            from lmnr.sdk.sandbox.modal import ModalSandbox
            return ModalSandbox(
                image=actual_image,
                dockerfile=dockerfile,
                timeout=self.timeout,
                app_name="evals",
                env=self.env,
            )
        else:
            raise ValueError(f"Unknown sandbox type: {self.type}")


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
