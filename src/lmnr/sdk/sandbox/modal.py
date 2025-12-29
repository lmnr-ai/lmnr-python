import io
import zipfile

import modal

from lmnr.sdk.sandbox.sandbox import Sandbox, ExecutionResult


class ModalSandbox(Sandbox):
    """
    Modal-based sandbox implementation.
    
    Uses Modal's Sandbox API to run code in isolated containers.
    See: https://modal.com/docs/guide/sandboxes
    
    Usage:
        sandbox = ModalSandbox(image="python:3.11-slim")
        await sandbox.start()
        await sandbox.upload_files({"main.py": b"print('hello')"})
        result = await sandbox.execute(["python", "main.py"])
        print(result.stdout)
        await sandbox.stop()
    """

    def __init__(
        self,
        image: str,
        app_name: str = "default-sandbox",
        timeout: int = 5 * 60,
        env: dict[str, str] | None = None,
        dockerfile: str | None = None,
    ):
        """
        Initialize the Modal sandbox.
        
        Args:
            image: Docker image to use (e.g., "python:3.11-slim")
            app_name: Modal app name to use
            timeout: Maximum sandbox lifetime in seconds (default: 5 minutes)
            env: Environment variables to set in the sandbox
            dockerfile: Path to Dockerfile to build (takes precedence over image)
        """
        super().__init__(image=image, env=env)
        self.app_name = app_name
        self.timeout = timeout
        self.dockerfile = dockerfile
        self._sandbox: modal.Sandbox | None = None
        self._app: modal.App | None = None

    async def start(self) -> None:
        """
        Start the Modal sandbox container.
        """
        # Get or create the Modal app
        self._app = modal.App.lookup(self.app_name, create_if_missing=True)
        
        # Build the image from Dockerfile or use registry image
        # Always ensure Python and uv are available by layering on top
        if self.dockerfile:
            base_image = modal.Image.from_dockerfile(self.dockerfile)
        else:
            base_image = modal.Image.from_registry(self.image)
        
        modal_image = (
            base_image
            .apt_install("python3", "python3-pip", "python3-venv", "curl", "unzip")
            .run_commands(
                # Use official uv installer (avoids PEP 668 externally-managed-environment error)
                "curl -LsSf https://astral.sh/uv/install.sh | sh",
                # Add uv to PATH
                "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> /etc/profile",
            )
            .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin"})
        )
        
        # Create the sandbox with environment variables
        modal.enable_output()
        self._sandbox = await modal.Sandbox.create.aio(
            app=self._app,
            image=modal_image,
            timeout=self.timeout,
            env=self.env if self.env else None,
        )

    async def upload_files(self, files: dict[str, bytes]) -> None:
        """
        Upload files to the Modal sandbox.
        
        Creates a zip archive of all files locally and uploads it as a single file,
        then extracts it in the sandbox. This is much faster than uploading files
        individually due to reduced network round-trips.
        
        Args:
            files: Mapping of relative file paths to file contents
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")
        
        # Create zip archive in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for path, content in files.items():
                if isinstance(content, str):
                    content = content.encode("utf-8")
                zf.writestr(path, content)
        
        zip_data = zip_buffer.getvalue()
        
        # Upload single zip file
        async with await self._sandbox.open.aio("/tmp/__bundle__.zip", "wb") as f:
            await f.write.aio(zip_data)
        
        # Extract zip in sandbox
        process = await self._sandbox.exec.aio("unzip", "-o", "-q", "/tmp/__bundle__.zip", "-d", ".")
        await process.wait.aio()
        
        # Clean up zip file
        process = await self._sandbox.exec.aio("rm", "/tmp/__bundle__.zip")
        await process.wait.aio()

    async def execute(self, command: list[str]) -> ExecutionResult:
        """
        Execute a command in the Modal sandbox.
        
        Args:
            command: Command to execute as list of arguments
            
        Returns:
            ExecutionResult with stdout, stderr, and return code
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")
        
        # Execute the command
        process = await self._sandbox.exec.aio(*command)
        
        # Read stdout and stderr
        stdout = await process.stdout.read.aio()
        stderr = await process.stderr.read.aio()
        
        # Wait for process to complete and get return code
        return_code = await process.wait.aio()
        
        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
        )

    async def stop(self) -> None:
        """
        Stop and clean up the Modal sandbox.
        """
        if self._sandbox is not None:
            await self._sandbox.terminate.aio()
            self._sandbox = None
