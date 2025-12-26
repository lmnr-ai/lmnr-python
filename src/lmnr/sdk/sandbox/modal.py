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
        if self.dockerfile:
            modal_image = modal.Image.from_dockerfile(self.dockerfile)
        else:
            modal_image = modal.Image.from_registry(self.image)
        
        # Create the sandbox with environment variables
        self._sandbox = modal.Sandbox.create(
            app=self._app,
            image=modal_image,
            timeout=self.timeout,
            env=self.env if self.env else None,
        )

    async def upload_files(self, files: dict[str, bytes]) -> None:
        """
        Upload files to the Modal sandbox.
        
        Args:
            files: Mapping of relative file paths to file contents
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")
        
        for path, content in files.items():
            # Create parent directories if needed
            parent_dir = "/".join(path.split("/")[:-1])
            if parent_dir:
                self._sandbox.exec("mkdir", "-p", parent_dir)
            
            # Write file content
            with self._sandbox.open(path, "w") as f:
                if isinstance(content, bytes):
                    f.write(content.decode("utf-8"))
                else:
                    f.write(content)

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
        process = self._sandbox.exec(*command)
        
        # Read stdout and stderr
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        
        # Wait for process to complete and get return code
        return_code = process.wait()
        
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
            self._sandbox.terminate()
            self._sandbox = None
