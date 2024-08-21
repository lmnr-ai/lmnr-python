from pathlib import Path
import sys
import requests
from dotenv import find_dotenv, get_key
import importlib
import os
import click
import logging
from cookiecutter.main import cookiecutter
from pydantic.alias_generators import to_pascal

from lmnr.cli.zip import zip_directory
from lmnr.sdk.registry import Registry as Pipeline
from lmnr.sdk.remote_debugger import RemoteDebugger
from lmnr.types import NodeFunction

from .parser.parser import runnable_graph_to_template_vars

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import time

logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def cli():
    "CLI for Laminar AI Engine"


@cli.command(name="pull")
@click.argument("pipeline_name")
@click.argument("pipeline_version_name")
@click.option(
    "-p",
    "--project-api-key",
    help="Project API key",
)
@click.option(
    "-l",
    "--loglevel",
    help="Sets logging level",
)
def pull(pipeline_name, pipeline_version_name, project_api_key, loglevel):
    loglevel_str_to_val = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logging.basicConfig()
    logging.getLogger().setLevel(loglevel_str_to_val.get(loglevel, logging.WARNING))

    project_api_key = project_api_key or os.environ.get("LMNR_PROJECT_API_KEY")
    if not project_api_key:
        env_path = find_dotenv(usecwd=True)
        project_api_key = get_key(env_path, "LMNR_PROJECT_API_KEY")
    if not project_api_key:
        raise ValueError("LMNR_PROJECT_API_KEY is not set")

    headers = {"Authorization": f"Bearer {project_api_key}"}
    params = {
        "pipelineName": pipeline_name,
        "pipelineVersionName": pipeline_version_name,
    }
    res = requests.get(
        "https://api.lmnr.ai/v2/pipeline-version-by-name",
        headers=headers,
        params=params,
    )
    if res.status_code != 200:
        try:
            res_json = res.json()
        except Exception:
            raise ValueError(
                f"Error in fetching pipeline version: {res.status_code}\n{res.text}"
            )
        raise ValueError(
            f"Error in fetching pipeline version: {res.status_code}\n{res_json}"
        )

    pipeline_version = res.json()

    class_name = to_pascal(pipeline_name.replace(" ", "_").replace("-", "_"))

    context = {
        "pipeline_name": pipeline_name,
        "pipeline_version_name": pipeline_version_name,
        "class_name": class_name,
        # _tasks starts from underscore because we don't want it to be templated
        # some tasks contains LLM nodes which have prompts
        # which we don't want to be rendered by cookiecutter
        "_tasks": runnable_graph_to_template_vars(pipeline_version["runnableGraph"]),
    }

    logger.info(f"Context:\n{context}")
    cookiecutter(
        "https://github.com/lmnr-ai/lmnr-python-engine.git",
        output_dir=".",
        config_file=None,
        extra_context=context,
        no_input=True,
        overwrite_if_exists=True,
    )


@cli.command(name="deploy")
@click.argument("endpoint_id")
@click.option(
    "-p",
    "--project-api-key",
    help="Project API key",
)
def deploy(endpoint_id, project_api_key):
    project_api_key = project_api_key or os.environ.get("LMNR_PROJECT_API_KEY")
    if not project_api_key:
        env_path = find_dotenv(usecwd=True)
        project_api_key = get_key(env_path, "LMNR_PROJECT_API_KEY")
    if not project_api_key:
        raise ValueError("LMNR_PROJECT_API_KEY is not set")

    current_directory = Path.cwd()
    zip_file_path = current_directory / "archive.zip"

    zip_directory(current_directory, zip_file_path)

    try:
        url = f"https://api.lmnr.ai/v2/endpoints/{endpoint_id}/deploy-code"
        with open(zip_file_path, "rb") as f:
            headers = {
                "Authorization": f"Bearer {project_api_key}",
            }
            files = {"file": f}
            response = requests.post(url, headers=headers, files=files)

            if response.status_code != 200:
                raise ValueError(
                    f"Error in deploying code: {response.status_code}\n{response.text}"
                )
    except Exception:
        logging.exception("Error in deploying code")
    finally:
        Path.unlink(zip_file_path, missing_ok=True)


def _load_functions(cur_dir: str) -> dict[str, NodeFunction]:
    parent_dir, name = os.path.split(cur_dir)  # e.g. /Users/username, project_name

    # Needed to __import__ pipeline.py
    if sys.path[0] != parent_dir:
        sys.path.insert(0, parent_dir)
    # Needed to import src in pipeline.py and other files
    if cur_dir not in sys.path:
        sys.path.insert(0, cur_dir)

    module_name = f"{name}.pipeline"
    if module_name in sys.modules:
        # Reload the module to get the updated version
        importlib.reload(sys.modules[module_name])
    else:
        # Import the module for the first time
        __import__(module_name)
    
    module = sys.modules[module_name]

    matches = [v for v in module.__dict__.values() if isinstance(v, Pipeline)]
    if not matches:
        raise ValueError("No Pipeline found in the module")
    if len(matches) > 1:
        raise ValueError("Multiple Pipelines found in the module")
    pipeline = matches[0]

    return pipeline.functions

class SimpleEventHandler(PatternMatchingEventHandler):
    def __init__(self, project_api_key: str, session_id: str, functions: dict[str, NodeFunction]):
        super().__init__(ignore_patterns=["*.pyc*", "*.pyo", "**/__pycache__"])
        self.project_api_key = project_api_key
        self.session_id = session_id
        self.functions = functions
        self.debugger = RemoteDebugger(project_api_key, session_id, functions)
        self.debugger.start()

    def on_any_event(self, event):
        print(f"Files at {event.src_path} updated. Restarting debugger...")
        self.debugger.stop()
        self.functions = _load_functions(os.getcwd())
        self.debugger = RemoteDebugger(self.project_api_key, self.session_id, self.functions)
        self.debugger.start()

@cli.command(name="dev")
@click.option(
    "-p",
    "--project-api-key",
    help="Project API key. If not provided, LMNR_PROJECT_API_KEY from os.environ or .env is used",
)
@click.option(
    "-s",
    "--dev-session-id",
    help="Dev session ID. If not provided, LMNR_DEV_SESSION_ID from os.environ or .env is used",
)
def dev(project_api_key, dev_session_id):
    cur_dir = os.getcwd()  # e.g. /Users/username/project_name
    env_path = find_dotenv(usecwd=True)
    project_api_key = project_api_key or os.environ.get("LMNR_PROJECT_API_KEY")
    if not project_api_key:
        project_api_key = get_key(env_path, "LMNR_PROJECT_API_KEY")
    if not project_api_key:
        raise ValueError("LMNR_PROJECT_API_KEY is not set")
    
    session_id = dev_session_id or os.environ.get("LMNR_DEV_SESSION_ID")
    if not session_id:
        session_id = get_key(env_path, "LMNR_DEV_SESSION_ID")
    if not session_id:
        raise ValueError("LMNR_DEV_SESSION_ID is not set")
    functions = _load_functions(cur_dir)
    
    observer = Observer()
    handler = SimpleEventHandler(project_api_key, session_id, functions)
    observer.schedule(handler, cur_dir, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        handler.debugger.stop()
        observer.stop()
    observer.join()
