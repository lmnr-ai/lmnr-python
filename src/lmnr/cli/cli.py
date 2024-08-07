from pathlib import Path
import sys
import requests
from dotenv import load_dotenv
import os
import click
import logging
from cookiecutter.main import cookiecutter
from pydantic.alias_generators import to_pascal

from lmnr.cli.zip import zip_directory
from lmnr.sdk.registry import Registry as Pipeline
from lmnr.sdk.remote_debugger import RemoteDebugger

from .parser.parser import runnable_graph_to_template_vars

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
        load_dotenv()
        project_api_key = os.environ.get("LMNR_PROJECT_API_KEY")
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
        load_dotenv()
        project_api_key = os.environ.get("LMNR_PROJECT_API_KEY")
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


@cli.command(name="dev")
@click.option(
    "-p",
    "--project-api-key",
    help="Project API key",
)
def dev(project_api_key):
    project_api_key = project_api_key or os.environ.get("LMNR_PROJECT_API_KEY")
    if not project_api_key:
        load_dotenv()
        project_api_key = os.environ.get("LMNR_PROJECT_API_KEY")
    if not project_api_key:
        raise ValueError("LMNR_PROJECT_API_KEY is not set")

    cur_dir = os.getcwd()  # e.g. /Users/username/project_name
    parent_dir, name = os.path.split(cur_dir)  # e.g. /Users/username, project_name

    # Needed to __import__ pipeline.py
    if sys.path[0] != parent_dir:
        sys.path.insert(0, parent_dir)
    # Needed to import src in pipeline.py and other files
    if cur_dir not in sys.path:
        sys.path.insert(0, cur_dir)

    module_name = f"{name}.pipeline"
    __import__(module_name)
    module = sys.modules[module_name]

    matches = [v for v in module.__dict__.values() if isinstance(v, Pipeline)]
    if not matches:
        raise ValueError("No Pipeline found in the module")
    if len(matches) > 1:
        raise ValueError("Multiple Pipelines found in the module")
    pipeline = matches[0]

    tools = pipeline.functions
    debugger = RemoteDebugger(project_api_key, tools)
    debugger.start()
