import requests
from dotenv import load_dotenv
import os
import click
import logging
from cookiecutter.main import cookiecutter
from importlib import resources as importlib_resources
from pydantic.alias_generators import to_pascal

from .parser.parser import runnable_graph_to_template_vars
import lmnr

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
        str(importlib_resources.files(lmnr)),
        output_dir=".",
        config_file=None,
        extra_context=context,
        directory="cli/",
        no_input=True,
        overwrite_if_exists=True,
    )
