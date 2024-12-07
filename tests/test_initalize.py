import dotenv
import pytest
import os

from lmnr import Laminar


def test_initialize():
    Laminar.initialize(
        project_api_key="test_key",
    )
    assert Laminar.is_initialized()


def test_initialize_rejects_no_project_api_key():
    if dotenv.find_dotenv(usecwd=True):
        # Initialize internally also looks at `.env`, but we don't want to
        # alter the file just for this test, so we skip the test if `.env`
        # exists.
        return
    with pytest.raises(ValueError):
        old_project_api_key = os.environ.get("LMNR_PROJECT_API_KEY")
        os.environ["LMNR_PROJECT_API_KEY"] = ""
        Laminar.initialize()

    if old_project_api_key:
        os.environ["LMNR_PROJECT_API_KEY"] = old_project_api_key


def test_initialize_rejects_port_in_base_url():
    with pytest.raises(ValueError):
        Laminar.initialize(base_url="http://localhost:443")
