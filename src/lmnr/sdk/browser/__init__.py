from lmnr.openllmetry_sdk.utils.package_check import is_package_installed
from .playwright_patch import init_playwright_tracing


def init_browser_tracing(http_url: str, project_api_key: str):
    if is_package_installed("playwright"):
        init_playwright_tracing(http_url, project_api_key)
    # Other browsers can be added here
