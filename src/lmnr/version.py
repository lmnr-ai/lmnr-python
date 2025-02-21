import sys
import requests
from packaging import version


SDK_VERSION = "0.4.62"
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


def is_latest_version() -> bool:
    try:
        return version.parse(SDK_VERSION) >= version.parse(get_latest_pypi_version())
    except Exception:
        return True


def get_latest_pypi_version() -> str:
    """
    Get the latest stable version of lmnr package from PyPI.
    Returns the version string or raises an exception if unable to fetch.
    """
    try:
        response = requests.get("https://pypi.org/pypi/lmnr/json")
        response.raise_for_status()

        releases = response.json()["releases"]
        stable_versions = [
            ver
            for ver in releases.keys()
            if not version.parse(ver).is_prerelease
            and not version.parse(ver).is_devrelease
            and not any(release.get("yanked", False) for release in releases[ver])
        ]

        if not stable_versions:
            # do not scare the user, assume they are on
            # latest version
            return SDK_VERSION

        latest_version = max(stable_versions, key=version.parse)
        return latest_version

    except Exception:
        # do not scare the user, assume they are on
        # latest version
        return SDK_VERSION
