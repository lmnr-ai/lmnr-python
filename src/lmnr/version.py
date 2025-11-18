import sys
import httpx
from packaging import version


__version__ = "0.7.23"
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


def is_latest_version() -> bool:
    try:
        return version.parse(__version__) >= version.parse(get_latest_pypi_version())
    except Exception:
        return True


def get_latest_pypi_version() -> str:
    """
    Get the latest stable version of lmnr package from PyPI.
    Returns the version string or raises an exception if unable to fetch.
    """
    try:
        response = httpx.get("https://pypi.org/pypi/lmnr/json")
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
            return __version__

        latest_version = max(stable_versions, key=version.parse)
        return latest_version

    except Exception:
        # do not scare the user, assume they are on
        # latest version
        return __version__
