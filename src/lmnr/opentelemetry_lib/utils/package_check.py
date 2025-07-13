from importlib.metadata import distributions

from typing import Optional

installed_packages = {
    (dist.name or dist.metadata.get("Name", "")).lower() for dist in distributions()
}


def is_package_installed(package_name: str) -> bool:
    return package_name.lower() in installed_packages


def get_package_version(package_name: str) -> Optional[str]:
    for dist in distributions():
        if (dist.name or dist.metadata.get("Name", "")).lower() == package_name.lower():
            return dist.version
    return None
