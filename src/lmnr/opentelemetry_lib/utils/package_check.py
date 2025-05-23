from importlib.metadata import distributions

installed_packages = {
    (dist.name or dist.metadata.get("Name", "")).lower() for dist in distributions()
}


def is_package_installed(package_name: str) -> bool:
    return package_name.lower() in installed_packages
