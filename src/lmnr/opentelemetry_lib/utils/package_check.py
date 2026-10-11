from importlib.metadata import Distribution, distributions


def _dist_name(dist: Distribution) -> str:
    # `PackageMetadata.get` is only declared on Python >= 3.12; `__contains__` and
    # `__getitem__` are available on every supported version.
    if dist.name:
        return dist.name.lower()
    if "Name" in dist.metadata:
        return (dist.metadata["Name"] or "").lower()
    return ""


installed_packages = {_dist_name(dist) for dist in distributions()}


def is_package_installed(package_name: str) -> bool:
    return package_name.lower() in installed_packages


def get_package_version(package_name: str) -> str | None:
    for dist in distributions():
        if _dist_name(dist) == package_name.lower():
            return dist.version
    return None
