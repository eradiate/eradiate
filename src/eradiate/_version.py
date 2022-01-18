from importlib.metadata import PackageNotFoundError, version

try:
    _version = version("eradiate")
except PackageNotFoundError as e:
    raise PackageNotFoundError(
        "Eradiate is not installed; please install it in your Python environment."
    ) from e
