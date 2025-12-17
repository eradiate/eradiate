import glob
import importlib
from pathlib import Path

from ruamel.yaml import YAML
from xarray_validate import DatasetSchema

SCHEMA_REGISTRY = {}


def _load_schemas() -> None:
    schema_paths = sorted(
        Path(x)
        for x in glob.glob(
            str(importlib.resources.files("eradiate.data.schemas").joinpath("*.yml"))
        )
    )

    yaml = YAML(typ="safe")
    for path in schema_paths:
        name = path.stem
        with open(path, "r") as f:
            schema = yaml.load(f)
        SCHEMA_REGISTRY[name] = DatasetSchema.deserialize(schema)


_load_schemas()
