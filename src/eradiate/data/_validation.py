from __future__ import annotations

import glob
import importlib.resources
from pathlib import Path

import cerberus
import xarray as xr
from ruamel.yaml import YAML

from eradiate.units import unit_registry as ureg
from eradiate.units import units_compatible


def _load_rules():
    cerberus.rules_set_registry.extend([("integer", {"type": "integer"})])
    cerberus.rules_set_registry.extend([("string", {"type": "string"})])


def _load_schemas():
    schema_paths = sorted(
        Path(x)
        for x in glob.glob(
            str(importlib.resources.files("eradiate.data.schemas").joinpath("*.yml"))
        )
    )  # TODO: check if that would work on Windows

    yaml = YAML(typ="safe")
    for path in schema_paths:
        name = path.stem
        schema = yaml.load(open(path))
        cerberus.schema_registry.add(name, schema)


_load_rules()
_load_schemas()


class DatasetValidator(cerberus.Validator):
    def _validate_equal_list(self, constraint, field, value):
        """
        {"type": "list"}
        """
        if list(value) != list(constraint):
            self._error(field, f"Must be equal to {constraint}")

    def _validate_units_compatible(self, constraint, field, value):
        """
        {"type": "string"}
        """
        u_constraint = ureg(constraint)
        u_value = ureg(value)
        if not units_compatible(u_constraint, u_value):
            self._error(field, f"Must be units compatible with {constraint}")

    def validate(self, document, schema=None, update=False, normalize=True):
        if isinstance(document, xr.Dataset):
            document = document.to_dict(data=False)
        super().validate(document, schema=schema, update=update, normalize=normalize)


def list_schemas():
    return sorted(cerberus.schema_registry._storage.keys())
