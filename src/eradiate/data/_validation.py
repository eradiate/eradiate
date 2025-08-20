from __future__ import annotations

import glob
import importlib.resources
from pathlib import Path

import cerberus
import xarray as xr
from ruamel.yaml import YAML

from ..units import unit_registry as ureg
from ..units import units_compatible


def _load_rules():
    cerberus.rules_set_registry.extend([("integer", {"type": "integer"})])
    cerberus.rules_set_registry.extend([("string", {"type": "string"})])


def _load_schemas() -> None:
    schema_paths = sorted(
        Path(x)
        for x in glob.glob(
            str(importlib.resources.files("eradiate.data.schemas").joinpath("*.yml"))
        )
    )  # TODO: check if that would work on Windows

    yaml = YAML(typ="safe")
    for path in schema_paths:
        name = path.stem
        with open(path, "r") as f:
            schema = yaml.load(f)
        cerberus.schema_registry.add(name, schema)


_load_rules()
_load_schemas()


class DatasetValidator(cerberus.Validator):
    """
    This class implements xarray dataset validation. In addition to providing
    a simple interface to validate the structure of a dataset against a Cerberus
    schema, it provides validation rules
    """

    def _validate_equal_list(self, constraint, field, value):
        """
        Check if the value is equal to a specific list.

        The rule's arguments are validated against this schema:
        {"type": "list"}
        """
        if list(value) != list(constraint):
            self._error(field, f"Must be equal to {constraint}")

    def _validate_units_compatible(self, constraint, field, value):
        """
        Check if a 'units' attribute field has a value that is compatible with
        specific units. The underlying implementation uses
        :func:`.units_compatible` and will consequently not return a false
        positive if validating angle units against dimensionless quantities.

        The rule's arguments are validated against this schema:
        {"type": "string"}
        """
        try:
            u_value = ureg.Unit(value)
        except Exception:
            self._error(field, f"Cannot convert {repr(value)} to valid units")
            return

        u_constraint = ureg.Unit(constraint)
        if not units_compatible(u_constraint, u_value):
            self._error(field, f"Must be units compatible with {constraint}")

    def validate(self, document, schema=None, update=False, normalize=True):
        # Inherit docstring
        if isinstance(document, xr.Dataset):
            document = document.to_dict(data=False)
        super().validate(document, schema=schema, update=update, normalize=normalize)


def list_schemas() -> list[str]:
    """
    List currently registered Cerberus schemas.
    """
    return sorted(cerberus.schema_registry._storage.keys())
