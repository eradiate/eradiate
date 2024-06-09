import importlib.resources

from ruamel.yaml import YAML

_yaml = YAML()

with importlib.resources.open_text(
    "eradiate.data.datasets", "molecular_absorption.yml"
) as f:
    MOLECULAR_ABSORPTION = _yaml.load(f)
