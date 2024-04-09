from __future__ import annotations

from pathlib import Path

from importlib_resources import files
from ruamel.yaml import YAML


def get_file_list(spec: str):
    yaml = YAML()

    def load_file_list(path):
        return yaml.load(path)

    FILTER_RULES = {
        "komodo": lambda x: x.startswith("spectra/absorption/mono/komodo"),
        "gecko": lambda x: x.startswith("spectra/absorption/mono/gecko"),
        "monotropa": lambda x: x.startswith("spectra/absorption/ckd/monotropa"),
        "mycena": lambda x: x.startswith("spectra/absorption/ckd/mycena"),
        "panellus": lambda x: x.startswith("spectra/absorption/ckd/panellus"),
    }

    if spec == "minimal":
        return load_file_list(Path(files("eradiate") / "data/downloads_minimal.yml"))

    file_list_all = load_file_list(Path(files("eradiate") / "data/downloads_all.yml"))

    if spec == "all":
        return file_list_all

    if spec in FILTER_RULES:
        rule = FILTER_RULES[spec]
        return [path for path in file_list_all if rule(path)]

    raise ValueError(f"unknown file list specification '{spec}'")
