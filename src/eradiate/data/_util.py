from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

import attrs
from ruamel.yaml import YAML


@attrs.define
class FileListLoader:
    """
    Lazy loader for file lists.
    """

    file_list: list[str] | Callable[[], bool]
    filter: Callable[[Any], bool] = attrs.field(default=lambda x: True)

    def get_file_list(self) -> list[str]:
        file_list = self.file_list() if callable(self.file_list) else self.file_list
        return [path for path in file_list if self.filter(path)]


_yaml = YAML()

FILE_LIST_LOADERS: dict[str, FileListLoader] = {
    "all": FileListLoader(
        lambda: _yaml.load(Path(files("eradiate") / "data/downloads_all.yml"))
    ),
    "minimal": FileListLoader(
        _yaml.load(Path(files("eradiate") / "data/downloads_minimal.yml"))
    ),
}

for name, rule in {
    "komodo": lambda x: x.startswith("spectra/absorption/mono/komodo"),
    "gecko": lambda x: x.startswith("spectra/absorption/mono/gecko"),
    "monotropa": lambda x: x.startswith("spectra/absorption/ckd/monotropa"),
    "mycena": lambda x: x.startswith("spectra/absorption/ckd/mycena"),
    "panellus": lambda x: x.startswith("spectra/absorption/ckd/panellus"),
}.items():
    FILE_LIST_LOADERS[name] = FileListLoader(
        FILE_LIST_LOADERS["all"].file_list, filter=rule
    )


def known_file_lists() -> list[str]:
    return list(FILE_LIST_LOADERS.keys())


def get_file_list(spec: str):
    try:
        return FILE_LIST_LOADERS[spec].get_file_list()
    except KeyError as e:
        raise ValueError(f"unknown file list specification '{spec}'") from e
