from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

import attrs
from ruamel.yaml import YAML

from ..attrs import define
from ..config import settings
from ..typing import PathLike


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
    "komodo": lambda x: x.startswith("spectra/absorption/mono/komodo/"),
    "gecko": lambda x: x.startswith("spectra/absorption/mono/gecko/"),
    "monotropa": lambda x: x.startswith("spectra/absorption/ckd/monotropa/"),
    "mycena": lambda x: x.startswith("spectra/absorption/ckd/mycena_v2/"),
    "mycena_v1": lambda x: x.startswith("spectra/absorption/ckd/mycena/"),
    "mycena_v2": lambda x: x.startswith("spectra/absorption/ckd/mycena_v2/"),
    "panellus": lambda x: x.startswith("spectra/absorption/ckd/panellus/"),
}.items():
    FILE_LIST_LOADERS[name] = FileListLoader(
        FILE_LIST_LOADERS["all"].file_list, filter=rule
    )


def known_file_lists() -> list[str]:
    return list(FILE_LIST_LOADERS.keys())


def get_file_list(spec: str):
    """
    Get a list of files corresponding to a specifier (i.e. an identifier).
    """
    try:
        return FILE_LIST_LOADERS[spec].get_file_list()
    except KeyError as e:
        raise ValueError(f"unknown file list specification '{spec}'") from e


def _validator_dir_exists(instance, attribute, value):
    if not value.is_dir():
        raise NotADirectoryError(value)


@define
class FileResolver:
    """
    This class resolves paths relative to a list of locations on disk.
    Locations are looked up in order upon calling the :meth:`.resolve` method.
    If a lookup is successful, the resolved absolute path is returned;
    otherwise, the input path is returned unchanged.

    Parameters
    ----------
    paths : list of path-likes, optional
        A list of search directories. Each entry must exist.
    """

    paths: list[Path] = attrs.field(
        factory=list,
        converter=lambda value: [Path(x).resolve() for x in value],
        validator=attrs.validators.deep_iterable(_validator_dir_exists),
    )

    def append(self, path: PathLike, avoid_duplicates: bool = True) -> None:
        """
        Append an entry to the end of the list of search paths.

        Parameters
        ----------
        path : path-like
            Path to append. The location must exist on disk.

        avoid_duplicates : bool, optional
            If ``True``, do not append again a path that is already registered.
        """
        path = Path(path).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"{path}")

        if avoid_duplicates and path not in self.paths:
            self.paths.append(path)

    def clear(self) -> None:
        """
        Clear the list of search paths.
        """
        self.paths.clear()

    def prepend(self, path: PathLike, avoid_duplicates: bool = True) -> None:
        """
        Prepend an entry at the beginning of the list of search paths.

        Parameters
        ----------
        path : path-like
            Path to prepend. The location must exist on disk.

        avoid_duplicates : bool, optional
            If ``True``, do not prepend again a path that is already registered.

        """
        path = Path(path).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"{path}")

        if avoid_duplicates and path not in self.paths:
            self.paths.insert(0, path)

    def resolve(self, path: PathLike, strict: bool = False, cwd: bool = False) -> Path:
        """
        Resolve a path: search all registered locations in order. If no file is
        found in any of the registered location, ``path`` is returned unchanged.

        Parameters
        ----------
        path : path-like
            Path to be resolved.

        strict : bool, default: False
            If ``True``, resolution failure will raise.

        cwd : bool, default: False
            If ``True``, check first if a file relative to the current working
            directory exists.

        Returns
        -------
        Resolved path

        Raises
        ------
        FileNotFoundError
            If ``strict`` is ``True`` and ``path`` was not found in one of the
            registered directories.
        """
        path = Path(path)

        if not path.is_absolute():
            for base in self.paths if not cwd else [Path.cwd()] + self.paths:
                combined = base / path
                if combined.exists():
                    return combined

        if strict:
            raise FileNotFoundError(path)

        return path


fresolver = FileResolver(settings["path"])
"""Unique file resolver instance."""
