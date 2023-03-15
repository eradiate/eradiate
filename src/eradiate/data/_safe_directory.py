"""
Manage files stored in the ``eradiate-data`` repository.
"""
from __future__ import annotations

import os
import typing as t
from pathlib import Path

import attrs

from ._core import DataStore, load_rules, make_registry, registry_from_file
from ..attrs import documented, parse_docs
from ..exceptions import DataError
from ..typing import PathLike


@parse_docs
@attrs.define
class SafeDirectoryDataStore(DataStore):
    """
    Serve files stored in a directory. This data store will only serve files
    listed in its registry.
    """

    path: Path = documented(
        attrs.field(converter=lambda x: Path(x).absolute()),
        type="Path",
        init_type="path-like",
        doc="Path to the root of the directory referenced by this data store.",
    )

    registry_fname: Path = documented(
        attrs.field(default="registry.txt", converter=Path),
        type="Path",
        init_type="path-like",
        default='"registry.txt"',
        doc="Path to the registry file, relative to `path`.",
    )

    @registry_fname.validator
    def _registry_fname_validator(self, attribute, value: Path):
        if value.is_absolute():
            raise ValueError(
                f"while validating '{attribute.name}': "
                "only paths relative to the store root path are allowed"
            )

    _registry: dict = attrs.field(factory=dict, converter=dict, repr=False, init=False)

    def __attrs_post_init__(self):
        self.registry_reload()

    @property
    def base_url(self) -> str:
        # Inherit docstring
        return str(self.path)

    @property
    def registry(self) -> dict:
        # Inherit docstring
        return self._registry

    def registry_files(
        self, filter: t.Callable[[t.Any], bool] | None = None
    ) -> list[str]:
        # Inherit docstring
        raise NotImplementedError

    @property
    def registry_path(self) -> Path:
        """
        Path: Path to the registry file.
        """
        return self.path / self.registry_fname

    def registry_make(self) -> None:
        """
        Generate a registry file from the contents of the ``self.path``
        directory, according to inclusion and exclusion rules defined in the
        ``self.path / "registry_rules.yml"`` file. The generated registry is
        written to ``self.path / self.registry_fname``.
        """
        # Load include and exclude rules
        rules = load_rules(self.path / "registry_rules.yml")

        # Write registry
        make_registry(
            self.path,
            self.registry_path,
            includes=rules["include"],
            excludes=rules["exclude"],
        )

    def registry_fetch(self) -> Path:
        """
        Get the absolute path to the registry file.
        If no file exists, one will be created based on the rules contained
        defined in ``self.path / "registry_rules.yml"``.
        """
        filename = self.registry_path

        if not filename.is_file():
            self.registry_make()

        return filename

    def registry_delete(self):
        """
        Delete the registry file.
        """
        os.remove(self.path / self.registry_fname)

    def registry_reload(self) -> None:
        """
        Reload the registry file from the hard drive.
        """
        self._registry = registry_from_file(self.registry_fetch())

    def fetch(
        self,
        filename: PathLike,
        **kwargs,
    ) -> Path:
        # No kwargs are actually accepted
        if kwargs:
            keyword = next(iter(kwargs.keys()))
            raise TypeError(f"fetch() got an unexpected keyword argument '{keyword}'")

        fname = str(filename)
        if fname in self.registry:
            return self.path / fname
        else:
            raise DataError(f"file '{fname}' is not in the registry")
