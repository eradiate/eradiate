"""
Manage files stored in the ``eradiate-data`` repository.
"""
import os
import typing as t
from pathlib import Path

import attr

from ._core import DataStore, load_rules, make_registry, registry_from_file
from ..exceptions import DataError
from ..typing import PathLike


@attr.s
class DirectoryDataStore(DataStore):
    """
    This class serves files stored in a directory.
    """

    path: Path = attr.ib(converter=lambda x: Path(x).absolute())
    registry_fname: Path = attr.ib(default="registry.txt", converter=Path)

    @registry_fname.validator
    def _registry_fname_validator(self, attribute, value: Path):
        if value.is_absolute():
            raise ValueError(
                f"while validating '{attribute.name}': "
                "only paths relative to the store root path are allowed"
            )

    _registry: t.Dict = attr.ib(factory=dict, converter=dict, repr=False, init=False)

    def __attrs_post_init__(self):
        self.registry_reload()

    def base_url(self) -> str:
        # Inherit docstring
        return str(self.path)

    def registry(self) -> t.Dict:
        # Inherit docstring
        return self._registry

    def registry_files(
        self, filter: t.Optional[t.Callable[[t.Any], bool]] = None
    ) -> t.List[str]:
        # Inherit docstring
        raise NotImplementedError

    @property
    def registry_path(self) -> Path:
        """
        Path: Path to the registry file.
        """
        return self.path / self.registry_fname

    def registry_make(self) -> None:
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

    def registry_reload(self, delete: bool = False) -> None:
        """
        Reload the registry file.
        """
        self.registry = registry_from_file(self.registry_fetch())

    def is_registered(self, filename: PathLike, allow_compressed: bool = True) -> Path:
        raise NotImplementedError

    def fetch(
        self,
        filename: os.PathLike,
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
