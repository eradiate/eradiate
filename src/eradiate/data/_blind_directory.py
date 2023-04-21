from __future__ import annotations

import typing as t
from pathlib import Path

import attrs

from ._core import DataStore
from ..attrs import documented, parse_docs
from ..exceptions import DataError
from ..typing import PathLike


@parse_docs
@attrs.define
class BlindDirectoryDataStore(DataStore):
    """
    Serve files stored in a directory.
    """

    path: Path = documented(
        attrs.field(converter=lambda x: Path(x).absolute()),
        type="Path",
        init_type="path-like",
        doc="Path to the root of the directory referenced by this data store.",
    )

    @property
    def base_url(self) -> str:
        """
        Raises :class:`NotImplementedError` (this data store has no target
        location).
        """
        raise NotImplementedError

    @property
    def registry(self) -> dict:
        """
        Raises :class:`NotImplementedError` (this data store has no registry).
        """
        raise NotImplementedError

    def registry_files(
        self, filter: t.Callable[[t.Any], bool] | None = None
    ) -> list[str]:
        """
        Returns an empty list (this data store has no registry).
        """
        return []

    def fetch(
        self,
        filename: PathLike,
        **kwargs,
    ) -> Path:
        # No kwargs are actually accepted
        if kwargs:
            keyword = next(iter(kwargs.keys()))
            raise TypeError(f"fetch() got an unexpected keyword argument '{keyword}'")

        fname = self.path / filename
        if fname.is_file():
            return fname
        else:
            raise DataError(f"file '{str(filename)}' is not in '{str(self.path)}'")
