from __future__ import annotations

import typing as t
from collections import OrderedDict
from pathlib import Path

import attrs

from ._core import DataStore
from ..attrs import documented, parse_docs
from ..exceptions import DataError
from ..typing import PathLike


@parse_docs
@attrs.define
class MultiDataStore(DataStore):
    """
    Chain requests on multiple data stores.

    Calls to the :meth:`~.MultiDataStore.fetch` method are successively redirected
    to each referenced data store. The first successful request is served.
    """

    stores: OrderedDict = documented(
        attrs.field(factory=OrderedDict, converter=OrderedDict),
        type="collections.OrderedDict",
        init_type="mapping",
        default="{}",
        doc="Data stores which will be queried successively.",
    )

    def __getitem__(self, item):
        return self.stores[item]

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

    def fetch(self, filename: PathLike, **kwargs) -> Path:
        # Inherit docstring
        # No kwargs are actually accepted
        if kwargs:
            keyword = next(iter(kwargs.keys()))
            raise TypeError(f"fetch() got an unexpected keyword argument '{keyword}'")

        # Try and serve data
        for _, store in self.stores.items():
            try:
                return store.fetch(filename)
            except DataError:
                continue

        raise DataError(f"file '{filename}' could not be served")
