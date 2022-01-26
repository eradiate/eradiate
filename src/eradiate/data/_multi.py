import typing as t
from collections import OrderedDict
from pathlib import Path

import attr

from ._core import DataStore
from ..attrs import documented, parse_docs
from ..exceptions import DataError
from ..typing import PathLike


@parse_docs
@attr.s
class MultiDataStore(DataStore):
    """
    A lightweight container chaining requests on multiple data stores.

    Calls to the :meth:`.fetch` method are successively redirected to each
    referenced data store. The first successful request is served.
    """

    stores: OrderedDict = documented(
        attr.ib(factory=OrderedDict, converter=OrderedDict),
        type="OrderedDict",
        init_type="mapping",
        default="{}",
        doc="Data stores which will be queried successively.",
    )

    def __getitem__(self, item):
        return self.stores[item]

    @property
    def base_url(self) -> str:
        raise NotImplementedError

    @property
    def registry(self) -> t.Dict:
        raise NotImplementedError

    def registry_files(
        self, filter: t.Optional[t.Callable[[t.Any], bool]] = None
    ) -> t.List[str]:
        raise NotImplementedError

    def fetch(self, filename: PathLike, **kwargs) -> Path:
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
