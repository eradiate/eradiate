from __future__ import annotations

import logging
import os
import shutil
import typing as t
from pathlib import Path

import attrs
import pooch

from ._core import DataStore, expand_rules, registry_from_file
from ..exceptions import DataError
from ..typing import PathLike
from ..util.misc import LoggingContext

logger = logging.getLogger(__name__)


@attrs.define(repr=False, init=False)
class SafeOnlineDataStore(DataStore):
    """
    Serve files located online, with integrity check.

    Parameters
    ----------
    base_url : str
        URL to the online storage location.

    path : path-like
        Path to the local cache location.

    registry_fname : path-like, optional
        Path to the registry file, relative to `path`.

    Fields
    ------
    manager : pooch.Pooch
        The Pooch instance used to manage downloaded content.

    registry_fname : Path
        Path to the registry file, relative to `path`.

    Notes
    -----
    This class basically wraps a :class:`pooch.Pooch` instance.
    """

    manager: pooch.Pooch = attrs.field()
    registry_fname: Path = attrs.field(converter=Path)

    def __init__(
        self, base_url: str, path: PathLike, registry_fname: PathLike = "registry.txt"
    ):
        # Initialize attributes
        if not base_url.endswith("/"):
            base_url += "/"
        path = Path(path).absolute()

        manager = pooch.create(
            base_url=base_url,
            path=path,
            registry=None,  # We'll load it later
        )
        self.__attrs_init__(manager=manager, registry_fname=registry_fname)

        # Initialize register load the registry
        registry = registry_from_file(self.registry_fetch())
        manager.registry = registry

    def __repr__(self):
        attr_reprs = [
            f"{x}={self.__getattribute__(x).__repr__()}"
            for x in [
                "base_url",
                "path",
            ]
        ]

        return f"SafeOnlineDataStore({', '.join(attr_reprs)})"

    @property
    def base_url(self) -> str:
        # Inherit docstring
        return self.manager.base_url

    @property
    def path(self) -> Path:
        """
        path : Absolute path to the local data storage folder.
        """
        return Path(self.manager.path)

    @property
    def registry(self) -> dict[str, str]:
        # Inherit docstring
        return self.manager.registry

    def registry_files(
        self, filter: t.Callable[[t.Any], bool] | None = None
    ) -> list[str]:
        """
        Get a list of registered files.

        Parameters
        ----------
        filter : callable, optional
            A filter function taking a file path as a single string argument and
            returning a Boolean. Filenames for which the filter returns ``True``
            will be returned.

        Returns
        -------
        files : list of str
            List of registered files.
        """
        if filter is None:
            return self.manager.registry_files
        else:
            return [x for x in self.manager.registry_files if filter(x)]

    @property
    def registry_path(self) -> Path:
        """
        Path: Absolute path to the registry file.
        """
        return self.path / self.registry_fname

    def registry_fetch(self) -> Path:
        """
        Get the absolute path to the registry file and make sure that it is
        written to the local cache.
        """
        filename = self.registry_path

        with LoggingContext(
            pooch.get_logger(), level="WARNING"
        ):  # Silence pooch messages temporarily
            result = pooch.retrieve(
                os.path.join(self.base_url, self.registry_fname),
                known_hash=None,
                fname=str(filename),
                path=self.path,
            )

        return Path(result)

    def registry_delete(self):
        """
        Delete the registry file.
        """
        os.remove(self.path / self.registry_fname)

    def registry_reload(self, delete: bool = False) -> None:
        """
        Reload the registry file from the local cache.

        Parameters
        ----------
        delete : bool, optional
            If ``True``, the existing registry file will be deleted and
            downloaded again.
        """
        if delete:
            self.registry_delete()

        registry_fname = self.registry_fetch()
        self.manager.registry = registry_from_file(registry_fname)

    def is_registered(self, filename: PathLike, allow_compressed: bool = True) -> Path:
        """
        Check if a file is registered, with an option to look for compressed
        data.

        Parameters
        ----------
        filename : path-like
            File name to fetch from the local storage, relative to the storage
            root.

        allow_compressed : bool, optional
            If ``True``, a query for ``foo.bar`` will result in a query for the
            gzip-compressed file name ``foo.bar.gz``. The compressed file takes
            precedence.

        Returns
        -------
        path : Path
            The file name which matched `filename`.

        Raises
        ------
        ValueError
            If `filename` could not be matched with any entry in the registry.
        """
        fname = str(filename)

        if allow_compressed and not fname.endswith(".gz"):
            fname_compressed = fname + ".gz"

            if fname_compressed in self.manager.registry:
                return Path(fname_compressed)

        if fname in self.manager.registry:
            return Path(fname)

        raise ValueError(f"File '{fname}' is not in the registry.")

    def fetch(
        self,
        filename: PathLike,
        downloader: t.Callable | None = None,
    ) -> Path:
        """
        Fetch a file from the data store. This method wraps
        :meth:`pooch.Pooch.fetch` and automatically selects compressed files
        when they are available.

        Parameters
        ----------
        filename : path-like
            File name to fetch from the local storage, relative to the storage
            root.

        downloader : callable, optional
            A callable that will be called to download a given URL to a provided
            local file name. This is mostly useful to
            `display progress bars <https://www.fatiando.org/pooch/latest/progressbars.html>`_
            during download.

        Returns
        -------
        path : Path
            Absolute path where the retrieved resource is located.

        Notes
        -----
        If a compressed resource exists, it will be served automatically.
        For instance, if ``"foo.nc"`` is requested and ``foo.nc.gz`` is
        registered, the latter will be downloaded, decompressed and served as
        ``"foo.nc"``.
        """
        # By default, just forward arguments
        fname: str = str(filename)
        processor = None

        # Look up the file in the registry
        # (also detects if a gzip-compressed resource is available)
        try:
            fname = str(self.is_registered(filename))

        except ValueError as e:
            raise DataError(
                f"file '{fname}' could not be retrieved from {self.base_url}"
            ) from e

        # If the matched registered resource is a compressed file, serve it
        root, ext = os.path.splitext(fname)
        if ext == ".gz":
            processor = pooch.processors.Decompress(name=os.path.basename(root))

        return Path(
            self.manager.fetch(fname, processor=processor, downloader=downloader)
        )

    def purge(self, keep: None | str | list[str] = None) -> None:
        """
        Purge local storage location. The default behaviour is very aggressive
        and will wipe out the entire directory contents.

        Parameters
        ----------
        keep : "registered" or list of str, optional
            If set to ``"registered"``, files in the registry, as well as the
            registry file itself, will not be deleted. Finer control is possible
            by passing a list of exclusion rules (paths relative to the store's
            local storage root, shell wildcards allowed).

        Notes
        -----
        Passing ``keep="registered"`` keeps registered files to minimize the
        amount of data to be downloaded upon future queries the the data store.
        This means, for instance, that if data is registered and downloaded as
        a compressed file, then served decompressed, the compressed file will be
        kept, while the decompressed file will be deleted.

        Warnings
        --------
        This is a destructive operation, make sure you know what you're doing!
        """
        # Fast track for simple case
        if keep is None:
            for filename in os.scandir(self.path):
                file_path = os.path.join(self.path, filename)

                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            return

        # List files to keep and delete
        if keep == "registered":
            excluded = expand_rules(
                rules=self.registry_files() + [str(self.registry_fname)],
                prefix=self.path,
            )
        else:
            excluded = expand_rules(rules=keep, prefix=self.path)
        included = expand_rules(rules=["**/*"], prefix=self.path)
        remove = sorted(included - excluded)

        for file in remove:
            os.remove(file)

        # Clean up empty directories
        for x in self.path.iterdir():
            if x.is_dir():
                try:
                    os.removedirs(x)
                except OSError:
                    pass
