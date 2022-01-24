import logging
import os
import shutil
import typing as t
from pathlib import Path

import attr
import pooch
from requests import RequestException

from ._core import DataStore, registry_from_file
from .._util import LoggingContext
from ..exceptions import DataError
from ..typing import PathLike

logger = logging.getLogger(__name__)


@attr.s(repr=False, init=False)
class OnlineDataStore(DataStore):
    """
    A wrapper around :class:`pooch.Pooch`.
    """

    manager: pooch.Pooch = attr.ib()
    registry_fname: Path = attr.ib(converter=Path)

    def __init__(
        self, base_url: str, path: PathLike, registry_fname: PathLike = "registry.txt"
    ):
        # Initialise attributes
        if not base_url.endswith("/"):
            base_url += "/"
        path = Path(path).absolute()

        manager = pooch.create(
            base_url=base_url,
            path=path,
            registry=None,  # We'll load it later
        )
        self.__attrs_init__(manager=manager, registry_fname=registry_fname)

        # Initialise register load the registry
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

        return f"OnlineDataStore({', '.join(attr_reprs)})"

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
    def registry(self) -> t.Dict[str, str]:
        """
        dict : Registry contents.
        """
        return self.manager.registry

    def registry_files(
        self, filter: t.Optional[t.Callable[[t.Any], bool]] = None
    ) -> t.List[str]:
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
        Path: Path to the registry file.
        """
        return self.path / self.registry_fname

    def registry_fetch(self) -> Path:
        """
        Get the absolute path to the registry file.
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
        Reload the registry file.
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
        path
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
        downloader: t.Optional[t.Callable] = None,
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
            local file name.

        Returns
        -------
        str
            Absolute path where the retrieved resource is located.

        Notes
        -----
        If a compressed resource exists, it will be served automatically.
        For instance, if ``foo.nc`` is requested and ``foo.nc.gz`` is
        registered, the latter will be downloaded, decompressed and served as
        ``foo.nc``.
        """
        # By default, just forward arguments
        fname: str = str(filename)
        processor = None

        # Look up the file in the registry
        # (also detects if a gzip-compressed resource is available)
        try:
            fname = str(self.is_registered(filename))

        # If file is unregistered, serve it blindly
        except ValueError:
            with LoggingContext(
                pooch.get_logger(), level="WARNING"
            ):  # Silence pooch messages temporarily
                # Try first to get a compressed file
                try:
                    return Path(
                        pooch.retrieve(
                            os.path.join(self.base_url, fname + ".gz"),
                            known_hash=None,
                            fname=fname + ".gz",
                            path=self.path,
                            processor=pooch.processors.Decompress(
                                name=os.path.basename(fname)
                            ),
                        )
                    )
                except RequestException:
                    pass

                    # If no gzip-compressed file is available, try the actual file
                    try:
                        print(os.path.join(self.base_url, fname))
                        return Path(
                            pooch.retrieve(
                                os.path.join(self.base_url, fname),
                                known_hash=None,
                                fname=fname,
                                path=self.path,
                            ),
                        )
                    except RequestException as e:
                        raise DataError(
                            f"file '{fname}' could not be retrieved from {self.base_url}"
                        ) from e

        # If file is registered, serve it with integrity check
        else:
            # If the matched registered resource is a compressed file, serve it
            root, ext = os.path.splitext(fname)
            if ext == ".gz":
                processor = pooch.processors.Decompress(name=os.path.basename(root))

            return Path(
                self.manager.fetch(fname, processor=processor, downloader=downloader)
            )

        # This is not supposed to happen
        raise DataError(f"file '{fname}' could not be retrieved from {self.base_url}")

    def purge(self, keep_registered: bool = False) -> None:
        """
        Purge local storage location.

        Warnings
        --------
        This is a destructive operation, make sure you know what you're doing!
        """
        # Fast track for simple case
        if not keep_registered:
            for filename in os.scandir(self.path):
                file_path = os.path.join(self.path, filename)

                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            return

        # More complicated: examine all files one by one
        for x in self.path.rglob("**/*"):
            if x.is_file():
                try:
                    self.is_registered(x.relative_to(self.path))
                except ValueError:
                    os.remove(x)

        # Clean up empty directories
        for x in self.path.iterdir():
            if x.is_dir():
                try:
                    os.removedirs(x)
                except OSError:
                    pass
