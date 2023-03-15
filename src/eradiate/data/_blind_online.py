from __future__ import annotations

import os
import shutil
import typing as t
from pathlib import Path

import attrs
import pooch
from requests import RequestException

from ._core import DataStore, expand_rules
from ..attrs import documented, parse_docs
from ..exceptions import DataError
from ..typing import PathLike
from ..util.misc import LoggingContext


@parse_docs
@attrs.define
class BlindOnlineDataStore(DataStore):
    """
    Serve data downloaded from a remote source without integrity check.
    """

    _base_url: str = documented(
        attrs.field(converter=lambda x: x + "/" if not x.endswith("/") else x),
        type="str",
        doc="URL to the online storage location.",
    )

    path: Path = documented(
        attrs.field(converter=lambda x: Path(x).absolute()),
        type="Path",
        init_type="path-like",
        doc="Path to the local cache location.",
    )

    @property
    def base_url(self) -> str:
        # Inherit docstring
        return self._base_url

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
        downloader: t.Callable | None = None,
    ) -> Path:
        """
        Fetch a file from the data store. This method wraps
        :func:`pooch.retrieve` and automatically selects compressed files
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
        For instance, if ``"foo.nc"`` is requested and ``"foo.nc.gz"`` is
        registered, the latter will be downloaded, decompressed and served as
        ``"foo.nc"``.
        """

        fname = str(filename)

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

    def purge(self, keep: None | str | list[str] = None) -> None:
        """
        Purge local storage location. The default behaviour is very aggressive
        and will wipe out the entire directory contents.

        Parameters
        ----------
        keep : str or list of str, optional
            A list of exclusion rules (paths relative to the store's local
            storage root, shell wildcards allowed) defining files which should
            be excluded from the purge process.

        Warnings
        --------
        This is a destructive operation, make sure you know what you're doing!
        """
        if keep is not None and not isinstance(keep, (list, tuple)):
            keep = list(keep)

        # Fast track for simple case
        if not keep:
            for filename in os.scandir(self.path):
                file_path = os.path.join(self.path, filename)

                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            return

        # List files to keep and delete
        included = expand_rules(rules=["**/*"], prefix=self.path)
        excluded = expand_rules(rules=keep, prefix=self.path)
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
