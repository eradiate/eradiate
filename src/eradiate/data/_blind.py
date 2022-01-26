import os
import shutil
import typing as t
from pathlib import Path

import attr
import pooch
from requests import RequestException

from ._core import DataStore, expand_rules
from .._util import LoggingContext
from ..exceptions import DataError
from ..typing import PathLike


@attr.s
class BlindDataStore(DataStore):
    _base_url: str = attr.ib(converter=lambda x: x + "/" if not x.endswith("/") else x)
    path: Path = attr.ib(converter=lambda x: Path(x).absolute())

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def registry(self) -> t.Dict:
        raise NotImplementedError

    def registry_files(
        self, filter: t.Optional[t.Callable[[t.Any], bool]] = None
    ) -> t.List[str]:
        raise NotImplementedError

    def fetch(
        self,
        filename: PathLike,
        downloader: t.Optional[t.Callable] = None,
    ) -> Path:
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

    def purge(self, keep: t.Union[None, str, t.List[str]] = None) -> None:
        """
        Purge local storage location. The default behaviour is very aggressive
        and will wipe out the entire directory contents.

        Parameter
        ---------
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
