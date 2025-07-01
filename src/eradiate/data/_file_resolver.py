from __future__ import annotations

from pathlib import Path

import attrs
import xarray as xr

from ._asset_manager import asset_manager
from ..attrs import define
from ..config import SOURCE_DIR, settings
from ..typing import PathLike


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

    def load_dataset(
        self, path: PathLike, strict: bool = False, cwd: bool = False
    ) -> xr.Dataset:
        """
        Chain :meth:`resolve` and :func:`xarray.load_dataset`.
        """
        fname = self.resolve(path, strict=strict, cwd=cwd)
        return xr.load_dataset(fname)


#: Unique file resolver instance
fresolver = FileResolver(settings["path"])
fresolver.append(asset_manager.install_dir)
if SOURCE_DIR:
    fresolver.append(SOURCE_DIR / "resources/data")
