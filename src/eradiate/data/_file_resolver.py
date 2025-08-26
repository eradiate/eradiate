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

    Notes
    -----
    * By default, the unique file resolver instance
      :data:`eradiate.fresolver <.fresolver>` is initialized with the
      installation path of the asset manager (see :class:`.AssetManager`) and,
      when in dev mode (see :data:`SOURCE_DIR <eradiate.config.SOURCE_DIR>`),
      the path to the location of test files.
    * The current working directory is not appended to the file resolver by
      default. It can however optionally be added temporarily when calling
      :meth:`resolve` (and wrappers such as :meth:`load_dataset`). This is
      useful when using the ``strict`` mode, which raises if the requested path
      cannot be resolved to an existing file.


    Examples
    --------
    A single instance of the file resolver is available as
    :data:`eradiate.fresolver <.fresolver>`:

    >>> from eradiate import fresolver

    To add a path to the file resolver, use the :meth:`~.FileResolver.append`
    or :meth:`~.FileResolver.prepend` methods, *e.g.*:

    >>> fresolver.append("some/path/on/the/drive")

    To resolve a relative path, use the :meth:`~.FileResolver.resolve` method:

    >>> fresolver.resolve("srf/sentinel_2a-msi-3.nc")

    If the path is expected to point to an existing dataset file, the
    :meth:`~.FileResolver.load_dataset` method can be used to load it immediately:

    >>> fresolver.load_dataset("srf/sentinel_2a-msi-3.nc")
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

        See Also
        --------
        :meth:`.prepend`
        """
        path = Path(path).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"{path}")

        if avoid_duplicates and path in self.paths:
            return

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

        See Also
        --------
        :meth:`.append`
        """
        path = Path(path).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"{path}")

        if avoid_duplicates and path in self.paths:
            return

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
        Path
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
        Dataset
        """
        fname = self.resolve(path, strict=strict, cwd=cwd)
        return xr.load_dataset(fname)

    def info(self, show: bool = False) -> dict | None:
        """
        Collect information about the file resolver.

        Parameters
        ----------
        show : bool
            If ``True``, display information to the terminal. Otherwise, return
            it as a dictionary.

        Returns
        -------
        dict or None
        """
        if show:
            title = "File resolver"
            print(title)
            print("-" * len(title))

            for path in self.paths:
                print(f"• {path}")

            return None

        else:
            return {"paths": self.paths}


#: Unique file resolver instance (exposed as :data:`eradiate.fresolver`)
fresolver = FileResolver(settings["path"])
fresolver.append(asset_manager.install_dir)
if SOURCE_DIR:
    fresolver.append(SOURCE_DIR / "resources/data")
