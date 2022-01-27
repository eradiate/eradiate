"""A simple file resolver."""

from pathlib import Path

from ._config import config
from ._util import Singleton


class PathResolver(metaclass=Singleton):
    """
    This class implements a simple cross-platform path resolver.
    It looks for a file or directory given its (possibly relative) name and a
    set of search paths. The implementation walks through the search paths in
    order and stops once the file is found.

    Fields
    ------
    paths : list of :class:`~pathlib.Path`
        Stored path list.
    """

    def __init__(self):
        """
        Initialize a new file resolver with the current working directory
        and Eradiate's data directory.
        """
        self.paths = []
        self.reset()

    def __len__(self):
        return self.paths.__len__()

    def __getitem__(self, item):
        return self.paths.__getitem__(item)

    def __setitem__(self, key, value):
        value = Path(value).absolute()
        return self.paths.__setitem__(key, value)

    def __repr__(self):
        s = ", ".join([str(x) for x in self.paths])
        return f"PathResolver(paths=[{s}])"

    def __str__(self):
        if not self.paths:
            return "PathResolver(paths=[])"
        else:
            s = ",\n".join(["  " + str(x) for x in self.paths])
            return f"PathResolver(paths=[\n{s}\n])"

    def reset(self):
        """
        Reset path list to default value:
        ``[$PWD, $ERADIATE_DATA_PATH, $ERADIATE_DIR/resources/data]``.

        See Also
        --------
        :ref:`List of environment variables <sec-config-env_vars>`
        """

        self.clear()
        self.append(Path.cwd())  # Current working directory

        if config.data_path:
            self.append(*config.data_path)  # Path list

        self.append(config.dir / "resources" / "data")  # Eradiate data directory

    def clear(self):
        """Clear the list of search paths."""
        self.paths.clear()

    def contains(self, path):
        """
        Check if a given path is included in the search path list.

        Parameters
        ----------
        path : path-like
            Path to be searched for.

        Returns
        -------
        bool
            True if the path is found.
        """
        path = Path(path).absolute()
        return path in self.paths

    def remove(self, item):
        """
        Erase the entry at the given index.

        Parameters
        ----------
        item : path-like or int
            If path-like, path to erase from the path list. If index, index in
            the path list where the item to be removed is located.
        """
        if isinstance(item, int):
            index = item
        else:
            path = Path(item).absolute()
            try:
                index = self.paths.index(path)
            except ValueError:
                raise ValueError(f"could not find {path} in path list")

        self.paths.pop(index)

    def prepend(self, path):
        """
        Prepend an entry at the beginning of the list of search paths.

        Parameters
        ----------
        path : path-like
            Path to prepend to the path list.
        """
        path_absolute = Path(path).absolute()
        if not path_absolute.is_dir():
            raise ValueError(f"{path} is not an existing directory")

        self.paths.insert(0, path_absolute)

    def append(self, *path):
        """
        Append an entry to the end of the list of search paths.

        Parameters
        ----------
        *path : path-like
            Path to append to the path list.
        """
        for _path in path:
            path_absolute = Path(_path).absolute()
            if not path_absolute.is_dir():
                raise ValueError(f"{_path} is not an existing directory")

            self.paths.append(path_absolute)

    def resolve(self, path):
        """
        Walk through the list of search paths and try to resolve the input path.

        Parameters
        ----------
        path : path-like
            Path to try and resolve.

        Returns
        -------
        :class:`pathlib.Path`:
            If path is found, absolute path to the found item.
            Otherwise, unchanged path.
        """
        path = Path(path)

        if not path.is_absolute():
            for path_base in self.paths:
                path_full = path_base / path
                if path_full.exists():
                    return path_full

        return path

    def glob(self, pattern: str):
        """
        Glob the given relative ``pattern`` in all search paths, yielding all
        matching files (of any kind). This function internally uses
        :func:`pathlib.Path.glob` and returns a generator.

        Parameters
        ----------
        pattern : str
            Pattern used for globbing.

        Yields
        ------
        :class:`~pathlib.Path`
            Globbed paths.
        """
        for path_base in self.paths:
            yield from path_base.glob(pattern)


#: Unique :class:`.PathResolver` instance.
path_resolver = PathResolver()
