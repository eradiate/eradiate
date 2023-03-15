from __future__ import annotations

import os
import typing as t
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import attrs
import pooch
import tqdm
from ruamel.yaml import YAML

from ..typing import PathLike


@attrs.define
class DataStore(ABC):
    """
    Interface class for all data stores.
    """

    @property
    @abstractmethod
    def base_url(self) -> str:
        """
        str : Address of the remote storage location.
        """
        pass

    @property
    @abstractmethod
    def registry(self) -> dict:
        """
        dict : Registry contents.
        """
        pass

    @abstractmethod
    def registry_files(
        self, filter: t.Callable[[t.Any], bool] | None = None
    ) -> list[str]:
        """
        Get a list of registered files.

        Parameters
        ----------
        filter : callable, optional
            A filter function taking a file path as a single string argument and
            returning a boolean. Filenames for which the filter returns ``True``
            will be returned.

        Returns
        -------
        files : list of str
            List of registered files.
        """
        pass

    @abstractmethod
    def fetch(
        self,
        filename: PathLike,
        **kwargs,
    ) -> Path:
        """
        Fetch a file from the data store.

        Parameters
        ----------
        filename : path-like
            File name to fetch from the data store, relative to the storage root.

        Returns
        -------
        Path
            Absolute path where the retrieved resource is located.

        Raises
        ------
        DataError
            The requested file could not be served.
        """
        pass


def registry_from_file(filename: PathLike, warn: bool = True) -> dict:
    """
    Read registry content from a file.

    Parameters
    ----------
    filename : path-like
        Path to the file to be read.

    warn : bool, optional
        If ``True``, ill-formed lines will result in a warning. Otherwise, an
        exception will be raised.

    Returns
    -------
    registry : dict
        Registry contents, parsed as a dictionary.

    Raises
    ------
    ValueError
        If `warn` is ``False`` and an error occurs while parsing the file.
    """
    result = {}

    with open(filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip comment lines
            if line.startswith("#"):
                continue

            items = line.split(maxsplit=1)

            # No metadata (just the filename)
            if len(items) == 1:
                result[items[0]] = ""
                continue

            # Found metadata
            if len(items) == 2:
                result[items[0]] = items[1]
                continue

            # Ill-formed line
            if len(items) > 2:
                if warn:
                    warnings.warn(
                        f"While parsing registry file {filename}: skipping "
                        f"ill-formed line {i}"
                    )
                    continue
                else:
                    raise ValueError(
                        f"While parsing registry file {filename}: "
                        f"ill-formed line {i}"
                    )

    return result


def registry_to_file(registry: dict, filename: PathLike) -> None:
    """
    Write a registry dictionary to a text file.

    Parameters
    ----------
    registry : dict
        A registry dictionary.

    filename : path-like
        Path to the file where registry contents are to be written.
    """
    lines = [f"{path} {registry[path]}".strip() for path in sorted(registry)]
    lines.append("")
    content = "\n".join(lines)
    with open(filename, "w") as f:
        f.write(content)


def load_rules(filename: PathLike) -> dict:
    """
    Load include and exclude rules from a YAML file.

    Parameters
    ----------
    filename : path-like
        Path to the YAML file from which rules are to be loaded.

    Returns
    -------
    rules : dict
        Dictionary containing a list of inclusion (resp. exclusion) rules under
        the ``"include"`` (resp. ``"exclude"``) key.
    """

    yaml = YAML()
    try:
        with open(filename) as f:
            rules = yaml.load(f)
    except OSError:
        rules = {"include": ["**/*.*"], "exclude": []}

    return rules


def expand_rules(
    rules: list[str],
    prefix: PathLike = ".",
    as_list: bool = False,
    include_dirs=False,
) -> list | set:
    """
    Expand a list of filesystem selection rules to paths.

    Parameters
    ----------
    rules : list of str
        List of inclusion to expand, relative to `prefix`. Each rule may be a
        path to a file or a shell glob.

    prefix : path-like, optional
        Path where to expand the rules. By default, the current working
        directory (``"."``) is used.

    as_list : bool, optional
        If ``True``, return the result as a sorted list; otherwise, return it
        as a set.

    include_dirs : bool, optional
        If ``True``, include directories in the list of expanded items.
        Otherwise, the expansion is restricted to files.

    Returns
    -------
    items : set or list
        Items corresponding to the expanded rules.
    """
    expanded = set()

    for rule in rules:
        expanded |= set(
            x for x in Path(prefix).rglob(rule) if (x.is_file() or include_dirs)
        )

    return sorted(expanded) if as_list else expanded


def list_files(
    path: PathLike,
    includes: list[str] | None = None,
    excludes: list[str] | None = None,
    as_list: bool = False,
) -> list | set:
    """
    List files in a directory based on inclusion and exclusion rules.

    Parameters
    ----------
    path : path-like
        Path to the target directory.

    includes : list of str, optional
        List of inclusion rules, relative to `path`. Each rule may be a path to
        a file or a shell glob. If no rule is passed, everything is included
        (*i.e.* ``["**/*"]`` is used).

    excludes : list of str, optional
        List of exclusion rules, relative to `path`. Each rule may be a path to
        a file or a shell glob. If no rule is passed, nothing is excluded
        (*i.e.* ``[]`` is used).

    as_list : bool
        If ``True``, return the result as a sorted list; otherwise, return it
        as a set.

    Returns
    -------
    items : set or list
        Files listed in the target directory, according to the rules.
    """
    if includes is None:
        includes = ["**/*"]
    if excludes is None:
        excludes = []

    included: set = expand_rules(includes, prefix=path)
    excluded: set = expand_rules(excludes, prefix=path)

    return sorted(included - excluded) if as_list else included - excluded


def make_registry(
    path: PathLike,
    filename: PathLike = "registry.txt",
    includes: list | None = None,
    excludes: list | None = None,
    alg="sha256",
    show_progress=False,
) -> None:
    """
    Create a registry file from items in a directory, possibly applying
    inclusion and exclusion rules.

    Parameters
    ----------
    path : path-like
        Path to the target directory.

    filename : path-like
        Path to the file where registry contents are to be written.

    includes : list of str, optional
        List of inclusion rules, relative to `path`. Each rule may be a path to
        a file or a shell glob. If no rule is passed, everything is included
        (*i.e.* ``["**/*"]`` is used).

    excludes : list of str, optional
        List of exclusion rules, relative to `path`. Each rule may be a path to
        a file or a shell glob. If no rule is passed, nothing is excluded
        (*i.e.* ``[]`` is used).

    alg : str, optional
        Hash algorithm used.
    """
    # Basically pooch.make_registry() with inclusion / exclusion rules
    files = list_files(path, includes, excludes)
    hashes = (
        [pooch.file_hash(x, alg=alg) for x in files]
        if alg is not None
        else ["" for _ in files]
    )

    registry = {}

    with tqdm.tqdm(total=len(hashes), disable=not show_progress) as pbar:
        for file, hash in zip(files, hashes):
            registry[os.path.relpath(file, path)] = f"{alg}:{hash}"
            pbar.update()

    registry_to_file(registry, filename)
