"""
Utilities for the Eradiate test suite.
"""

from __future__ import annotations

import os
import typing as t
from pathlib import Path
from typing import Any, Callable, TypeVar

import matplotlib.pyplot as plt
import pytest
from typing_extensions import ParamSpec, TypeAlias

from .. import fresolver
from ..typing import PathLike


def skipif_data_not_found(path: PathLike, action: t.Callable | None = None) -> None:
    """
    During a Pytest session, skip the current test if the referenced dataset
    cannot be resolved by the file resolver.

    Parameters
    ----------
    path : path-like
        Path to the required data file.

    action : callable, optional
        An optional callable with no arguments which performs an action prior to
        skipping the test (*e.g.* output an artefact placeholder).
    """
    try:
        fresolver.resolve(path, strict=True, cwd=False)
    except FileNotFoundError:
        if action is not None:
            action()

        pytest.skip(f"File resolver could not resolve file '{path}'.")


def missing_artefact(filename: PathLike) -> None:
    """
    Create a placeholder file for a missing artefact.

    Parameters
    ----------
    filename : path-like
        Path to the created placeholder.

    Raises
    ------
    ValueError
        Required file type is not supported.

    Notes
    -----
    The following types are currently supported:

    * PNG
    """
    filename = Path(filename)

    if filename.suffix == ".png":
        plt.figure()
        plt.text(0.5, 0.5, "Placeholder", fontsize="xx-large", ha="center")
        plt.gca().set_aspect(1)
        os.makedirs(filename.parent, exist_ok=True)
        plt.savefig(filename)
        plt.close()

    else:
        raise ValueError(f"unsupported file extension {filename.suffix}")


T = TypeVar("T")
P = ParamSpec("P")
WrappedFuncDeco: TypeAlias = Callable[[Callable[P, T]], Callable[P, T]]


def copy_doc(copy_func: Callable[..., Any]) -> WrappedFuncDeco[P, T]:
    """Copies the doc string of the given function to another.
    This function is intended to be used as a decorator.

    .. code-block:: python3

        def foo():
            '''This is a foo doc string'''
            ...

        @copy_doc(foo)
        def bar():
            ...
    """

    def wrapped(func: Callable[P, T]) -> Callable[P, T]:
        func.__doc__ = copy_func.__doc__
        return func

    return wrapped


def append_doc(copy_func: Callable[..., Any], prepend=False) -> WrappedFuncDeco[P, T]:
    """Append the doc string of the given function to another.
    If prepend is true, will place the copied doc string in front
    of the decorated function's doc string.
    This function is intended to be used as a decorator.

    .. code-block:: python3

        def foo():
            '''This is a foo doc string'''
            ...

        @append_doc(foo)
        def bar():
            '''This is a bar doc string'''
            ...
    """

    def wrapped(func: Callable[P, T]) -> Callable[P, T]:
        if prepend:
            func.__doc__ = copy_func.__doc__ + "\n" + func.__doc__
        else:
            func.__doc__ = func.__doc__ + "\n" + copy_func.__doc__
        return func

    return wrapped


def check_plugin(config, name):
    items = dict(config.pluginmanager.list_name_plugin())
    if name in items:
        if items[name] is not None:
            return True
    return False
