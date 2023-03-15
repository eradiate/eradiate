"""
Utilities for the Eradiate test suite.
"""
from __future__ import annotations

import os
import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from ..data import data_store
from ..exceptions import DataError
from ..typing import PathLike


def skipif_data_not_found(path: PathLike, action: t.Callable | None = None) -> None:
    """
    During a Pytest session, skip the current test if the referenced dataset
    cannot be fetched from the data store.

    Parameters
    ----------
    path : path-like
        Path to the required data file (in the data store).

    action : callable, optional
        An optional callable with no arguments which performs an action prior to
        skipping the test (*e.g.* output an artefact placeholder).
    """
    try:
        data_store.fetch(path)
    except DataError:
        if action is not None:
            action()

        pytest.skip(f"Could not find dataset '{path}' in the data store.")


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
