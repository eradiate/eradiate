"""
Utilities for the Eradiate test suite.
"""

import os
import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import eradiate


def skipif_data_not_found(
    dataset_category: str, dataset_id: str, action: t.Optional[t.Callable] = None
) -> None:
    """
    During a Pytest session, skip the current test if the referenced dataset
    cannot be found.

    Parameters
    ----------
    dataset_category : str
        Required dataset category.

    dataset_id : str
        Required dataset ID.

    action : callable, optional
        An optional callable with no arguments which performs an action prior to
        skipping the test (*e.g.* output an artefact placeholder).
    """
    dataset_path = eradiate.data.getter(dataset_category).PATHS[dataset_id]

    if not eradiate.data.find(dataset_category)[dataset_id]:
        if action is not None:
            action()

        pytest.skip(
            f"Could not find dataset '{dataset_category}.{dataset_id}'; "
            f"please download dataset files and place them in "
            f"'data/{dataset_path}' directory."
        )


def missing_artefact(filename: os.PathLike) -> None:
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
