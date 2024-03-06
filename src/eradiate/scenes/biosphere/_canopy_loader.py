from __future__ import annotations

import json
import typing as t
from pathlib import Path

import numpy as np

SCENARIO_FILE_NAME = "scenario.json"


def _update_material(
    elem: dict,
    canopy_name: str,
    spectral_data: dict[str, t.Any | dict[str, t.Any]] | None,
) -> dict:
    """
    Update the material of an element with spectral data if available.

    Parameters
    ----------
    elem : dict
        The element to update.
    canopy_name : str
        The name of the canopy or ground.
    spectral_data : dict[str, t.Any | dict[str, t.Any]] | None
        The spectral data to apply to the element.

    Returns
    -------
    dict
        The updated element.
    """
    elem_copy = elem.copy()
    if (
        spectral_data is not None
        and canopy_name in spectral_data
        and (canopy_name == "ground" or elem["id"] in spectral_data[canopy_name])
    ):
        assert isinstance(spectral_data, dict)
        try:
            del elem_copy["reflectance"]
        except KeyError:
            pass
        try:
            del elem_copy["transmittance"]
        except KeyError:
            pass
        if canopy_name == "ground":
            return {**elem_copy, **spectral_data[canopy_name]}
        else:
            assert isinstance(spectral_data[canopy_name], dict)
            return {**elem_copy, **spectral_data[canopy_name][elem["id"]]}
    else:
        return elem


def _parse_rpv_surface(
    surface: dict[str, t.Any], spectral_data: dict[str, t.Any]
) -> dict[str, t.Any]:
    """
    Convert surface data to RPV representation.

    Parameters
    ----------
    surface : dict
        Surface data to convert.
    spectral_data : dict
        Spectral data to apply to the surface.

    Returns
    -------
    dict
        RPV representation of the surface data.
    """
    return {
        **surface,
        **_update_material(
            {"rho_0": surface.get("rho_0", {})},
            "ground",
            spectral_data,
        ),
        **_update_material(
            {"g": surface.get("g", {})},
            "ground",
            spectral_data,
        ),
        **_update_material(
            {"k": surface.get("k", {})},
            "ground",
            spectral_data,
        ),
    }


def _parse_lambertian_surface(
    surface: dict[str, t.Any], spectral_data: dict[str, t.Any]
) -> dict[str, t.Any]:
    """
    Convert surface data to reflectance representation.

    Parameters
    ----------
    surface : dict
        Surface data to convert.
    spectral_data : dict
        Spectral data to apply to the surface.

    Returns
    -------
    dict
        Reflectance representation of the surface data.
    """
    return {
        **surface,
        **_update_material(
            {"reflectance": surface.get("reflectance", {})},
            "ground",
            spectral_data,
        ),
        **(
            _update_material(
                {"transmittance": surface.get("transmittance", {})},
                "ground",
                spectral_data,
            )
            if "transmittance" in surface
            else {}
        ),
    }


def apply_transformation(transf: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Apply transformation matrix to origin and translate the instance positions to the center of the scenario.

    Parameters
    ----------
    transf : np.ndarray
        Array of transformation values.
    center : np.ndarray
        Array representing the center of the scenario.

    Returns
    -------
    np.ndarray
        Transformed positions with center adjustment.
    """
    origin = np.array([0.0, 0.0, 0.0, 1.0]).T

    # Apply the transformation matrix to origin and adjust the center
    return (transf @ origin)[:3].T - center


def load_scenario(
    scenario_folder: Path,
    padding: int,
    spectral_data: dict[str, t.Any | dict[str, t.Any]] | None = None,
) -> dict:
    """
    Parse JSON file of scenario from a given path. Apply transformation to the
    data by converting units of wavelengths and compute instance positions.

    Parameters
    ----------
    scenario_folder : path-like
        Path of the folder containing scenario JSON file.
    padding : int
        Padding to apply to the scenario.
    spectral_data : dict[str, t.Any or dict[str, t.Any]] or None
        Spectral data to apply to the scenario, defaults to None (keep original).
        Example:

        .. code:: python

            spectral_data = {
                "ground": {
                    "reflectance": ground_reflectance,
                },
                "object_name": {
                    "subobject_name": {
                        # Spectral data for the subobject, such as
                        "reflectance": reflectance,
                        "transmittance": transmittance,
                    },
                }
            }

        Each spectral data specified replaces the original data completely, so
        it is necessary to specify all the data for the object.

    Returns
    -------
    dict
        Returns a dictionary parsed from JSON with transformations applied.
    """

    # Load "scenario.json" as dictionary object
    scenario = json.loads((scenario_folder / SCENARIO_FILE_NAME).read_text())

    # Update dictionary elements with transformations
    surface = scenario["surface"]
    size = scenario["canopy"]["size"]
    center_2d = np.array([size[0], size[1], 0.0]) / 2
    return {
        **scenario,
        "surface": (
            _parse_rpv_surface(surface, spectral_data)
            if surface["type"] == "rpv"
            else _parse_lambertian_surface(surface, spectral_data)
        ),
        "canopy": {
            **scenario["canopy"],
            "instanced_canopy_elements": [
                {
                    **elem,
                    "instance_positions": [
                        apply_transformation(transf, center_2d)
                        for transf in elem["instance_positions"]
                    ],
                    "canopy_element": {
                        **elem["canopy_element"],
                        "mesh_tree_elements": [
                            _update_material(
                                {
                                    **tree,
                                    "mesh_filename": (
                                        scenario_folder / tree["mesh_filename"]
                                    ),
                                },
                                canopy_name=elem["canopy_element"]["id"],
                                spectral_data=spectral_data,
                            )
                            for tree in elem["canopy_element"]["mesh_tree_elements"]
                        ],
                    },
                }
                for elem in scenario["canopy"]["instanced_canopy_elements"]
            ],
            "padding": padding,
        },
    }
