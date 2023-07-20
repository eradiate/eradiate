import typing as t

import numpy as np
import pandas as pd

from . import InstancedCanopyElement, MeshTree
from ..spectra import InterpolatedSpectrum
from ... import data
from ...units import unit_registry as ureg


def wellington_citrus_orchard(
    padding=0, properties: t.Literal["rami", "hyperspectral"] = "rami"
) -> dict:
    """
    Generate a base experiment dictionary based on the Wellington Citrus Orchard
    scene from the RAMI benchmark series.

    Parameters
    ----------
    padding : int, optional, default: 0
        Amount of padding applied to the generated canopy.

    properties : {"rami", "hyperspectral"}, optional, default: "rami"
        If set to ``"rami"``, the radiative properties used for the RAMI
        benchmark series is used. If set to ``"hyperspectral"``, leaf and soil
        reflectance are replaced with spectral datasets.

    Returns
    -------
    dict
        A dictionary suitable for keyword argument specification for the
        :class:`.CanopyExperiment` and :class:`CanopyAtmosphereExperiment`
        constructors. It contains the surface and canopy specifications.

    Warnings
    --------
    * This is an experimental feature, please use with caution and report issues.
    * The underlying data is sourced from the scene file archived distributed by
      the DART team and post-processed to accommodate Eradiate-specific frame
      conventions and formats. We do not recommend using it to run RAMI
      benchmark cases.
    """
    spectrum_table = pd.read_csv(
        data.data_store.fetch("trees/citrus_sinensis/spectral.txt"),
        sep=r"\s+",
        comment="#",
        index_col=0,
    )

    # Collect wavelengths
    w = (
        0.5
        * (spectrum_table.loc["WLMIN"].values + spectrum_table.loc["WLMAX"].values)
        * ureg.nm
    )  # Central wavelength for each band

    # Collect trunk reflectance
    trunk_reflectance = InterpolatedSpectrum(
        quantity="reflectance",
        wavelengths=w,
        values=spectrum_table.loc["CTR[1-10]_WOOD_R"].values,
    )

    if properties == "rami":
        # Collect ground reflectance
        ground_reflectance = InterpolatedSpectrum(
            quantity="reflectance",
            wavelengths=w,
            values=spectrum_table.loc["BGROUND_REFL"].values,
        )

        # Collect foliage reflectance/transmittance and define individual trees
        citrus_sinensis = []

        for i in range(1, 11):
            transmittance = InterpolatedSpectrum(
                quantity="transmittance",
                wavelengths=w,
                values=spectrum_table.loc[f"CTR{i}_FOLI_TRAN"].values,
            )
            reflectance = InterpolatedSpectrum(
                quantity="reflectance",
                wavelengths=w,
                values=spectrum_table.loc[f"CTR{i}_FOLI_REFL"].values,
            )
            citrus_sinensis.append(
                MeshTree(
                    id=f"cisi{i}",
                    mesh_tree_elements=[
                        {
                            "id": f"trunk{i}",
                            "mesh_filename": data.data_store.fetch(
                                f"trees/citrus_sinensis/CISI{i}/trunk.ply"
                            ),
                            "reflectance": trunk_reflectance,
                        },
                        {
                            "id": f"foliage{i}",
                            "mesh_filename": data.data_store.fetch(
                                f"trees/citrus_sinensis/CISI{i}/foliage.ply"
                            ),
                            "transmittance": transmittance,
                            "reflectance": reflectance,
                        },
                    ],
                )
            )

    elif properties == "hyperspectral":
        # Collect ground reflectance
        ground_ds = data.open_dataset("spectra/reflectance/lambertian_soil.nc").sel(
            brightness="dark", drop=True
        )
        ground_reflectance = InterpolatedSpectrum(
            quantity="reflectance",
            wavelengths=ureg.Quantity(
                ground_ds.w.values,
                ground_ds.w.units,
            ),
            values=ground_ds["reflectance"].values,
        )

        # Collect foliage reflectance/transmittance and define individual trees
        citrus_sinensis = []
        foliage_ds = data.open_dataset(
            "spectra/reflectance/bilambertian_leaf_cellulose.nc"
        )
        w = ureg.Quantity(foliage_ds.w.values, foliage_ds.w.units)
        transmittance = InterpolatedSpectrum(
            quantity="transmittance",
            wavelengths=w,
            values=foliage_ds.transmittance.values,
        )
        reflectance = InterpolatedSpectrum(
            quantity="reflectance",
            wavelengths=w,
            values=foliage_ds.reflectance.values,
        )

        for i in range(1, 11):
            citrus_sinensis.append(
                MeshTree(
                    id=f"cisi{i}",
                    mesh_tree_elements=[
                        {
                            "id": f"trunk{i}",
                            "mesh_filename": data.data_store.fetch(
                                f"trees/citrus_sinensis/CISI{i}/trunk.ply"
                            ),
                            "reflectance": trunk_reflectance,
                        },
                        {
                            "id": f"foliage{i}",
                            "mesh_filename": data.data_store.fetch(
                                f"trees/citrus_sinensis/CISI{i}/foliage.ply"
                            ),
                            "transmittance": transmittance,
                            "reflectance": reflectance,
                        },
                    ],
                )
            )

    else:
        raise ValueError(f"unknown spectral property specification '{properties}'")

    # Load instance positions
    citrus_sinensis_locations = []

    for i in range(1, 11):
        loc = np.genfromtxt(
            data.data_store.fetch(f"trees/citrus_sinensis/CISI{i}/location.txt")
        )
        zeros = np.zeros((len(loc), 1))
        citrus_sinensis_locations.append(np.concatenate((loc, zeros), axis=1))

    instanced_elements = []

    for i, tree, locations in zip(
        range(1, 11), citrus_sinensis, citrus_sinensis_locations
    ):
        instanced_elements.append(
            InstancedCanopyElement(
                canopy_element=tree,
                instance_positions=locations,
            )
        )

    return {
        "surface": {"type": "lambertian", "reflectance": ground_reflectance},
        "canopy": {
            "type": "discrete_canopy",
            "instanced_canopy_elements": instanced_elements,
            "size": [100, 100, 10],
            "padding": padding,
        },
    }
