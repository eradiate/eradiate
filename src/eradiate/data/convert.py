from __future__ import annotations

import datetime

import numpy as np
import pint
import xarray as xr

from ..units import symbol, to_quantity
from ..units import unit_registry as ureg


def make_aer_core_v2(
    w: pint.Quantity,
    phamat: list[str],
    mu: pint.Quantity,
    theta: pint.Quantity,
    ext: pint.Quantity,
    ssa: pint.Quantity,
    phase: pint.Quantity,
    attrs: dict,
) -> xr.Dataset:
    """
    Create a new dataset in the Aer-Core v2 format.

    Parameters
    ----------
    w : quantity
        Wavelength.

    phamat : list of str
        Phase matrix component list.

    mu : quantity
        Scattering angle cosine.

    theta : quantity
        Scattering angle.

    ext : quantity
        Extinction coefficient.

    ssa : quantity
        Single-scattering albedo.

    phase : quantity
        Phase matrix values.

    attrs : dict
        Dataset attributes.

    Returns
    -------
    Dataset
    """
    data_vars = {
        "ext": (
            "w",
            ext.m,
            {
                "standard_name": "extinction_coefficient",
                "long_name": "extinction coefficient",
                "units": symbol(ext.u),
            },
        ),
        "ssa": (
            "w",
            ssa.m,
            {
                "standard_name": "single_scattering_albedo",
                "long_name": "single-scattering albedo",
                "units": symbol(ssa.u),
            },
        ),
        "phase": (
            ["phamat", "w", "iangle"],
            phase.m,
            {
                "standard_name": "phase_matrix",
                "long_name": "phase matrix",
                "units": symbol(phase.u),
            },
        ),
        # "nmom": ("w", [1]),
        # "pmom": (["w", "imom"], [[0]])
    }

    coords = {
        "w": (
            "w",
            w.m,
            {
                "standard_name": "wavelength",
                "long_name": "wavelength",
                "units": symbol(w.u),
            },
        ),
        "phamat": (
            "phamat",
            phamat,
            {
                "standard_name": "component",
                "long_name": "phase matrix component",
            },
        ),
        "mu": (
            ["w", "iangle"],
            mu,
            {
                "standard_name": "cos_scattering_angle",
                "long_name": "scattering angle cosine",
                "units": "",
            },
        ),
        "theta": (
            ["w", "iangle"],
            theta.m,
            {
                "standard_name": "scattering_angle",
                "long_name": "scattering angle",
                "units": symbol(theta.u),
            },
        ),
        # "imom": ("imom", [0]),
    }

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def aer_v1_to_aer_core_v2(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert a dataset in the Aer v1 format to Aer-Core v2.

    Parameters
    ----------
    ds : Dataset

    Returns
    -------
    Dataset
    """

    # This list maps phamat values to (i, j) pairs. The order follows libRadtran's
    # layout (first, coefficients for spherical particles; then additional coefficients
    # for spheroidal particles)
    PHAMAT_TO_IDX = [
        # Coefficients for spherical particles
        ("11", (0, 0)),
        ("12", (0, 1)),
        ("33", (2, 2)),
        ("34", (2, 3)),
        # Additional for spheroidal particles
        ("22", (1, 1)),
        ("44", (3, 3)),
    ]

    w = to_quantity(ds["w"]).astype("float32")
    ext = to_quantity(ds["sigma_t"]).astype("float32")
    ssa = to_quantity(ds["albedo"]).astype("float32")
    nangles = ds.sizes["mu"]
    nw = ds.sizes["w"]
    mu = np.broadcast_to(ds["mu"].values.astype("float32"), (nw, nangles))
    theta = (np.acos(mu) * ureg("rad")).astype("float32")

    _phase_datasets = {}
    for ij, (i, j) in PHAMAT_TO_IDX:
        try:
            _phase_datasets[ij] = ds["phase"].sel(i=i, j=j, drop=True)
        except KeyError:
            pass

    # Remove duplicates (indicate particle symmetry)
    for x, y in [("11", "22"), ("33", "44")]:
        try:
            if np.allclose(_phase_datasets[x], _phase_datasets[y]):
                del _phase_datasets[y]
        except KeyError:
            pass

    phase = to_quantity(
        xr.concat(_phase_datasets.values(), dim="phamat").transpose("phamat", "w", "mu")
    ).astype("float32")

    attrs = ds.attrs.copy()
    utcnow = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    attrs["history"] += f"\n{utcnow} - Conversion from Aer v1 to Aer-Core v2"

    if len(_phase_datasets) == 1 or len(_phase_datasets) == 4:
        attrs["particle_shape"] = "sphere"
    elif len(_phase_datasets) == 6:
        attrs["particle_shape"] = "spheroid"
    else:
        raise ValueError("could not determine particle shape")

    return make_aer_core_v2(
        w=w,
        phamat=list(_phase_datasets.keys()),
        mu=mu,
        theta=theta,
        ext=ext,
        ssa=ssa,
        phase=phase,
        attrs=attrs,
    )


def libradtran_to_aer_core_v2(ds: xr.Dataset) -> xr.Dataset:
    pass
