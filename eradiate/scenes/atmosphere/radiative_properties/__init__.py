"""Atmospheric radiative properties calculation package.

.. admonition:: Atmospheric radiative properties data set specification (1D).

    The data structure is a :class:`~xarray.Dataset` with specific data
    variables, dimensions and data coordinates.

    Data variables must be:
    - ``sigma_a``: absorption coefficient [m^-1],
    - ``sigma_s``: scattering coefficient [m^-1],
    - ``sigma_t``: extinction coefficient [m^-1],
    - ``albedo``: albedo [dimensionless]

    The dimensions are ``z_layer`` and ``z_level``. All data variables depend
    on ``z_layer``.

    The data coordinates are:
    - ``z_layer``: layer altitude [m]. The layer altitude is an altitude representative of the given layer, e.g. the middle of the layer.
    - ``z_level``: level altitude [m]. The sole purpose of this data coordinate is to store the information on the layers sizes.
"""

from datetime import datetime

import numpy as np
import xarray as xr

from eradiate import __version__
from ....util.units import ureg
from .rayleigh import sigma_s_air

_Q = ureg.Quantity


def compute_monochromatic_radiative_properties(profile, wavelength=550.,
                                               scattering_on=True,
                                               absorption_on=True):
    r"""Compute the monochromatic radiative properties corresponding to a given
    atmospheric vertical profile (1D atmospheric thermophysical field).

    Parameter ``profile`` (:class:`~xr.Dataset`):
        Atmospheric vertical profile.

    Parameter ``wavelength`` (float):
        Wavelength [nm].

    Parameter ``scattering_on`` (bool):
        If True, the scattering properties are computed.

    Parameter ``absorption_on`` (bool):
        If True, the absorption properties are computed.

        .. note::
            At least one of ``scattering_on`` and ``absorption_on`` must be
            True.

    Returns → :class:`~xr.Dataset`
        Atmospheric radiative properties data set.
    """

    properties = create_dataset_from_profile(profile)

    # compute absorption coefficient
    sigma_a = np.full(profile.n.shape, np.nan)
    if absorption_on:
        raise NotImplementedError

    # compute scattering coefficient
    sigma_s = np.full(profile.n.shape, np.nan)
    if scattering_on:
        sigma_s = sigma_s_air(
            wavelength=wavelength,
            number_density=_Q(profile.n_tot.values, profile.n_tot.units)
        )
        properties.sigma_s.values = sigma_s.to("m^-1").magnitude

    # compute the extinction coefficient and albedo
    if absorption_on:
        if scattering_on:
            sigma_t = sigma_a + sigma_s
            properties.sigma_t.values = sigma_t
            properties.albedo.values = sigma_s / sigma_t
        else:
            properties.sigma_t.values = sigma_a
            properties.albedo.values = np.full(sigma_a.shape, 0.)

    else:
        if scattering_on:
            properties.sigma_t.values = sigma_s
            properties.albedo.values = np.full(sigma_s.shape, 1.)
        else:
            raise ValueError("absorption_on and scattering_on both False. At"
                             " least one of the two must be True.")

    return properties


def create_dataset_from_profile(profile):
    r"""Initialises the atmospheric radiative properties data set.

    Parameter ``profile`` (:class:`~xr.Dataset`):
        Atmospheric vertical profile.

    Returns → :class:`~xarray.Dataset`:
        Initialised data set.
    """

    z_layer = profile.z_layer.values
    data_vars = {
        "sigma_a": (
            "z_layer",
            np.full(z_layer.shape, np.nan),
            {
                "units": "m^-1",
                "standard_name": "absorption_coefficient"
            }
        ),
        "sigma_s": (
            "z_layer",
            np.full(z_layer.shape, np.nan),
            {
                "units": "m^-1",
                "standard_name": "scattering_coefficient"
            }
        ),
        "sigma_t": (
            "z_layer",
            np.full(z_layer.shape, np.nan),
            {
                "units": "m^-1",
                "standard_name": "extinction_coefficient"
            }
        ),
        "albedo": (
            "z_layer",
            np.full(z_layer.shape, np.nan),
            {
                "units": "m^-1",
                "standard_name": "albedo"
            }
        )
    }

    coords = {
        "z_layer": ("z_layer", profile.z_layer.values, {"units": ""}),
        "z_level": ("z_level", profile.z_level.values, {"units": ""})
    }

    # TODO: set function name in history field dynamically
    attrs = {
        "convention": "CF-1.8",
        "title": "Atmospheric monochromatic radiative properties",
        "history":
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
            f"data set initialisation - "
            f"eradiate.scenes.atmosphere.radiative_properties.init_data_set",
        "source": f"eradiate, version {__version__}",
        "references": "",
    }

    return xr.Dataset(data_vars, coords, attrs)
