r"""Atmosphere vertical profile data structure definition and creation

.. admonition:: Profile dataset specification

    The data structure is a :class:`~xarray.Dataset` with specific data
    variables, dimensions and data coordinates.

    Data variables must be:

    - ``pressure``: pressure in the layer [Pa],
    - ``temperature``: temperature in the layer [K],
    - ``total_number_density``: total number density in the layer [m^-3],
    - ``number_density``: number density of the individual species in the layer [m^-3],
    - ``upper_bound``: altitude of each layer's upper bound [m],
    - ``lower_bound``: altitude of each layer's lower bound [m],

    Dimensions are ``altitude`` and ``species``. All data variables depend on
    the ``altitude`` dimension. Only ``number_density`` also has the
    ``species`` dimension.

    Data coordinates are:

    - ``altitude``: array of altitude values. Each altitude is representative of
        a layer, e.g. the altitude at the middle of the layer.
    - ``species``: array of strings that identifies the individual gas species.
"""

from datetime import datetime

import numpy as np

from ...util.units import ureg
from .us76 import create


def check(profile):
    r"""Checks that a given data set is an atmosphere profile.

    Parameter ``profile`` (:class:`~xarray.Dataset`):
        The atmosphere profile to check.

    Returns → bool:
         True if the data set is an atmosphere profile, False otherwise.
    """

    required_data_vars = [
        "pressure",
        "temperature",
        "total_number_density",
        "number_density",
        "upper_bound",
        "lower_bound"
    ]

    required_dims = ["altitude", "species"]
    required_coords = ["altitude", "species"]

    if not all(var in list(profile.data_vars) for var in required_data_vars):
        return False

    if not len(profile.data_vars) == 6:
        return False

    if not all(dim in list(profile.dims) for dim in required_dims):
        return False

    if not len(profile.dims) == 2:
        return False

    if not all(coord in list(profile.coords) for coord in required_coords):
        return False

    if not len(profile.coords) == 2:
        return False

    # TODO: add units check (units must be present and correct)

    return True


@ureg.wraps(ret=None, args=("m", None), strict=False)
def us76(height=ureg.Quantity(1e5, "m"), n_layers=50):
    r"""Generates an atmosphere vertical profile based on the
    US76 Standard Atmosphere.

    .. note::

        Profile variables are given at the centers of a uniform altitude mesh
        with :math:`n_\mathrm{layers} + 1` nodes between 0 and ``height``.

    Parameter ``height`` (float):
        Height of the atmosphere [m].
        Valid range: 0+ to 1000000 m.
        Default value: 100000 m.

    Parameter ``n_layers`` (int):
        Number of layers in the profile [dimensionless].
        Default value: 50.

    Returns → :class:`~xarray.Dataset`:
        Data set holding the values of the pressure, temperature,
        total number density and number densities of the individual
        gas species in each layer.
    """

    if height <= 0 or height > 1000000:
        raise ValueError("height must be in ]0, 1000000] meters")

    if n_layers <= 0:
        raise ValueError("n_layers must be positive")

    nodes = np.linspace(0.0, height, n_layers + 1)
    centers = (nodes[:-1] + nodes[1:]) / 2

    ds = create(
        ureg.Quantity(centers, "m"),
        variables=[
            "pressure",
            "temperature",
            "number_density",
            "total_number_density"
        ],
    )
    return ds.assign({
        "upper_bound": ("altitude", nodes[1:]),
        "lower_bound": ("altitude", nodes[:-1])
    })


def rescale_co2(profile, surf_ppmv, inplace=False):
    r"""Scales the number density of carbon dioxide at all altitude in an
    atmosphere profile, such that it reaches a given value in the first layer.

    Parameter ``profile`` (:class:`~xr.Dataset`):
        Atmosphere profile.
        Note: the profile must include the CO2 species.

    Parameter ``surf_ppmv`` (float):
        Number density of carbon dioxide at the surface [ppmv].

    Returns → :class:`~xr.Dataset`:
        Rescaled atmosphere profile.

    .. warning::
        This function is implemented for the user's convenience but may produce
        atmosphere profiles with unphysical data.

        Scaling gas species number densities in an atmosphere profile can have
        undesired consequences. If the gas species volume fraction is not small
        and/or if the scaling factor is large, the total number density is
        changed. Furthermore, depending on the model used to generate the
        atmosphere profile, the value of the total number density is related to
        the values of the temperature and pressure altitude (possibly to the
        values of the other gas species number densities), e.g. through the
        perfect gas law. Hence, a change in the total number density leads to a
        change in the other atmospheric variables. For a relatively small scaling
        factor, these changes are small and may be neglected. However,
        no restrictions (other than physical) is imposed on the value of the
        scaling factor here. The user has the freedom to set the scaling factor
        but they are also responsible whether the resulting atmosphere profile is
        physically meaningful.

        Since this function does not know about the relation between the
        atmospheric variables, only the total number density variable is
        updated to reflect the change in the CO2 number density.
    """

    if "CO2" not in profile["species"]:
        raise ValueError("this profile does not contain CO2")

    if 1e6 <= surf_ppmv <= 0:
        raise ValueError("surf_ppmv must be in ]0, 1e6[")

    if inplace:
        ds = profile
    else:
        ds = profile.copy(deep=True)

    # compute the scaling factor
    initial_density = \
        ds["number_density_individual_species"].loc[dict(species="CO2")].values
    surface_density = initial_density[0]
    surface_total_density = ds["total_number_density"].values[0]
    initial_surf_ppmv = (surface_density / surface_total_density) * 1e6

    if initial_surf_ppmv != 0:
        scaling_factor = surf_ppmv / initial_surf_ppmv
    else:
        raise ZeroDivisionError("cannot scale CO2 number density because "
                                "surface CO2 number density is zero")

    # scale CO2 number density in profile
    ds["number_density_individual_species"].loc[dict(species="CO2")] *= \
        scaling_factor

    # propagate the change of CO2 number density to the total number density
    new_number_density = \
        ds["number_density_individual_species"].loc[dict(species="CO2")].values
    ds["total_number_density"].loc[dict()] += \
        new_number_density - initial_density

    # update data set history
    ds.attrs["history"].loc[dict()] += \
        f"\n" \
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - " \
        f"CO2 concentration rescaling - " \
        f"eradiate.scenes.atmosphere.profiles.scale_co2()"

    return ds
