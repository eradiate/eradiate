"""Utility functions to manipulate atmospheric profiles. """

from datetime import datetime

import iapws
import numpy as np
import xarray as xr

from .._units import unit_registry as ureg
from ..xarray.metadata import (
    DatasetSpec,
    VarSpec
)


profile_dataset_spec = DatasetSpec(
    var_specs={
        "p": VarSpec(standard_name="air_pressure", units="Pa", long_name="air pressure"),
        "t": VarSpec(standard_name="air_temperature", units="K", long_name="air temperature"),
        "n": VarSpec(standard_name="number_density", units="m^-3", long_name="number density"),
        "n_tot": VarSpec(standard_name="air_number_density", units="m^-3",
                         long_name="air number density"),
    },
    coord_specs="atmospheric_profile"
)


@ureg.wraps(ret="Pa", args="K", strict=False)
def water_vapor_saturation_pressure(t):
    """Computes the water vapor saturation pressure over liquid water or ice,
    at the given temperature.

    .. note::
        Valid for pressures larger than the triple point pressure (~611 Pa).

    Parameter ``t`` (float):
        Temperature [K].

    Returns → float:
        Water vapor saturation pressure [Pa].
    """
    if t >= 273.15:  # water is liquid
        p = ureg.Quantity(iapws.iapws97._PSat_T(t), "MPa")
    else:  # water is solid
        p = ureg.Quantity(iapws._iapws._Sublimation_Pressure(t), "MPa")
    return p.to("Pa").magnitude


@ureg.wraps(ret=ureg.dimensionless, args=("Pa", "K"), strict=False)
def equilibrium_water_vapor_fraction(p, t):
    """Computes the water vapor volume fraction at equilibrium, i.e., when the
    rate of condensation of water vapor equals the rate of evaporation of
    liquid water or ice, depending on the temperature.

    The water vapor volume fraction :math:`x_w` is computed with:

    .. math::
       x_w(p,T) = \\frac{p_w(T)}{p}

    where
    :math:`p` is the pressure,
    :math:`T` is the temperature and
    :math:`p_w` is the water vapor saturation pressure at the given temperature.

    This water vapor volume fraction corresponds to a relative humidity of 100%.

    .. note::
       For some values of the pressure and temperature, the equilibrium does
       not exist.
       An exception is raised in those cases.

    Parameter ``p`` (float):
        Pressure [Pa].

    Parameter ``t`` (float):
        Temperature [K].

    Returns → float:
        Water vapor volume fraction [dimensionless].
    """
    p_water_vapor = water_vapor_saturation_pressure(t).magnitude
    if p_water_vapor <= p:
        return p_water_vapor / p
    else:
        raise ValueError(
            f"Equilibrium cannot be reached in these conditions (p = "
            f"{round(p, 2)} Pa, t = {round(t, 2)} K)")


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
        atmosphere thermoprops with unphysical data.

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
        f"eradiate.thermoprops.scale_co2()"

    return ds


def make_profile_regular(profile, atol):
    r"""Converts the atmospheric profile with an irregular altitude mesh to a
    profile defined over a regular altitude mesh

    Parameter ``profile`` (:class:`~xr.Dataset`):
        Original atmospheric profile, defined over an irregular altitude mesh.

    Parameter ``atol`` (float):
        Absolute tolerance used in the conversion of the irregular altitude mesh
        to a regular altitude mesh.

    Returns -> :class:`~xr.Dataset`:
        Converted atmospheric profile, defined over a regular altitude mesh.
    """
    profile.ert.validate_metadata(profile_dataset_spec)

    # compute the regular altitude nodes mesh
    regular_z_level = _to_regular(mesh=profile.z_level.values, atol=atol)

    # compute corresponding altitude centers mesh
    regular_z_layer = (regular_z_level[:-1] + regular_z_level[1:]) / 2.

    # compute the atmospheric variables with the regular altitude mesh
    n_z = len(regular_z_layer)
    p = np.zeros(n_z)
    t = np.zeros(n_z)
    n_tot = np.zeros(n_z)
    n_species = profile.species.size
    n = np.zeros((n_species, n_z))
    layer = 0
    for i, z in enumerate(regular_z_layer):
        # when altitude is larger than current layer's upper bound, jump to
        # the next layer
        if z >= profile.z_level.values[layer + 1]:
            layer += 1

        p[i] = profile["p"].values[layer]
        t[i] = profile["t"].values[layer]
        n_tot[i] = profile["n_tot"].values[layer]
        n[:, i] = profile["n"].values[:, layer]

    species = profile["species"].values

    # copy attributes
    attrs = profile.attrs

    # update history
    new_line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - made " \
               f"profile altitude mesh regular - " \
               f"eradiate.thermoprops.util.make_profile_regular "
    attrs["history"] += f"\n{new_line}"

    dataset = xr.Dataset(
        data_vars={
            "p": ("z_layer", p),
            "t": ("z_layer", t),
            "n_tot": ("z_layer", n_tot),
            "n": (("species", "z_layer"), n)
        },
        coords={
            "z_layer": ("z_layer", regular_z_layer),
            "z_level": ("z_level", regular_z_level),
            "species": ("species", species)
        },
        attrs=attrs
    )
    dataset.ert.normalize_metadata(profile_dataset_spec)

    return dataset


def _to_regular(mesh, atol):
    r"""Converts an irregular altitude mesh into a regular altitude mesh.

    .. note::
        The bound altitudes in the irregular altitude mesh remain the same in
        the output regular altitude mesh. Only the intermediate node altitudes
        are modified.

    .. warning::
        The algorithm is not optimised to find the approximating regular mesh
        with the smallest number of layers. Depending on the value of ``atol``,
        the resulting mesh size can be large.

    Parameter ``mesh`` (array):
        Irregular altitude mesh with values sorted in increasing order.

    Parameter ``atol`` (float):
        Absolute tolerance used in the conversion.

    Returns -> array:
        Regular altitude mesh.
    """

    n, _ = _find_regular_params_gcd(mesh, atol)
    return np.linspace(start=mesh[0], stop=mesh[-1], num=n)


def _find_regular_params_gcd(mesh, unit_number=1.):
    r"""Finds the parameters (number of cells, constant cell width) of the
    regular mesh that approximates the irregular input mesh.

    The algorithm finds the greatest common divisor (GCD) of all cells widths in
    the integer representation specified by the parameter ``unit_number``.
    This GCD is used to define the constant cells width of the approximating
    regular mesh.

    .. warning::
        There are no safeguards regarding how large the number of cells in the
        regular mesh can be. Use the parameter ``unit_number`` with caution.

    Parameter ``mesh`` (array):
        1-D array with floating point values.
        Values must be sorted by increasing order.

    Parameter ``unit_number`` (float):
        Defines the unit used to convert the floating point numbers to integer.
        numbers.

        Default: 1.

    Returns -> int, float:
        Number of points in the regular mesh and the value of the constant cells
        width.
    """

    # Convert float cell widths to integer cell widths
    eps = np.finfo(float).eps
    if unit_number >= eps:
        mesh = np.divide(mesh, unit_number).astype(int)
    else:
        raise ValueError(f"Parameter unit_number ({unit_number}) must be "
                         f"larger than machine epsilon ({eps}).")
    widths = mesh[1:] - mesh[:-1]

    # Find the greatest common divisor (GCD) of all integer cell widths
    # The constant cell width in the regular mesh is given by that GCD.
    from math import gcd
    w = gcd(widths[0], widths[1])
    for x in widths[2:]:
        w = gcd(x, w)

    # Compute the number of points in the regular mesh
    total_width = mesh[-1] - mesh[0]
    n = total_width // w + 1

    return n, float(w) * unit_number

# def find_regular_params_tol(mesh, rtol=1e-3, n_cells_max=10000):
#     r"""Finds the number of cells and constant cell width of the regular 1-D
#     mesh that approximates a 1-D irregular mesh the best.
#
#     Parameter ``mesh`` (array):
#         Irregular 1-D mesh. Values must be sorted in increasing order.
#
#     Parameter ``rtol`` (float):
#         Relative tolerance on the cells widths. This parameter controls the
#         accuracy of the approximation.
#         The parameters of the approximating regular mesh are computed so that
#         for each layer of the irregular mesh, the corresponding layer or layers
#         in the regular mesh has a width or have a total width that is not larger
#         than ``rtol`` times the width of the cell in the irregular mesh.
#
#         Default: 1e-3
#
#     Parameter ``n_cells_max`` (float):
#         Maximum number of cells in the regular mesh. This parameter controls the
#         size of the resulting regular mesh.
#
#         Default: 10000
#
#     Returns -> int, float:
#         Number of cells and constant cells width in the approximating regular
#         mesh.
#     """
#
#     raise NotImplemented
# TODO: implement this function
