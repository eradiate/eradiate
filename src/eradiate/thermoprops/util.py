"""
Utility functions to manipulate atmosphere thermophysical properties data
sets.
"""
from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pint
import scipy.constants
import xarray as xr

from .. import data
from ..units import to_quantity
from ..units import unit_registry as ureg

ATOMIC_MASS_CONSTANT = ureg.Quantity(
    *scipy.constants.physical_constants["atomic mass constant"][:-1]
)


def column_number_density(ds: xr.Dataset, species: str) -> pint.Quantity:
    """
    Computes the column number density of a given species in an atmospheric
    profile.
    The column number density is computed according to the formula:

    .. math::
       N = \\sum_{i=0}^{L-1} n_i \\, (z_{i+1} - z_i)

    where

    :math:`N` denotes the column number density,
    :math:`z_i` are the level altitudes and
    :math:`n_i` are the number densities of that given species inside the
    :math:`L` atmospheric layers.

    Parameters
    ----------
    ds : Dataset
        Atmosphere thermophysical properties data set.

    species : str
        Species.

    Returns
    -------
    quantity
        Column number density.
    """
    mr = ds.mr.sel(species=species).values
    n = to_quantity(ds.n)
    n_species = mr * n
    z_level = to_quantity(ds.z_level)
    dz = z_level[1:] - z_level[:-1]
    return (n_species * dz).sum()


def column_mass_density(ds: xr.Dataset, species: str) -> pint.Quantity:
    """
    Computes the column (mass) density of a given species in an atmospheric
    profile.
    The column mass density is computed according to the formula:

    .. math::
       \\sigma = N * m_u

    where

    :math:`\\sigma` denotes the column mass density,
    :math:`N` is the column number density,
    :math:`m_u` is the atomic mass constant.

    Parameters
    ----------
    ds : Dataset
        Atmosphere thermophysical property data set.

    species : str
        Species.

    Returns
    -------
    quantity
        Column mass density.
    """
    with data.open_dataset("chemistry/molecular_masses.nc") as molecular_mass:
        m = molecular_mass.m.sel(s=species).values * ATOMIC_MASS_CONSTANT

    return m * column_number_density(ds=ds, species=species)


def number_density_at_surface(ds: xr.Dataset, species: str) -> pint.Quantity:
    """
    Compute the number density at the surface of a given species in an
    atmospheric profile.

    Parameters
    ----------
    ds : Dataset
        Atmosphere thermophysical properties data set.

    species : str
        Species.

    Returns
    -------
    :class:`~pint.Quantity`
        Number density at the surface.
    """
    surface_mr = to_quantity(ds.mr.sel(species=species))[0]
    surface_n = to_quantity(ds.n)[0]
    return surface_mr * surface_n


def mass_density_at_surface(ds: xr.Dataset, species: str) -> pint.Quantity:
    """Compute the mass density at the surface of a given species in an
    atmospheric profile.

    Parameters
    ----------
    ds : Dataset
        Atmosphere thermophysical properties data set.

    species : str
        Species.

    Returns
    -------
    quantity
        Mass density at the surface.
    """
    with data.open_dataset("chemistry/molecular_masses.nc") as molecular_mass:
        m = to_quantity(molecular_mass.m.sel(s=species)) * ATOMIC_MASS_CONSTANT

    return m * number_density_at_surface(ds=ds, species=species)


def volume_mixing_ratio_at_surface(ds: xr.Dataset, species: str) -> pint.Quantity:
    """Compute the volume mixing ratio at the surface of a given species in an
    atmospheric profile.

    Parameters
    ----------
    ds : Dataset
        Atmosphere thermophysical properties data set.

    species : str
        Species.

    Returns
    -------
    quantity
        Volume mixing ratio at the surface.
    """
    return to_quantity(ds.mr.sel(species=species))[0]


def _scaling_factor(
    initial_amount: pint.Quantity,
    target_amount: pint.Quantity,
) -> float:
    """
    Compute the scaling factor to reach target amount from initial amount.

    Initial and target amount should be scalar and  have compatible dimensions.

    Parameters
    ----------
    initial_amount: quantity
        Initial amount.

    target_amount: quantity
        Target quantity.

    Raises
    ------
    ValueError
        When both initial and target amounts are zero.

    Returns
    -------
    float
        Scaling factor.
    """
    if initial_amount.m == 0.0:
        if target_amount.m == 0.0:
            return 0.0
        else:
            raise ValueError(
                f"Cannot compute scaling factor when initial amount is 0.0 and "
                f"target amount is non-zero (got {target_amount})"
            )
    else:
        return (target_amount / initial_amount).m_as(ureg.dimensionless)


def compute_scaling_factors(
    ds: xr.Dataset, concentration: dict[str, pint.Quantity]
) -> dict[str, float]:
    r"""
    Compute the scaling factors to be applied to the mixing ratio values
    of each species in an atmosphere thermophysical properties data set, so
    that the integrated number/mass density and/or the number/mass density at
    the surface, match given values.

    Parameters
    ----------
    ds : Dataset
        Atmosphere thermophysical properties data set.

    concentration : dict
        Mapping of species (str) and target concentration
        (:class:`~pint.Quantity`).

        If the target concentration has dimensions of inverse square length
        (:math:`[L^{-2}]`), the value is interpreted as a column
        number density for that given species and the scaling factor, :math:`f`,
        is obtained by dividing that column number density,
        :math:`N_{\mathrm{target}}`,
        by the initial column number density,
        :math:`N_{\mathrm{initial}}`:

        .. math::
           f = \frac{N_{\mathrm{target}}}{N_{\mathrm{initial}}}

        If the target concentration has dimensions of mass times inverse square
        length (:math:`[ML^{-2}]`), the value is interpreted as a column (mass)
        density for that species and the scaling factor is obtained by dividing
        that column mass density,
        :math:`\sigma_{\mathrm{target}}`,
        by the initial column mass density,
        :math:`\sigma_{\mathrm{initial}}`:

        .. math::
           f = \frac{\sigma_{\mathrm{target}}}{\sigma_{\mathrm{initial}}}

        If the target concentration has dimensions of inverse cubic length
        (:math:`[L^{-3}]`), the value is interpreted as a number density at the
        surface for that given species and the scaling factor is computed by
        dividing that number density at the surface,
        :math:`n_{\mathrm{surface, target}}`,
        by the initial number density at the surface,
        :math:`n_{\mathrm{surface, initial}}`:

        .. math::
           f = \frac{n_{\mathrm{surface, target}}}{n_{\mathrm{surface, initial}}}

        If the target concentration has dimensions of inverse cubic length
        (:math:`[ML^{-3}]`), the value is interpreted as a mass density at the
        surface for that given species and the scaling factor is computed by
        dividing that mass density at the surface,
        :math:`\sigma_{\mathrm{surface, target}}`,
        by the initial mass density at the surface,
        :math:`\sigma_{\mathrm{surface, initial}}`:

        .. math::
           f = \frac{\sigma_{\mathrm{surface, target}}}{\sigma_{\mathrm{surface, initial}}}

        If the target concentration is dimensionless, the value is
        interpreted as a mixing ratio at the surface for that given species
        and the scaling factor is computed by dividing that mixing ratio at
        the surface,
        :math:`x_{\mathrm{surface, target}}`,
        by the initial mixing ratio at the surface,
        :math:`x_{\mathrm{surface, initial}}`:

        .. math::
           f = \frac{x_{\mathrm{target}}}{x_{\mathrm{initial}}}

    Returns
    -------
    dict
        Mapping of species (str) and scaling factors (float).
    """
    compute_initial_amount = {
        "[length]^-2": column_number_density,
        "[mass] * [length]^-2": column_mass_density,
        "[length]^-3": number_density_at_surface,
        "[mass] * [length]^-3": mass_density_at_surface,
        "": volume_mixing_ratio_at_surface,
    }
    factors = {}
    for species in concentration:
        target_amount = concentration[species]
        initial_amount = None
        for dimensions in list(compute_initial_amount.keys()):
            if target_amount.check(dimensions):
                initial_amount = compute_initial_amount[dimensions](
                    ds=ds, species=species
                )
        if initial_amount is None:
            raise ValueError(
                f"Invalid dimension for {species} concentration: {target_amount.units}."
                f"Expected dimension: [length]^-2, [mass] * [length]^-2, "
                f"[length]^-3, [mass] * [length]^-2 or no dimension."
            )
        factors[species] = _scaling_factor(
            initial_amount=initial_amount,
            target_amount=target_amount,
        )

    return factors


def human_readable(items: list[str]) -> str:
    """
    Transforms a list into readable human text.

    Example: ``["a", "b", "c"]`` -> ``"a, b and c"``

    Parameters
    ----------
    elements : list
        List.

    Returns
    -------
    str
        Human readable text.
    """
    x = f"{items[0]}"
    for s in items[1:-1]:
        x += f", {s}"
    if len(items) > 1:
        x += f" and {items[-1]}"
    return x


def rescale_concentration(
    ds: xr.Dataset, factors: dict[str, float], inplace: bool = False
) -> xr.Dataset | None:
    """
    Rescale mixing ratios in an atmosphere thermophysical properties data
    set by given factors for each species.

    Parameters
    ----------
    ds : Dataset
        Initial atmosphere thermophysical properties data set.

    factors : dict
        Mapping of species (str) and scaling factors (float).

    inplace : bool
        If ``True``, the atmosphere thermophysical properties data set object
        is modified.
        Else, a new atmosphere thermophysical properties data set object is
        returned.

    Returns
    -------
    Dataset
        Rescaled atmosphere thermophysical properties data set.
    """
    if not inplace:
        ds = ds.copy(deep=True)

    mr = ds.mr.values
    for i, species in enumerate(ds.species.values):
        if species in factors:
            mr[i] *= factors[species]

    machine_epsilon = np.finfo(float).eps
    if any(mr.sum(axis=0) > 1.0 + machine_epsilon):
        raise ValueError(
            f"Cannot rescale concentration with these factors "
            f"({factors}) because the sum of mixing ratios would "
            f"be larger than 1: {mr.sum(axis=0)}"
        )

    species = list(factors.keys())
    ds.attrs["history"] += (
        f"\n"
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
        f"Concentration-rescaled for {human_readable(species)} - "
        f"eradiate.thermoprops.util.rescale_concentration"
    )

    return ds


@ureg.wraps(ret=None, args=(None, "km", None, None), strict=False)
def interpolate(
    ds: xr.Dataset,
    z_level: np.ndarray | pint.Quantity,
    method: str = "linear",
    conserve_columns: bool = False,
) -> xr.Dataset:
    """
    Interpolates an atmosphere thermophysical properties data set onto a
    new level altitude mesh.

    Parameters
    ----------
    ds : Dataset
        Initial atmosphere thermophysical properties data set.

    z_level : ndarray
        Level altitude mesh [km].

    method : str
        The method used to interpolate (same for all data variables).
        Choose from ``"linear"``, ``"nearest"``, ``"zero"``, ``"slinear"``, \
        ``"quadratic"``, and ``"cubic"``.

        Default: ``"linear"``.

    conserve_columns : bool
        If ``True``, multiply the number densities in the atmosphere
        thermophysical properties data set so that the column number densities
        in the initial and interpolated atmosphere thermophysical properties
        data set are in same.

        Default: ``False``.

    Returns
    -------
    Dataset
        Interpolated atmosphere thermophysical properties data set.

    Notes
    -----
    Returns a new atmosphere thermophysical properties data set object.
    """
    z_level = ureg.Quantity(z_level, "km")
    z_layer = (z_level[1:] + z_level[:-1]) / 2.0
    interpolated = ds.interp(
        z_layer=z_layer.m_as(ds.z_layer.units),
        method=method,
        kwargs=dict(fill_value="extrapolate"),
    )
    z_level_attrs = ds.z_level.attrs
    interpolated.update(
        dict(z_level=("z_level", z_level.m_as(ds.z_level.units), z_level_attrs))
    )

    if conserve_columns:
        initial_amounts = {s: column_number_density(ds, s) for s in ds.species.values}
        factors = compute_scaling_factors(
            ds=interpolated, concentration=initial_amounts
        )
        interpolated = rescale_concentration(
            ds=interpolated, factors=factors, inplace=True
        )

    return interpolated


@ureg.wraps(ret="Pa", args="K", strict=False)
def water_vapor_saturation_pressure(t: float) -> pint.Quantity:
    """
    Computes the water vapor saturation pressure over liquid water or ice, at
    the given temperature.

    Parameters
    ----------
    t : float
        Temperature [K].

    Returns
    -------
    quantity
        Water vapor saturation pressure.

    Notes
    -----
    Valid for pressures larger than the triple point pressure (~611 Pa).
    """
    try:
        import iapws
    except ModuleNotFoundError:
        warnings.warn(
            "To use the collision detection feature, you must install IAPWS.\n"
            "See instructions on https://iapws.readthedocs.io/en/latest/modules.html#installation."
        )
        raise

    if t >= 273.15:  # water is liquid
        p = ureg.Quantity(iapws.iapws97._PSat_T(t), "MPa")
    else:  # water is solid
        p = ureg.Quantity(iapws._iapws._Sublimation_Pressure(t), "MPa")
    return p.m_as("Pa")


@ureg.wraps(ret=ureg.dimensionless, args=("Pa", "K"), strict=False)
def equilibrium_water_vapor_fraction(p: float, t: float) -> pint.Quantity:
    """
    Computes the water vapor volume fraction at equilibrium, i.e., when the
    rate of condensation of water vapor equals the rate of evaporation of
    liquid water or ice, depending on the temperature.

    The water vapor volume fraction :math:`x_w` is computed with:

    .. math::
       x_w(p,T) = \\frac{p_w(T)}{p}

    where

    * :math:`p` is the pressure,
    * :math:`T` is the temperature and
    * :math:`p_w` is the water vapor saturation pressure at the given
      temperature.

    This water vapor volume fraction corresponds to a relative humidity of 100%.

    Parameters
    ----------
    p : float
        Pressure [Pa].

    t : float
        Temperature [K].

    Returns
    -------
    quantity
        Water vapor volume fraction.

    Raises
    ------
    ValueError
        If equilibrium cannot be reached in the pressure and temperature
        conditions.

    Notes
    -----
    For some values of the pressure and temperature, the equilibrium does not
    exist. An exception is raised in those cases.
    """
    p_water_vapor = water_vapor_saturation_pressure(t).magnitude
    if p_water_vapor <= p:
        return p_water_vapor / p
    else:
        raise ValueError(
            f"Equilibrium cannot be reached in these conditions (p = "
            f"{round(p, 2)} Pa, t = {round(t, 2)} K)"
        )


def make_profile_regular(profile: xr.Dataset, atol: float) -> xr.Dataset:
    """
    Converts the atmosphere thermophysical properties data set with an
    irregular altitude mesh to a profile defined over a regular altitude mesh.

    Parameters
    ----------
    profile : Dataset
        Original atmosphere thermophysical properties data set, defined over
        an irregular altitude mesh.

    atol : float
        Absolute tolerance used in the conversion of the irregular altitude
        mesh to a regular altitude mesh.

    Returns
    -------
    Dataset
        Converted atmosphere thermophysical properties data set, defined over
        a regular altitude mesh.
    """

    # compute the regular altitude nodes mesh
    regular_z_level = _to_regular(mesh=profile.z_level.values, atol=atol)

    # compute corresponding altitude centers mesh
    regular_z_layer = (regular_z_level[:-1] + regular_z_level[1:]) / 2.0

    # compute the atmospheric variables with the regular altitude mesh
    n_z = len(regular_z_layer)
    p = np.zeros(n_z)
    t = np.zeros(n_z)
    n = np.zeros(n_z)
    n_species = profile.species.size
    mr = np.zeros((n_species, n_z))
    layer = 0
    for i, z in enumerate(regular_z_layer):
        # when altitude is larger than current layer's upper bound, jump to
        # the next layer
        if z >= profile.z_level.values[layer + 1]:
            layer += 1

        p[i] = profile.p.values[layer]
        t[i] = profile.t.values[layer]
        n[i] = profile.n.values[layer]
        mr[:, i] = profile.mr.values[:, layer]

    species = profile["species"].values

    # copy attributes
    attrs = profile.attrs

    # update history
    new_line = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - made "
        f"profile altitude mesh regular - "
        f"eradiate.thermoprops.util.make_profile_regular "
    )
    attrs["history"] += f"\n{new_line}"

    dataset = xr.Dataset(
        data_vars=dict(
            p=("z_layer", p),
            t=("z_layer", t),
            n=("z_layer", n),
            mr=(("species", "z_layer"), mr),
        ),
        coords={
            "z_layer": ("z_layer", regular_z_layer),
            "z_level": ("z_level", regular_z_level),
            "species": ("species", species),
        },
        attrs=attrs,
    )

    return dataset


def _to_regular(mesh: np.ndarray, atol: float) -> np.ndarray:
    """
    Converts an irregular altitude mesh into a regular altitude mesh.

    Parameters
    ----------
    mesh : ndarray
        Irregular altitude mesh with values sorted in increasing order.

    atol : float
        Absolute tolerance used in the conversion.

    Returns
    -------
    ndarray
        Regular altitude mesh.

    Warnings
    --------
    The algorithm is not optimized to find the approximating regular mesh with
    the smallest number of layers. Depending on the value of ``atol``, the
    resulting mesh size can be large.

    Notes
    -----
    The bound altitudes in the irregular altitude mesh remain the same in
        the output regular altitude mesh. Only the intermediate node altitudes
        are modified.
    """

    n, _ = _find_regular_params_gcd(mesh, atol)
    return np.linspace(start=mesh[0], stop=mesh[-1], num=n)


def _find_regular_params_gcd(
    mesh: np.ndarray, unit_number: float = 1.0
) -> tuple[int, float]:
    """
    Finds the parameters (number of cells, constant cell width) of the
    regular mesh that approximates the irregular input mesh.

    The algorithm finds the greatest common divisor (GCD) of all cells widths
    in the integer representation specified by the parameter ``unit_number``.
    This GCD is used to define the constant cells width of the approximating
    regular mesh.

    Parameters
    ----------
    mesh : ndarray
        1-D array with floating point values.
        Values must be sorted by increasing order.

    unit_number : float, default: 1.0
        Defines the unit used to convert the floating point numbers to integer.
        numbers.

    Returns
    -------
    int
        Number of points in the regular mesh.

    float
        Value of the constant cells width.

    Warnings
    --------
    There are no safeguards regarding how large the number of cells in the
    regular mesh can be. Use the parameter ``unit_number`` with caution.
    """

    # Convert float cell widths to integer cell widths
    eps = np.finfo(float).eps
    if unit_number >= eps:
        mesh = np.divide(mesh, unit_number).astype(int)
    else:
        raise ValueError(
            f"Parameter unit_number ({unit_number}) must be "
            f"larger than machine epsilon ({eps})."
        )
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
#     r"""
#     Finds the number of cells and constant cell width of the regular 1-D
#     mesh that approximates a 1-D irregular mesh the best.
#
#     Parameters
#     ----------
#     mesh : ndarray
#         Irregular 1-D mesh. Values must be sorted in increasing order.
#
#     rtol : float
#         Relative tolerance on the cells widths. This parameter controls the
#         accuracy of the approximation.
#         The parameters of the approximating regular mesh are computed so that
#         for each layer of the irregular mesh, the corresponding layer or layers
#         in the regular mesh has a width or have a total width that is not larger
#         than ``rtol`` times the width of the cell in the irregular mesh.
#
#         Default: 1e-3
#
#     n_cells_max : float
#         Maximum number of cells in the regular mesh. This parameter controls the
#         size of the resulting regular mesh.
#
#         Default: 10000
#
#     Returns
#     -------
#     int
#         Number of cells in the approximating regular mesh.
#     float
#         Constant cells width in the approximating regular mesh.
#     """
#
#     raise NotImplemented
# TODO: implement this function
