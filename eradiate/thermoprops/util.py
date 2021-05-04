"""
Utility functions to manipulate atmosphere thermophysical properties data 
sets.
"""

from datetime import datetime

import iapws
import numpy as np
import scipy.constants
import xarray as xr

import eradiate.data as data
import eradiate.mesh as mesh

from . import profile_dataset_spec
from ..units import unit_registry as ureg
from ..units import to_quantity

ATOMIC_MASS_CONSTANT = ureg.Quantity(
    *scipy.constants.physical_constants["atomic mass constant"][:-1]
)


def compute_column_number_density(ds, species):
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

    Parameter ``ds`` (:class:`xarray.Dataset`):
        Atmosphere thermophysical properties data set.

    Parameter ``species`` (str):
        Species.

    Returns → :class:`~pint.Quantity`:
        Column number density.
    """
    mr = ds.mr.sel(species=species).values
    n = to_quantity(ds.n)
    n_species = mr * n
    z_level = to_quantity(ds.z_level)
    dz = z_level[1:] - z_level[:-1]
    return (n_species * dz).sum()


def compute_column_mass_density(ds, species):
    """Computes the column (mass) density of a given species in an atmospheric
    profile.
    The column mass density is computed according to the formula:

    .. math::
      \\sigma = N * m_u

    where

    :math:`\\sigma` denotes the column mass density,
    :math:`N` is the column number density,
    :math:`m_u` is the atomic mass constant.

    Parameter ``ds`` (:class:`xarray.Dataset`):
        Atmosphere thermophysical properties data set.

    Parameter ``species`` (str):
        Species.

    Returns → :class:`ureg.Quantity`:
        Column mass density.
    """
    molecular_mass = data.open(category="chemistry", id="molecular_masses")
    m = molecular_mass.m.sel(s=species).values * ATOMIC_MASS_CONSTANT
    return m * compute_column_number_density(ds=ds, species=species)


def compute_number_density_at_surface(ds, species):
    """Compute the number density at the surface of a given species in an
    atmospheric profile.

    Parameter ``ds`` (:class:`xarray.Dataset`):
        Atmosphere thermophysical properties data set.

    Parameter ``species`` (str):
        Species.

    Returns → :class:`~pint.Quantity`:
        Number density at the surface.
    """
    surface_mr = to_quantity(ds.mr.sel(species=species))[0]
    surface_n = to_quantity(ds.n)[0]
    return surface_mr * surface_n


def compute_mass_density_at_surface(ds, species):
    """Compute the mass density at the surface of a given species in an
    atmospheric profile.

    Parameter ``ds`` (:class:`xarray.Dataset`):
        Atmosphere thermophysical properties data set.

    Parameter ``species`` (str):
        Species.

    Returns → :class:`ureg.Quantity`:
        Mass density at the surface.
    """
    molecular_mass = data.open(category="chemistry", id="molecular_masses")
    m = to_quantity(molecular_mass.m.sel(s=species)) * ATOMIC_MASS_CONSTANT
    return m * compute_number_density_at_surface(ds=ds, species=species)


def compute_scaling_factors(ds, concentration):
    """Compute the scaling factors to be applied to the mixing ratio values
    of each species in an atmosphere thermophysical properties data set, so
    that the integrated number/mass density and/or the number/mass density at
    the surface, match given values.

    Parameter ``ds`` (:class:`xarray.Dataset`):
        Atmosphere thermophysical properties data set.

    Parameter ``concentration`` (dict):
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
            f = \\frac{N_{\mathrm{target}}}{N_{\mathrm{initial}}}

        If the target concentration has dimensions of mass times inverse square
        length (:math:`[ML^{-2}]`), the value is interpreted as a column (mass)
        density for that species and the scaling factor is obtained by dividing
        that column mass density,
        :math:`\\sigma_{\mathrm{target}}`,
        by the initial column mass density,
        :math:`\\sigma_{\mathrm{initial}}`:

        .. math::
            f = \\frac{\\sigma_{\mathrm{target}}}{\\sigma_{\mathrm{initial}}}

        If the target concentration has dimensions of inverse cubic length
        (:math:`[L^{-3}]`), the value is interpreted as a number density at the
        surface for that given species and the scaling factor is computed by
        dividing that number density at the surface,
        :math:`n_{\mathrm{surface, target}}`,
        by the initial number density at the surface,
        :math:`n_{\mathrm{surface, initial}}`:

        .. math::
            f = \\frac{n_{\mathrm{surface, target}}}{n_{\mathrm{surface, initial}}}

        If the target concentration has dimensions of inverse cubic length
        (:math:`[ML^{-3}]`), the value is interpreted as a mass density at the
        surface for that given species and the scaling factor is computed by
        dividing that mass density at the surface,
        :math:`\\sigma_{\mathrm{surface, target}}`,
        by the initial mass density at the surface,
        :math:`\\sigma_{\mathrm{surface, initial}}`:

        .. math::
            f = \\frac{\\sigma_{\mathrm{surface, target}}}{\\sigma_{\mathrm{surface, initial}}}

        If the target concentration is dimensionless, the value is
        interpreted as a mixing ratio at the surface for that given species
        and the scaling factor is computed by dividing that mixing ratio at
        the surface,
        :math:`x_{\mathrm{surface, target}}`,
        by the initial mixing ratio at the surface,
        :math:`x_{\mathrm{surface, initial}}`:

        .. math::
            f = \\frac{x_{\mathrm{target}}}{x_{\mathrm{initial}}}

    Returns → dict:
        Mapping of species (str) and scaling factors (float).
    """
    factors = {}
    for species in concentration:
        amount = concentration[species]
        if amount.check("[length]^-2"):  # column number density
            initial_amount = compute_column_number_density(ds=ds, species=species)
            factor = amount.to("m^-2") / initial_amount.to("m^-2")
        elif amount.check("[mass] * [length]^-2"):
            initial_amount = compute_column_mass_density(ds=ds, species=species)
            factor = amount.to("kg/m^2") / initial_amount.to("kg/m^2")
        elif amount.check("[length]^-3"):  # number density at the surface
            initial_amount = compute_number_density_at_surface(ds=ds, species=species)
            factor = amount.to("m^-3") / initial_amount.to("m^-3")
        elif amount.check("[mass] * [length]^-2"):
            initial_amount = compute_mass_density_at_surface(ds=ds, species=species)
            factor = amount.to("km/m^3") / initial_amount.to("kg/m^3")
        elif amount.check(""):  # mixing ratio at the surface
            surface_mr_species = amount
            initial_surface_mr_species = to_quantity(ds.mr.sel(species=species))[0]
            factor = surface_mr_species / initial_surface_mr_species
        else:
            raise ValueError(
                f"Invalid dimension for {species} concentration:" f" {amount.units}."
            )
        factors[species] = factor.magnitude

    return factors


def human_readable(items):
    """
    Transforms a list into readable human text.

    Example: ``["a", "b", "c"]`` -> ``"a, b and c"``

    Parameter ``elements`` (list):
        List.

    Returns → str:
        Human readable text.
    """
    x = f"{items[0]}"
    for s in items[1:-1]:
        x += f", {s}"
    if len(items) > 1:
        x += f" and {items[-1]}"
    return x


def rescale_concentration(ds, factors, inplace=False):
    """
    Rescale mixing ratios in an atmosphere thermophysical properties data
    set by given factors for each species.

    Parameter ``ds`` (:class:`xarray.Dataset`):
        Initial atmosphere thermophysical properties data set.

    Parameter ``factors`` (dict):
        Mapping of species (str) and scaling factors (float).

        Mapping of the species and corresponding scaling factors.

    Parameter ``inplace`` (bool):
        If ``True``, the atmosphere thermophysical properties data set object
        is modified.
        Else, a new atmosphere thermophysical properties data set object is
        returned.

    Returns → :class:`~xarray.Dataset`:
        Rescaled atmosphere thermophysical properties data set.
    """
    if not inplace:
        ds = ds.copy(deep=True)

    mr = ds.mr.values
    for i, species in enumerate(ds.species.values):
        if species in factors:
            mr[i] *= factors[species]

    if any(mr.sum(axis=0) > 1.0):
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
def interpolate(ds, z_level, method="linear", conserve_columns=False):
    """
    Interpolates an atmosphere thermophysical properties data set onto a
    new level altitude mesh.

    .. note::
        Returns a new atmosphere thermophysical properties data set object.

    Parameter ``ds`` (:class:`xarray.Dataset`):
        Initial atmosphere thermophysical properties data set.

    Parameter ``z_level`` (:class:`numpy.ndarray`):
        Level altitude mesh [km].

    Parameter ``method`` (str):
        The method used to interpolate (same for all data variables).
        Choose from ``"linear"``, ``"nearest"``, ``"zero"``, ``"slinear"``, ``"quadratic"``, and ``"cubic"``.

        Default: ``"linear"``.

    Parameter ``conserve_columns`` (bool):
        If ``True``, multiply the number densities in the atmosphere
        thermophysical properties data set so that the column number densities
        in the initial and interpolated atmosphere thermophysical properties
        data set are in same.

        Default: ``False``.

    Returns → :class:`xarray.Dataset`:
        Interpolated atmosphere thermophysical properties data set
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
        initial_amounts = {
            s: compute_column_number_density(ds, s) for s in ds.species.values
        }
        factors = compute_scaling_factors(
            ds=interpolated, concentration=initial_amounts
        )
        interpolated = rescale_concentration(
            ds=interpolated, factors=factors, inplace=True
        )

    return interpolated


@ureg.wraps(ret="Pa", args="K", strict=False)
def water_vapor_saturation_pressure(t):
    """Computes the water vapor saturation pressure over liquid water or ice,
    at the given temperature.

    .. note::
        Valid for pressures larger than the triple point pressure (~611 Pa).

    Parameter ``t`` (float):
        Temperature [K].

    Returns → :class:`pint.Quantity`:
        Water vapor saturation pressure.
    """
    if t >= 273.15:  # water is liquid
        p = ureg.Quantity(iapws.iapws97._PSat_T(t), "MPa")
    else:  # water is solid
        p = ureg.Quantity(iapws._iapws._Sublimation_Pressure(t), "MPa")
    return p.m_as("Pa")


@ureg.wraps(ret=ureg.dimensionless, args=("Pa", "K"), strict=False)
def equilibrium_water_vapor_fraction(p, t):
    """Computes the water vapor volume fraction at equilibrium, i.e., when the
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

    .. note::
       For some values of the pressure and temperature, the equilibrium does
       not exist.
       An exception is raised in those cases.

    Parameter ``p`` (float):
        Pressure [Pa].

    Parameter ``t`` (float):
        Temperature [K].

    Returns → :class:`pint.Quantity`:
        Water vapor volume fraction.

    Raises → ValueError:
        If equilibrium cannot be reached in the pressure and temperature
        conditions.
    """
    p_water_vapor = water_vapor_saturation_pressure(t).magnitude
    if p_water_vapor <= p:
        return p_water_vapor / p
    else:
        raise ValueError(
            f"Equilibrium cannot be reached in these conditions (p = "
            f"{round(p, 2)} Pa, t = {round(t, 2)} K)"
        )


def make_profile_regular(profile, atol):
    """
    Converts the atmosphere thermophysical properties data set with an
    irregular altitude mesh to a profile defined over a regular altitude mesh.

    Parameter ``profile`` (:class:`~xarray.Dataset`):
        Original atmosphere thermophysical properties data set, defined over
        an irregular altitude mesh.

    Parameter ``atol`` (float):
        Absolute tolerance used in the conversion of the irregular altitude
        mesh to a regular altitude mesh.

    Returns → :class:`~xarray.Dataset`:
        Converted atmosphere thermophysical properties data set, defined over
        a regular altitude mesh.
    """
    profile.ert.validate_metadata(profile_dataset_spec)

    # compute the regular altitude nodes mesh
    regular_z_level = mesh.to_regular(x=profile.z_level.values, atol=atol)

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
    dataset.ert.normalize_metadata(profile_dataset_spec)

    return dataset
