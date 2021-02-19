"""Utility functions to manipulate atmospheric profiles. """

from datetime import datetime

import numpy as np
import xarray as xr

from ..util.units import ureg


def column_number_density(ds, species):
    """Computes the column number density of a given species in an atmospheric
    profile.

    The column number density is computed according to the formula:

    .. math::
      N = \\sum_{i=0}^{L-1} n_i \\, (z_{i+1} - z_i)

    where
    :math: denotes the column number density,
    :math:`z_i` are the level altitudes and
    :math:`n_i` are the number densities of that given species inside the
    :math:`L` atmospheric layers.

    Parameter ``ds`` (:class:`xarray.Dataset`):
        Atmospheric profile.

    Parameter ``species`` (str):
        Species.

    Returns → :class:`ureg.Quantity`:
        Column number density [m^-2].
    """
    n = ureg.Quantity(
        value=ds.n.sel(species=species).values,
        units=ds.n.units)
    dz = ureg.Quantity(
        value=ds.z_level.values[1:] - ds.z_level.values[:-1],
        units=ds.z_level.units)
    column = (n * dz).sum().to("m^-2")
    return column


def compute_scaling_factors(ds, column_amounts, surface_amounts):
    """Compute the scaling factors to be applied to the number density values
    of each species in an atmospheric profile, so that the integrated number
    density and/or the surface number density, match given values.

    Parameter ``ds`` (:class:`xarray.Dataset`):
        Initial atmospheric profile.

    Parameter ``column_amounts`` (dict):
        Column number density values to be matched.
        Mapping of the species to the column amounts:
        ``str`` -> ``ureg.Quantity``.
        Column amounts must have the dimensions [length^-2].

    Parameter ``surface_amounts`` (dict):
        Number density values at the surface to be matched.
        Mapping of the species to the surface amounts:
        ``str`` -> ``ureg.Quantity``.
        Surface amounts must either have the dimensions [length^-3] or be
        dimensionless, in which case they are interpreted as volume fractions.

    Returns → dict:
        Scaling factors for each applicable species.
    """
    factors = {}

    for species in column_amounts:
        amount = column_amounts[species].to("m^-2").magnitude
        initial_amount = ds.ert.column(species=species).magnitude
        factors[species] = amount / initial_amount

    for species in surface_amounts:
        if species in column_amounts:
            raise ValueError("a species cannot be both in 'column_amounts' and "
                             "'surface_amounts'")

        quantity = surface_amounts[species]
        if quantity.units == ureg.dimensionless:  # interpret as volume fraction
            amount = quantity.magnitude * ds.n_tot.values[0]
        else:
            amount = quantity.to("m^-3").magnitude

        initial_amount = ds.n.sel(species=species).values[0]
        factors[species] = amount / initial_amount

    return factors


def rescale_number_density(ds, factors):
    """Multiply the number density values by a given factor.

    .. note::
        This will also update the total number density data variable in the
        dataset.

    Parameter ``ds`` (:class:`xarray.Dataset`):
        Initial atmospheric profile.

    Parameter ``factors`` (dict):
        Scaling factors.
        Mapping of the species to its corresponding scaling factor:
        ``str`` -> ``float``

    Returns ``xarray.Dataset``:
        Rescaled atmospheric profile.
    """
    # rescale individual number densities
    n = ds.n.values
    for i, species in enumerate(ds.species.values):
        n[i] *= factors[species]
    n_var = (("species", "z_layer"), n, dict(
        standard_name="number_density",
        long_name="number density",
        units="m^-3"))

    # update total number density
    n_tot = n.sum(axis=0)
    n_tot_var = ("z_layer", n_tot, dict(
        standard_name="total_number_density",
        long_name= "total number density",
        units="m^-3"))

    return ds.assign_coords(n=n_var).assign_coords(n_tot=n_tot_var)


@ureg.wraps(ret=None, args=(None, "m", None, None), strict=False)
def interpolate(ds, z_level, method="linear", conserve_columns=False):
    """Interpolates an atmospheric profile onto a new level altitude mesh.

    Parameter ``ds`` (:class:`xarray.Dataset`):
        Initial atmospheric profile.

    Parameter ``z_level`` (array):
        Level altitude mesh [m].

    Parameter ``method`` (str):
        The method used to interpolate. Choose from ``"linear"``,
        ``"nearest"``, ``"zero"``, ``"slinear"``, ``"quadratic"``, ``"cubic"``.

    Parameter ``conserve_columns`` (bool):
        Rescale the number densities in the atmospheric profile so that the
        column number densities in the initial and interpolated atmospheric
        profile are in the same.

        Default: ``False``.

    Returns → :class:`xarray.Dataset`:
        Interpolated atmospheric profile
    """
    z_layer = (z_level[1:] + z_level[:-1]) / 2.
    interpolated = ds.interp(z_layer=z_layer,
                             method=method,
                             kwargs=dict(fill_value="extrapolate"))
    projection = interpolated.assign_coords(z_level=(
        "z_level", z_level, {
            "standard_name": "level_altitude",
            "long_name": "level altitude",
            "units": "m"}))

    if conserve_columns:
        factors = compute_scaling_factors(ds=projection,
                                          column_amounts=ds.ert.columns)
        projection = rescale_number_density(ds=projection,
                                            factors=factors)

    return projection


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
