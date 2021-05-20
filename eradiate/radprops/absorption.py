"""Functions to compute monochromatic absorption.
"""
import numpy as np
import xarray as xr
from scipy.constants import physical_constants

from .._units import to_quantity
from .._units import unit_registry as ureg

_BOLTZMANN = ureg.Quantity(*physical_constants["Boltzmann constant"][:2])


@ureg.wraps(ret="m^-1", args=(None, "nm", "Pa", "K", "m^-3", None, None), strict=False)
def compute_sigma_a(
    ds, wl=550.0, p=101325.0, t=288.15, n=None, fill_values=None, methods=None
):
    """
    Computes the monochromatic absorption coefficient at given wavelength,
    pressure and temperature values.

    The absorption coefficient is computed according to the formula:

    .. math::
        k_{a\\lambda} = n \\, \\sigma_{a\\lambda} (p, T)

    where

    * :math:`k_{a\\lambda}` is the absorption coefficient [:math:`L^{-1}`],
    * :math:`\\lambda` is the wavelength [:math:`L`],
    * :math:`n` is the number density [:math:`L^{-3}`],
    * :math:`\\sigma_a` is the absorption cross section [:math:`L^2`],
    * :math:`p` is the pressure [:math:`ML^{-1}T^{-2}`] and
    * :math:`t` is the temperature [:math:`\\Theta`].

    .. warning::
       The values of the absorption cross section at the desired wavelength,
       pressure and temperature values,
       :math:`\\sigma_{a\\lambda} (p, T)`,
       are obtained by interpolating the input absorption cross section data
       set along the corresponding dimensions.

    Parameter ``ds`` (:class:`~xarray.Dataset`):
        Absorption cross section data set.

    Parameter ``wl`` (float):
        Wavelength value [nm].

    Parameter ``p`` (float or array):
        Pressure [Pa].

        .. note::
           If ``p``, ``t`` and ``n`` are arrays, their length must be the same.

    Parameter ``t`` (float or array):
        Temperature [K].

        .. note::
           If the coordinate ``t`` is not in the input dataset ``ds``, the
           interpolation on temperature is not performed.

    Parameter ``n`` (float or array):
        Number density [m^-3].

        .. note::
           If ``n`` is ``None``, the values of ``t`` and ``p`` are then used
           only to compute the corresponding number density.

    Parameter ``fill_values`` (dict):
        Mapping of coordinates (in ``["w", "pt"]``) and fill values (either
        ``None`` or float).
        If not ``None``, out of bounds values are assigned the fill value
        during interpolation along the wavelength or pressure and temperature
        coordinates.
        If ``None``, out of bounds values will trigger the raise of a
        ``ValueError``.
        Only one fill value can be provided for both pressure and temperature
        coordinates.

    Parameter ``methods`` (dict):
        Mapping of coordinates (in ``["w", "pt"]``) and interpolation methods.
        Default interpolation method is linear.
        Only one interpolation method can be specified for both pressure
        and temperature coordinates.

    Returns → :class:`~pint.Quantity`:
        Absorption coefficient values.

    Raises → ``ValueError``:
        When wavelength, pressure, or temperature values are out of the range
        of the data set and the corresponding fill value in ``fill_values`` is
        ``None``.
    """
    if fill_values is None:
        fill_values = dict(w=None, pt=None)

    if methods is None:
        methods = dict(w="linear", pt="linear")

    for name in ["w", "pt"]:
        if name not in fill_values:
            fill_values[name] = None
        if name not in methods:
            methods[name] = "linear"

    # Interpolate along wavenumber dimension
    xsw = ds.interp(
        w=1e7 / wl,  # wavenumber in cm^-1
        method=methods["w"],
        kwargs=dict(
            bounds_error=(fill_values["w"] is None),
            fill_value=fill_values["w"],
        ),
    )

    # If the data set includes a temperature coordinate, we interpolate along
    # both pressure and temperature dimensions.
    # Else, we interpolate only along the pressure dimension.
    p_values = np.array([p]) if isinstance(p, float) else p
    pz = xr.DataArray(p_values, dims="pt")
    if "t" in ds.coords:
        t_values = np.array([t] * len(p_values)) if isinstance(t, float) else t
        tz = xr.DataArray(t_values, dims="pt")
        interpolated = xsw.interp(
            p=pz,
            t=tz,
            method=methods["pt"],
            kwargs=dict(
                bounds_error=(fill_values["pt"] is None),
                fill_value=fill_values["pt"],
            ),
        )
    else:
        interpolated = xsw.interp(
            p=pz,
            method=methods["pt"],
            kwargs=dict(
                bounds_error=(fill_values["pt"] is None),
                fill_value=fill_values["pt"],
            ),
        )

    xs = to_quantity(interpolated.xs)

    # If 'n' is None, we compute it using the ideal gas state equation.
    if n is None:
        k = _BOLTZMANN.magnitude
        n = p / (k * t)  # ideal gas state equation
    n = ureg.Quantity(value=n, units="m^-3")

    return (n * xs).m_as("m^-1")
