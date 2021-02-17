"""Functions to compute monochromatic absorption.

.. admonition:: Absorption cross section data set format:

    The data structure is a :class:`~xarray.Dataset` with specific data
    variables, dimensions and data coordinates.

    Data variables:

    - ``xs``: absorption cross section [cm^2]

    If the absorber is a mixture, an additional data variable is required:

    - ``mr``: mixing ratio []

    The dimensions of ``xs`` should be one of the following:

    - ``w``
    - (``w``, ``p``)
    - (``w``, ``t``)
    - (``w``, ``p``, ``t``)

    What dimensions are included indicate the relationship between the current dataset to other datasets and allow to combine these datasets together.
    Datasets with identical dimensions can be combined together along those dimensions, provided the absorber is the same.
    For example, a dataset with the dimensions (``w``, ``p``) for the ``us76`` absorber can be combined with other ``us76`` datasets with dimensions (``w``, ``p``), along the ``p`` dimension.

    The dimension of ``mr`` is ``m``.

    Data coordinates:

    - ``m``: absorbing molecule(s) []
    - ``w``: wavenumber [cm^-1]
    - ``p``: pressure [Pa]
    - ``t``: temperature [K]

    All these data coordinates are required even, if the corresponding dimensions do not not appear in the ``xs`` and ``mr``.
    In the latter case however, the data coordinates must be renamed by adding a ``c`` to the data coordinate name.
    This is to indicate that the data coordinate is a non-dimension data coordinate.
    See the difference between dimension coordinate and non-dimension coordinate at http://xarray.pydata.org/en/stable/data-structures.html

    Attributes:

    - ``convention``
    - ``title``
    - ``history``
    - ``source``
    - ``references``
    - ``absorber``: name of the absorbing molecule/mixture

    The meaning of the first 5 attributes is explained in the C.F. 1.8
    convention (http://cfconventions.org/).
"""
import numpy as np
from scipy.constants import physical_constants

from .._units import unit_registry as ureg

_BOLTZMANN = ureg.Quantity(*physical_constants['Boltzmann constant'][:2])


def _check_within_range(name, value, lower, upper):
    value = np.array(value)
    if (value < lower).any() or (value > upper).any():
        raise ValueError(f"{name} value(s) ({value}) is outside of range "
                         f"({lower}, {upper})")


@ureg.wraps(ret="m^-1", args=(None, "nm", "Pa", "K", "m^-3", None), strict=False)
def compute_sigma_a(ds, wl=550., p=101325., t=288.15, n=None,
                    p_fill_value=None):
    """Computes the monochromatic absorption coefficient at given wavelength,
    pressure and temperature values, according to the formula:

    .. math::
        k_{a\\lambda} = n \\, \\sigma_{a\\lambda} (p, T)

    where
    :math:`k_{a\\lambda}` is the absorption coefficient,
    :math:`\\lambda` is the wavelength,
    :math:`n` is the number density,
    :math:`\\sigma_a` is the absorption cross section,
    :math:`p` is the pressure and
    :math:`t` is the temperature.

    .. note::
        if the coordinate ``t`` is not in the dataset ``ds``, the interpolation
        on temperature is not performed.
        If ``n`` is ``None``, the value of the parameter ``t`` is then used
        only to compute the corresponding number density.
        Else, the value of ``t`` is simply ignored.

    .. note::
        If at least two of ``p``, ``t`` and ``n`` are arrays, either their
        length are the same, or one of them has a length of 1.

    Parameter ``ds`` (:class:`~xarray.Dataset`):
        Absorption cross section dataset.

    Parameter ``wl`` (float):
        Wavelength value [nm].

        Default: 550 nm.

    Parameter ``p`` (float or array):
        Pressure [Pa].

        Default: 101325 Pa.

    Parameter ``t`` (float or array):
        Temperature [K].

        Default: 288.15 K.

    Parameter ``n`` (float or array):
        Number density [m^-3].

        Default: ``None``.

    Parameter ``p_fill_value`` (float):
        If not ``None``, out of bounds values are assigned ``p_fill_value``
        during interpolation on pressure.

        Default: ``None``

    Returns → float or array:
        Absorption coefficient [m^-1].
    """
    # check wavelength is within range of dataset
    wn_max = ureg.Quantity(value=ds.w.values.max(), units=ds.w.units)
    wn_min = ureg.Quantity(value=ds.w.values.min(), units=ds.w.units)
    _check_within_range(name="wavelength",
                        value=wl,
                        lower=(1 / wn_max).to("nm").magnitude,
                        upper=(1 / wn_min).to("nm").magnitude)

    # compute wavenumber
    wn = (1.0 / ureg.Quantity(wl, "nm")).to(ds.w.units).magnitude

    # interpolate in wavenumber
    xsw = ds.xs.interp(w=wn)

    # interpolate in pressure
    if p_fill_value is None:
        _check_within_range(name="pressure",
                            value=p,
                            lower=xsw.p.values.min(),
                            upper=xsw.p.values.max())
        xsp = xsw.interp(p=p)
    else:
        xsp = xsw.interp(p=p, kwargs=dict(fill_value=p_fill_value))

    # interpolate in temperature
    if "t" in ds.coords:
        _check_within_range(name="temperature",
                            value=t,
                            lower=xsw.t.values.min(),
                            upper=xsw.t.values.max())
        xst = xsp.interp(t=t)
    else:
        xst = xsp

    xs = ureg.Quantity(xst.values, xst.units)

    # compute number density
    if n is None:
        k = _BOLTZMANN.magnitude
        n = p / (k * t)
    n = ureg.Quantity(value=n, units="m^-3")

    # compute absorption coefficient
    sigma = (n * xs).to("m^-1").magnitude

    return sigma
