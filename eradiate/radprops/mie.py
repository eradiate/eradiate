"""
Mie scattering properties computation.
"""
import typing as t

import miepython
import numpy as np
import pint
import xarray as xr

from ..units import unit_registry as ureg

xr.set_options(keep_attrs=True)

_DEFAULT_MU = np.linspace(-1, 1, 201)


@ureg.wraps(ret=None, args=("nm", None, None, "dimensionless"), strict=False)
def compute_properties(
    w: t.Union[ureg.Quantity, np.ndarray],
    rdist: xr.DataArray,
    m: xr.DataArray,
    mu: t.Union[ureg.Quantity, np.ndarray] = _DEFAULT_MU,
) -> xr.DataArray:
    """
    Compute multi-wavelength radiative properties of spherical particles with
    a radius distribution.

    Parameters
    ----------
    w: :class:`~pint.Quantity` or :class:`~numpy.ndarray`
        Wavelength values [nm].

    rdist: :class:`~xarray.DataArray`
        Particle radius probability distribution.

    m: :class:`~xarray.DataArray`
        Complex refractive index.

    mu: :class:`~pint.Quantity` or :class:`~numpy.ndarray`
        Scattering angle cosines [dimensionless].

    Returns
    -------
    :class:`~xarray.DataArray`
        Radiative properties.
    """
    datasets = []
    m_max = m.values.max()
    m_min = m.values.min()
    for wavelength in w:
        m_value = complex(
            m.interp(w=wavelength, kwargs=dict(fill_value=(m_min, m_max))).values
        )
        ds = compute_mono_properties(w=wavelength, rdist=rdist, m=m_value, mu=mu)
        datasets.append(ds)

    return xr.concat(datasets, dim="w")


@ureg.wraps(
    ret=None,
    args=("nanometer", None, "dimensionless", "dimensionless"),
    strict=False,
)
def compute_mono_properties(
    w: t.Union[ureg.Quantity, float],
    rdist: xr.DataArray,
    m: t.Union[ureg.Quantity, complex],
    mu: t.Union[ureg.Quantity, np.ndarray] = _DEFAULT_MU,
):
    """
    Compute the mono-wavelength radiative properties of spherical particles
    with radii specified by a probabiblity distribution.

    Parameters
    ----------
    w: :class:`~pint.Quantity` or float
        Wavelength [nanometer].

    rdist: :class:`~xarray.DataArray`
        Radius probability distribution.

    m: :class:`~pint.Quantity` or complex
        Complex refractive index [dimensionless].

    mu: :class:`~pint.Quantity` or :class:`numpy.ndarray`
        Scattering angle cosines [dimensionless].

    Returns
    -------
    :class:`~xarray.Dataset`
         Monochromatic radii-averaged radiative properties.
    """
    ds = compute_mono_properties_multiple_radius(w=w, r=rdist.r.values, m=m, mu=mu)
    return ds.weighted(rdist).mean(dim="r")


@ureg.wraps(
    ret=None,
    args=("nanometer", "micrometer", "dimensionless", "dimensionless"),
    strict=False,
)
def compute_mono_properties_multiple_radius(
    w: t.Union[ureg.Quantity, float],
    r: t.Union[ureg.Quantity, np.ndarray],
    m: t.Union[ureg.Quantity, complex],
    mu: t.Union[ureg.Quantity, np.ndarray] = _DEFAULT_MU,
) -> xr.Dataset:
    """
    Compute the mono-wavelength multiple-radius radiative properties of
    spherical particles.

    Parameters
    ----------
    w: :class:`~pint.Quantity` or float
        Wavelength [nanometer].

    r: :class:`~pint.Quantity` or :class:`~numpy.ndarray`
        Radius values [micrometer].

    m: :class:`~pint.Quantity` or complex
        Complex refractive index [dimensionless].

    mu: :class:`~pint.Quantity` or :class:`numpy.ndarray`
        Scattering angle cosines [dimensionless].

    Returns
    -------
    :class:`~xarray.Dataset`
         Multiple-radius monochromatic radiative properties.
    """
    dsi = []
    for ri in r:
        dsi.append(compute_mono_properties_single_radius(w=w, r=ri, m=m, mu=mu))
    return xr.concat(dsi, dim="r")


@ureg.wraps(ret=None, args=("nanometer", "micrometer", "", ""), strict=False)
def compute_mono_properties_single_radius(
    w: t.Union[float, ureg.Quantity],
    r: t.Union[float, ureg.Quantity],
    m: t.Union[complex, ureg.Quantity],
    mu: t.Union[np.ndarray] = _DEFAULT_MU,
) -> xr.Dataset:
    """
    Compute monochromatic radiative properties for one given particle radius.

    The radiative properties are computed for one particle radius value and at
    a single wavelength.
    The computed radiative properties are the extinction cross section, the
    albedo and the scattering phase function.
    The phase function is normalised so that its integral over the unit sphere
    is 1.

    Parameters
    ----------
    w: float or :class:`pint.Quantity`
        wavelength in a vacuum [nm].

    r: float or :class:`pint.Quantity`
        radius of the spherical scatterer [micrometer].

    m: complex
        index of refraction [dimensionless].

    mu: array
        Cosine of the scattering angle [dimensionless].

    Returns
    -------
    :class:`~xarray.Dataset`
        Computed monochromatic radiative properties.
    """
    # make input parameters physical quantities (except 'm' and 'mu')
    w = ureg.Quantity(w, "nanometer")
    r = ureg.Quantity(r, "micrometer")

    # compute size parameter
    x = (2 * np.pi * r / w).m_as(ureg.dimensionless)

    # compute efficiencies
    qext, qsca, _, _ = miepython.mie(m=m, x=x)

    # compute extinction cross section and albedo
    area = np.array(np.pi) * (r.to("m")) ** 2
    xs_t = qext * area
    albedo = ureg.Quantity(qsca / qext, "")

    # compute phase function
    phase = miepython.i_unpolarized(m=m, x=x, mu=mu)

    # re-normalise phase function
    phase = ureg.Quantity(phase * 1 / albedo.magnitude, "steradian^-1")

    return make_data_set(
        phase=phase,
        xs_t=xs_t,
        albedo=albedo,
        w=w,
        r=r,
        m=m,
        mu=mu,
    )


@ureg.wraps(
    ret=None, args=("steradian^-1", "m^2", "", "nm", "micrometer", "", ""), strict=False
)
def make_data_set(
    phase: t.Union[pint.Quantity, np.ndarray],
    xs_t: t.Union[pint.Quantity, float],
    albedo: t.Union[pint.Quantity, float],
    w: t.Union[pint.Quantity, float],
    r: t.Union[pint.Quantity, float],
    m: t.Union[pint.Quantity, complex],
    mu: t.Union[pint.Quantity, np.ndarray],
) -> xr.Dataset:
    """
    Make a radiative properties data set.

    Parameters
    ----------
    phase: :class:`~pint.Quantity` or :class:`~numpy.ndarray`
        Scattering phase function [steradian^-1].

    xs_t: :class:`~pint.Quantity` or float
        Extinction cross section [m^2].

    albedo: :class:`~pint.Quantity` or float)
        Albedo [dimensionless].

    w: :class:`~pint.Quantity` or float)
        Wavelength [nm].

    r: :class:`~pint.Quantity` or float
        Radius [micrometer].

    m: :class:`~pint.Quantity` or complex
        Index of refraction [dimensionless].

    mu: :class:`~pint.Quantity` or :class:`~numpy.ndarray`
        Scattering angles cosine values [dimensionless].

    Returns:
    --------
        Radiative properties :class:`~xarray.Dataset`
    """
    return xr.Dataset(
        data_vars={
            "phase": (
                ["w", "r", "mu"],
                phase.reshape(1, 1, len(mu)),
                dict(
                    standard_name="scattering_phase_function",
                    long_name="scattering phase function",
                    units="steradian^-1",
                ),
            ),
            "xs_t": (
                ["w", "r"],
                np.array([xs_t]).reshape(1, 1),
                dict(
                    standard_name="extinction_cross_section",
                    long_name="extinction cross section",
                    units="m^2",
                ),
            ),
            "albedo": (
                ["w", "r"],
                np.array([albedo]).reshape(1, 1),
                dict(standard_name="albedo", long_name="albedo", units=""),
            ),
        },
        coords={
            "w": (
                "w",
                [w],
                dict(
                    standard_name="radiation_wavelength",
                    long_name="radiation_wavelength",
                    units="nm",
                ),
            ),
            "r": (
                "r",
                [r],
                dict(
                    standard_name="particle_radius",
                    long_name="particle radius",
                    units="micrometer",
                ),
            ),
            "m": (
                "m",
                [m],
                dict(
                    standard_name="refractive_index",
                    long_name="refractive index",
                    units="",
                ),
            ),
            "mu": (
                "mu",
                mu,
                dict(
                    standard_name="scattering_angle_cosine",
                    long_name="scattering angle cosine",
                    units="",
                ),
            ),
        },
    )
