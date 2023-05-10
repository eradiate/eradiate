from __future__ import annotations

import logging
import typing as t
import warnings

import attrs
import numpy as np
import pint
import pinttr
import xarray as xr

import eradiate

from ._core import PipelineStep
from ..attrs import documented, parse_docs
from ..exceptions import UnsupportedModeError
from ..frame import angles_in_hplane
from ..radprops._afgl1986 import G16
from ..scenes.illumination import ConstantIllumination, DirectionalIllumination
from ..scenes.measure import Measure
from ..scenes.spectra import (
    InterpolatedSpectrum,
    Spectrum,
)
from ..units import symbol, to_quantity
from ..units import unit_context_kernel as uck
from ..units import unit_registry as ureg

logger = logging.getLogger(__name__)


@parse_docs
@attrs.define
class AddIllumination(PipelineStep):
    """
    Add illumination data.

    This post-processing pipeline step adds illumination data:

    * if `illumination` is a :class:`.DirectionalIllumination` instance, then
      a data variable (holding the incoming top-of-scene flux with respect to a
      horizontal surface) is created, with dimensions ``sza`` and ``vaa``;
    * if `illumination` is a :class:`.ConstantIllumination` instance, then
      the created data variable has no coordinate.
    """

    illumination: DirectionalIllumination | ConstantIllumination = documented(
        attrs.field(
            validator=attrs.validators.instance_of(
                (DirectionalIllumination, ConstantIllumination)
            ),
            repr=lambda self: f"{self.__class__.__name__}(id='{self.id}', ...)",
        ),
        doc="An :class:`.Illumination` instance from which the illumination "
        "data originates.",
        type=":class:`.DirectionalIllumination` or :class:`.ConstantIllumination`",
    )

    measure: Measure = documented(
        attrs.field(
            validator=attrs.validators.instance_of(Measure),
            repr=lambda self: f"{self.__class__.__name__}(id='{self.id}', ...)",
        ),
        doc="A :class:`.Measure` instance from the data originates.",
        type=":class:`.Measure`",
    )

    irradiance_var: str = documented(
        attrs.field(default="irradiance", validator=attrs.validators.instance_of(str)),
        doc="Name of the variable storing irradiance (incoming flux) values.",
        type="str",
        default='"irradiance"',
    )

    def transform(self, x: t.Any) -> t.Any:
        logger.debug("add_illumination: begin")
        k_irradiance_units = uck.get("irradiance")
        illumination = self.illumination
        # measure = self.measure
        result = x.copy(deep=False)

        # Collect spectral coordinate values for verification purposes
        wavelengths_dataset = to_quantity(x.w)

        def eval_illumination_spectrum(
            field_name: str, k_units: pint.Unit
        ) -> pint.Quantity:
            # Local helper function to help with illumination spectrum evaluation

            spectrum: Spectrum = getattr(illumination, field_name)
            results_wavelengths = to_quantity(x.w)

            if eradiate.mode().is_mono:
                wavelengths = results_wavelengths
                assert np.allclose(wavelengths, wavelengths_dataset)
                return spectrum.eval_mono(wavelengths).m_as(k_units)

            elif eradiate.mode().is_ckd:
                result = spectrum.eval_ckd(
                    w=results_wavelengths,
                    g=G16,  # TODO: PR#311 hack
                ).m_as(k_units)

                # Reorder data by ascending wavelengths
                # indices = results_wavelengths.argsort()
                wavelengths = results_wavelengths  # [indices]
                assert np.allclose(wavelengths, wavelengths_dataset)
                # result = result[indices]

                return result

            else:
                raise UnsupportedModeError(supported=("monochromatic", "ckd"))

        if isinstance(illumination, DirectionalIllumination):
            # Collect illumination angular data
            saa = illumination.azimuth.m_as(ureg.deg)
            sza = illumination.zenith.m_as(ureg.deg)
            cos_sza = np.cos(np.deg2rad(sza))

            # Add angular dimensions
            result = result.expand_dims({"sza": [sza], "saa": [saa]}, axis=(0, 1))
            result.coords["sza"].attrs = {
                "standard_name": "solar_zenith_angle",
                "long_name": "solar zenith angle",
                "units": symbol("deg"),
            }
            result.coords["saa"].attrs = {
                "standard_name": "solar_azimuth_angle",
                "long_name": "solar azimuth angle",
                "units": symbol("deg"),
            }

            # Collect illumination spectral data
            irradiances = eval_illumination_spectrum("irradiance", k_irradiance_units)

            # Add irradiance variable
            result[self.irradiance_var] = (
                ("sza", "saa", "w"),
                (irradiances * cos_sza).reshape((1, 1, len(irradiances))),
            )

        elif isinstance(illumination, ConstantIllumination):
            # Collect illumination spectral data
            k_radiance_units = uck.get("radiance")
            radiances = eval_illumination_spectrum("radiance", k_radiance_units)

            # Add irradiance variable
            result[self.irradiance_var] = (
                ("w",),
                np.pi * radiances.reshape((len(radiances),)),
            )

        else:
            raise TypeError(
                "field 'illumination' must be one of "
                "(DirectionalIllumination, ConstantIllumination), got a "
                f"{illumination.__class__.__name__}"
            )

        result[self.irradiance_var].attrs = {
            "standard_name": "horizontal_solar_irradiance_per_unit_wavelength",
            "long_name": "horizontal spectral irradiance",
            "units": symbol(k_irradiance_units),
        }
        logger.debug("add_illumination: end")
        return result


@parse_docs
@attrs.define
class AddViewingAngles(PipelineStep):
    """
    Create new ``vza`` and ``vaa`` coordinate variables mapping viewing angles
    to other coordinates.
    """

    measure: Measure = documented(
        attrs.field(
            validator=attrs.validators.instance_of(Measure),
            repr=lambda self: f"{self.__class__.__name__}(id='{self.id}', ...)",
        ),
        doc="A :class:`.Measure` instance from which the processed data originates.",
        type=":class:`.Measure`",
    )

    def transform(self, x: t.Any) -> t.Any:
        measure = self.measure
        viewing_angles = measure.viewing_angles

        # Collect zenith and azimuth values
        theta = viewing_angles[:, :, 0]
        phi = viewing_angles[:, :, 1]

        # Attach coordinates
        with xr.set_options(keep_attrs=True):
            result = x.assign_coords(
                {
                    "vza": (
                        ("x_index", "y_index"),
                        theta.m_as(ureg.deg),
                        {
                            "standard_name": "viewing_zenith_angle",
                            "long_name": "viewing zenith angle",
                            "units": symbol("deg"),
                        },
                    ),
                    "vaa": (
                        ("x_index", "y_index"),
                        phi.m_as(ureg.deg),
                        {
                            "standard_name": "viewing_azimuth_angle",
                            "long_name": "viewing azimuth angle",
                            "units": symbol("deg"),
                        },
                    ),
                }
            )

        return result


@ureg.wraps(ret=("rad", "rad"), args=("rad", "rad", "rad"), strict=True)
def _remap_viewing_angles_plane(
    plane: np.typing.ArrayLike,
    theta: np.typing.ArrayLike,
    phi: np.typing.ArrayLike,
) -> tuple[np.typing.ArrayLike, np.typing.ArrayLike]:
    r"""
    Remap viewing angles to a hemisphere plane cut.

    Parameters
    ----------
    plane : quantity
         Plane cut orientation (scalar value).

    theta : quantity
        List of zenith angle values with (N,) shape.

    phi : quantity
        List of azimuth angle values with (N,) shape.

    Returns
    -------
    theta : quantity
        List of zenith angle values in :math:`[-\pi, \pi]` with (N,) shape.

    phi : quantity
        List of azimuth angle values in :math:`[0, 2\pi]` with (N,) shape
        (equal to `plane` modulo :math:`\pi`).

    Warns
    -----
    When zenith angle values are not sorted in ascending order.
    """
    # Normalize all angles
    twopi = 2.0 * np.pi
    plane = plane % twopi
    theta = theta % twopi
    phi = phi % twopi

    # Check that phi values are compatible with requested plane
    in_plane_positive, in_plane_negative = angles_in_hplane(
        plane, theta, phi, raise_exc=True
    )

    # Check if any point is allocated to both half-planes (uncomment to debug)
    # assert not np.any(in_plane_positive & in_plane_negative)

    # Normalise zenith values
    theta = np.where(in_plane_positive, theta, -theta)

    # Normalize azimuth values
    phi = np.full_like(theta, plane)

    # Check ordering and warn if it is not strictly increasing
    if not _is_sorted(theta):
        warnings.warn(
            "Viewing zenith angle values are not sorted in ascending order, "
            "you might want to consider changing direction definitions."
        )

    return theta, phi


_is_sorted = lambda a: np.all(a[:-1] <= a[1:])


@parse_docs
@attrs.define
class AddSpectralResponseFunction(PipelineStep):
    """
    Add spectral response function data.

    This post-processing pipeline step adds spectral response function data to
    the processed dataset.
    """

    measure: Measure = documented(
        attrs.field(
            validator=attrs.validators.instance_of(Measure),
            repr=lambda self: f"{self.__class__.__name__}(id='{self.id}', ...)",
        ),
        doc="A :class:`.Measure` instance from which the processed data originates.",
        type=":class:`.Measure`",
    )

    def transform(self, x: t.Any) -> t.Any:
        logger.debug("add_spectral_response_function: begin")
        result = x.copy(deep=False)

        # Evaluate SRF
        srf = self.measure.srf

        if isinstance(srf, InterpolatedSpectrum):
            srf_w = srf.wavelengths
            srf_values = pinttr.util.ensure_units(srf.values, ureg.dimensionless)

        else:
            raise TypeError(f"unsupported SRF type '{srf.__class__.__name__}'")

        # Add SRF spectral coordinates to dataset
        w_units = ureg.nm
        result = result.assign_coords(
            {
                "srf_w": (
                    "srf_w",
                    srf_w.m_as(w_units),
                    {
                        "standard_name": "radiation_wavelength",
                        "long_name": "wavelength",
                        "units": "nm",
                    },
                )
            }
        )

        # Add SRF variable to dataset
        result["srf"] = ("srf_w", srf_values.m)
        result.srf.attrs = {
            "standard_name": "spectral_response_function",
            "long_name": "spectral response function",
            "units": "",
        }
        logger.debug("add_spectral_response_function: end")

        return result
