from __future__ import annotations

import attrs
import mitsuba as mi
import numpy as np
import pint

from ._distant import AbstractDistantMeasure
from ... import frame, validators
from ...attrs import define, documented
from ...config import settings
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...warp import square_to_uniform_hemisphere


@define(eq=False, slots=False)
class DistantFluxMeasure(AbstractDistantMeasure):
    """
    Distant radiosity measure scene element [``distantflux``, ``distant_flux``].

    This measure records the [spectral] radiosity (in W/m²[/nm]) at an infinite
    distance in a hemisphere defined by its ``direction`` parameter.

    When used with a backward tracing algorithm, rays traced by the sensor
    target a shape which can be controlled through the ``target`` parameter.
    This feature is useful if one wants to compute the average flux leaving
    a particular subset of the scene.

    Notes
    -----
    * Setting the ``target`` parameter is required to get meaningful results.
      Experiment classes should take care of setting it appropriately.
    * The film resolution can be adjusted to manually stratify film sampling
      and reduce variance in results. The default 32x32 is generally a good
      choice, but scenes with sharp reflection lobes may benefit from higher
      values.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    azimuth_convention: frame.AzimuthConvention = documented(
        attrs.field(
            default=None,
            converter=lambda x: (
                settings.azimuth_convention
                if x is None
                else (frame.AzimuthConvention[x.upper()] if isinstance(x, str) else x)
            ),
            validator=attrs.validators.instance_of(frame.AzimuthConvention),
        ),
        doc="Azimuth convention. If ``None``, the global default configuration "
        "is used (see :ref:`sec-user_guide-config`).",
        type=".AzimuthConvention",
        init_type=".AzimuthConvention or str, optional",
        default="None",
    )

    direction: np.ndarray = documented(
        attrs.field(
            default=[0, 0, 1],
            converter=np.array,
            validator=validators.is_vector3,
        ),
        doc="A 3-vector defining the normal to the reference surface for which "
        "the exitant flux density is computed.",
        type="array-like",
        default="[0, 0, 1]",
    )

    _film_resolution: tuple[int, int] = documented(
        attrs.field(
            default=(32, 32),
            validator=attrs.validators.deep_iterable(
                member_validator=attrs.validators.instance_of(int),
                iterable_validator=validators.has_len(2),
            ),
        ),
        doc="Film resolution as a (width, height) 2-tuple.",
        type="array-like",
        default="(32, 32)",
    )

    @property
    def film_resolution(self) -> tuple[int, int]:
        return self._film_resolution

    @property
    def viewing_angles(self) -> pint.Quantity:
        """
        quantity: Viewing angles computed from stored film coordinates as a
            (width, height, 2) array. The last dimension is ordered as
            (zenith, azimuth).
        """
        # Compute viewing angles at pixel locations
        # Angle computation must match the kernel plugin's direction sampling
        # routine
        angle_units = ucc.get("angle")

        # Compute pixel locations in film coordinates
        xs = (
            np.linspace(0, 1, self.film_resolution[0], endpoint=False)
            + 0.5 / self.film_resolution[0]
        )
        ys = (
            np.linspace(0, 1, self.film_resolution[1], endpoint=False)
            + 0.5 / self.film_resolution[1]
        )

        # Compute corresponding angles in specified azimuth convention
        xy = np.array([(x, y) for x in xs for y in ys])
        angles = frame.direction_to_angles(
            square_to_uniform_hemisphere(xy),
            azimuth_convention=self.azimuth_convention,
        ).to(angle_units)

        # Reshape array to match film size on first 2 dimensions
        return angles.reshape((len(xs), len(ys), 2))

    # --------------------------------------------------------------------------
    #                        Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def kernel_type(self) -> str:
        # Inherit docstring
        return "distantflux"

    @property
    def template(self) -> dict:
        # Inherit docstring
        result = super().template

        _, up = mi.coordinate_system(self.direction)
        result["to_world"] = mi.ScalarTransform4f.look_at(
            origin=[0, 0, 0],
            target=mi.ScalarVector3f(self.direction),
            up=up,
        )

        if self.target is not None:
            result["target"] = self.target.kernel_item()

        if self.ray_offset is not None:
            result["ray_offset"] = self.ray_offset.m_as(uck.get("length"))

        return result

    # --------------------------------------------------------------------------
    #                        Post-processing information
    # --------------------------------------------------------------------------

    @property
    def var(self) -> tuple[str, dict]:
        # Inherit docstring
        return "sector_radiosity", {
            "standard_name": "sector_radiosity",
            "long_name": "sector radiosity",
            "units": symbol(uck.get("irradiance")),
        }
