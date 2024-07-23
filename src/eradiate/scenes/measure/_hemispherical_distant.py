from __future__ import annotations

import attrs
import drjit as dr
import mitsuba as mi
import numpy as np
import pint
import pinttr

from ._distant import AbstractDistantMeasure
from ... import frame, validators
from ...attrs import define, documented
from ...config import settings
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...warp import square_to_uniform_hemisphere


@define(eq=False, slots=False)
class HemisphericalDistantMeasure(AbstractDistantMeasure):
    """
    Hemispherical distant radiance measure scene element
    [``hdistant``, ``hemispherical_distant``].

    This scene element records radiance leaving the scene in a hemisphere
    defined by its ``direction`` parameter. A distinctive feature of this
    measure is that it samples continuously the direction space instead of
    computing radiance values for a fixed set of directions, thus potentially
    capturing effects much harder to distinguish using *e.g.* the
    :class:`.MultiDistantMeasure` class. On the other side, features located
    at a precise angle will not be captured very well by this measure.

    This measure is useful to get a global view of leaving radiance patterns
    over a surface.

    Notes
    -----
    * Setting the ``target`` parameter is required to get meaningful results.
      Experiment classes should take care of setting it appropriately.
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

    _film_resolution: tuple[int, int] = documented(
        attrs.field(
            default=(32, 32),
            validator=attrs.validators.deep_iterable(
                member_validator=attrs.validators.instance_of(int),
                iterable_validator=validators.has_len(2),
            ),
        ),
        doc="Film resolution as a (width, height) 2-tuple. "
        "If the height is set to 1, direction sampling will be restricted to a "
        "plane.",
        type="array-like",
        default="(32, 32)",
    )

    orientation: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity(0.0, ureg.deg),
            validator=[validators.is_positive, pinttr.validators.has_compatible_units],
            units=ucc.deferred("angle"),
        ),
        doc="Azimuth angle defining the orientation of the sensor in the "
        "horizontal plane.\n"
        "\n"
        "Unit-enabled field (default: ucc['angle']).",
        type="float",
        default="0.0 deg",
    )

    direction = documented(
        attrs.field(
            default=[0, 0, 1],
            converter=np.array,
            validator=validators.is_vector3,
        ),
        doc="A 3-vector orienting the hemisphere mapped by the measure.",
        type="array-like",
        default="[0, 0, 1]",
    )

    @property
    def film_resolution(self):
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
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def kernel_type(self) -> str:
        # Inherit docstring
        return "hdistant"

    @property
    def template(self) -> dict:
        # Inherit docstring
        result = super().template

        up = dr.normalize(
            dr.cross(
                mi.ScalarVector3f(self.direction),
                mi.ScalarVector3f(
                    np.cos(self.orientation.m_as(ureg.rad)),
                    np.sin(self.orientation.m_as(ureg.rad)),
                    0.0,
                ),
            )
        )
        result["to_world"] = mi.ScalarTransform4f.look_at(
            origin=[0.0, 0.0, 0.0], target=self.direction, up=up
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
        return "radiance", {
            "standard_name": "radiance",
            "long_name": "radiance",
            "units": symbol(uck.get("radiance")),
        }
