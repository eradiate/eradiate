from __future__ import annotations

from pathlib import Path

import attrs
import drjit as dr
import mitsuba as mi
import numpy as np
import pint
import pinttr
from numpy.typing import ArrayLike

from ._core import Illumination
from ..core import NodeSceneElement
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import define, documented
from ...converters import resolve_path
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...validators import has_quantity


@define(eq=False, slots=False)
class SpotIllumination(Illumination):
    """
    Spot illumination scene element [``spot``].

    Notes
    -----
    This illuminant can be applied a beam profile using a texture. Eradiate
    ships a Gaussian beam profile texture (3σ decay) as part of the
    ``texture/core`` resource. Once installed, this file can be accessed through
    the file resolver as ``texture/gaussian_3sigma.bmp``.
    """

    origin: pint.Quantity = documented(
        pinttr.field(
            factory=lambda: [1.0, 1.0, 1.0] * ureg.m,
            validator=[validators.has_len(3), pinttr.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="A 3-vector specifying the position of the spot.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
        default="[1, 1, 1] m",
    )

    target: pint.Quantity = documented(
        pinttr.field(
            factory=lambda: [0, 0, 0] * ureg.m,
            validator=[validators.has_len(3), pinttr.validators.has_compatible_units],
            units=ucc.deferred("length"),
        ),
        doc="Point location targeted by the spot.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
        default="[0, 0, 0] m",
    )

    @target.validator
    @origin.validator
    def _target_origin_validator(self, attribute, value):
        if np.allclose(self.target, self.origin):
            raise ValueError(
                f"While initializing {attribute}: "
                f"Origin and target must not be equal, "
                f"got target = {self.target}, origin = {self.origin}"
            )

    up: np.ndarray = documented(
        attrs.field(
            converter=np.array,
            validator=validators.has_len(3),
        ),
        doc="A 3-vector specifying the up direction of the spot.\n"
        "This vector must be different from the spots's pointing direction,\n"
        "which is given by ``target - origin``.",
        type="array",
        init_type="array-like",
        default="[0, 0, 1]",
    )

    @up.default
    def _up_factory(self):
        direction = dr.normalize(
            mi.ScalarVector3f(
                self.origin.m_as(ucc.get("length"))
                - self.target.m_as(ucc.get("length"))
            )
        )
        return mi.coordinate_system(direction)[0]

    @up.validator
    def _up_validator(self, attribute, value):
        direction = self.target - self.origin
        if np.allclose(np.cross(direction, value), 0):
            raise ValueError(
                f"While initializing '{attribute.name}': "
                f"up direction must not be colinear with viewing direction, "
                f"got up = {self.up}, direction = {direction}"
            )

    beam_width: pint.Quantity = documented(
        pinttr.field(default=10.0 * ureg.deg, units=ucc.deferred("angle")),
        doc="Spot light beam width.\n\nUnit-enabled field (default: degree).",
        type="quantity",
        init_type="quantity or float",
        default="10°",
    )

    intensity: Spectrum = documented(
        attrs.field(
            default=1.0,
            converter=spectrum_factory.converter("intensity"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                has_quantity("intensity"),
            ],
        ),
        doc="Emitted power in the plane orthogonal to the illumination direction. "
        "Must be an intensity spectrum (in W/sr/nm or compatible unit). "
        "Can be initialized with a dictionary processed by "
        ":meth:`.SpectrumFactory.convert`.",
        type=":class:`~eradiate.scenes.spectra.Spectrum`",
        init_type=":class:`~eradiate.scenes.spectra.Spectrum` or dict or float",
        default="1.0 ucc[intensity]",
    )

    beam_profile: Path | None = documented(
        attrs.field(default=None, converter=attrs.converters.optional(resolve_path)),
        type="Path or None",
        init_type="path-like, optional*",
        default=None,
        doc="Path to the file describing the beam profile. Must be a valid bitmap "
        "image file.",
    )

    @classmethod
    def from_size_at_target(
        cls,
        target: ArrayLike,
        direction: ArrayLike,
        spot_radius: pint.Quantity,
        beam_width: pint.Quantity,
        **kwargs,
    ) -> SpotIllumination:
        """
        Create a :class:`.SpotIllumination` which illuminates a region of a
        specified size around the target point.

        The illuminated area is a circle defined by the spot radius around the
        target point in the plane orthogonal to the spot direction. The spot
        origin is inferred from the spot radius and a beam width value
        (in angle units): the spot origin will be positioned closer to the
        target point as the beam width increases and further from the target
        point as the beam width decreases.

        Parameters
        ----------
        target : array-like
            Target point for the spot.
            Unitless values are converted to ``ucc['length']``.

        direction : array-like
            Pointing direction for the spot, towards target.

        spot_radius : float or quantity
            Radius of the desired spot at the target position.
            Unitless values are converted to ``ucc['length']``.

        beam_width : float or quantity
            Divergence angle of the spot.
            Unitless values are converted to ``ucc['angle']``.

        **kwargs
            Remaining keyword arguments are forwarded to the
            :class:`.SpotIllumination` constructor.

        Returns
        -------
        SpotIllumination
        """

        if "origin" in kwargs:
            raise TypeError(
                "The from_size_at_target constructor computes the origin position."
            )

        target = pinttr.util.ensure_units(target, default_units=ucc.get("length"))
        half_angle = (
            pinttr.util.ensure_units(beam_width, default_units=ureg.radian) / 2.0
        )
        tan_divergence = np.tan(half_angle)
        distance = spot_radius / tan_divergence

        origin = target - (direction * distance)

        return cls(origin=origin, target=target, beam_width=beam_width, **kwargs)

    @property
    def _to_world(self) -> mi.ScalarTransform4f:
        target = self.target.m_as(uck.get("length"))
        origin = self.origin.m_as(uck.get("length"))
        return mi.ScalarTransform4f.look_at(origin=origin, target=target, up=self.up)

    @property
    def template(self) -> dict:
        retdict = {
            "type": "spot",
            "beam_width": self.beam_width.m_as(uck.get("angle")),
            "cutoff_angle": self.beam_width.m_as(uck.get("angle")),
            "to_world": self._to_world,
        }

        if self.beam_profile is not None:
            retdict["texture"] = {
                "type": "bitmap",
                "filename": str(self.beam_profile),
            }
        return retdict

    @property
    def objects(self) -> dict[str, NodeSceneElement]:
        return {"intensity": self.intensity}
