from __future__ import annotations

from pathlib import Path

import attrs
import drjit as dr
import mitsuba as mi
import numpy as np
import pint
import pinttr

from ._core import Illumination
from ..core import traverse
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import define, documented
from ...kernel import SearchSceneParameter
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...validators import has_quantity


@define(eq=False, slots=False)
class SpotIllumination(Illumination):
    """
    Spot illumination scene element [``spot``].

    Eradiate ships a beam texture that implements a gaussian beam profile with
    three standard deviations included in the total beam width. This texture is
    named ``gaussian_3sigma.bmp`` and can be used by retrieving the path to the
    file using `eradiate.data.data_store.fetch()`.
    """

    origin: pint.Quantity = documented(
        pinttr.field(
            factory=lambda: [1, 1, 1] * ureg.m,
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
        attrs.field(
            default=None,
            converter=attrs.converters.optional(Path),
        ),
        type="Path",
        init_type="path-like",
        default=None,
        doc="Path to the file describing the beam profile. Must be a valid bitmap "
        "image file.",
    )

    @classmethod
    def from_size_at_target(
        cls,
        target: np.typing.ArrayLike,
        direction: np.typing.ArrayLike,
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
        angle_units = uck.get("angle")
        result = {
            "type": "spot",
            "beam_width": self.beam_width.m_as(angle_units),
            "cutoff_angle": self.beam_width.m_as(angle_units),
            "to_world": self._to_world,
            "id": self.id,
        }

        if self.beam_profile is not None:
            result["texture.type"] = "bitmap"
            result["texture.filename"] = str(self.beam_profile)

        kdict, _ = traverse(self.intensity)
        for k, v in kdict.items():
            result[f"intensity.{k}"] = v

        return result

    @property
    def params(self):
        _, kpmap_intensity = traverse(self.intensity)

        result = {}
        for key, param in kpmap_intensity.items():
            result[f"intensity.{key}"] = attrs.evolve(
                param,
                tracks=SearchSceneParameter(
                    node_type=mi.Emitter,
                    node_id=self.id,
                    parameter_relpath=f"intensity.{param.tracks.strip()}",
                )
                if isinstance(param.tracks, str)
                else param,
            )

        return result
