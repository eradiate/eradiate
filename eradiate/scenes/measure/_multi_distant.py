from __future__ import annotations

import typing as t

import attr
import numpy as np
import pint
import pinttr

from ._core import Measure, MeasureFlags, measure_factory
from ._target import Target, TargetPoint, TargetRectangle
from ..core import KernelDict
from ...attrs import documented, get_doc, parse_docs
from ...contexts import KernelDictContext
from ...frame import angles_in_hplane, angles_to_direction, direction_to_angles
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg


@measure_factory.register(type_id="distant", allow_aliases=True)
@measure_factory.register(type_id="mdistant", allow_aliases=True)
@measure_factory.register(type_id="multi_distant", allow_aliases=True)
@parse_docs
@attr.s
class MultiDistantMeasure(Measure):
    """
    Multi-distant radiance measure scene element [``distant``, ``mdistant``,
    ``multi_distant``].

    This scene element creates a measure consisting of an array of
    radiancemeters positioned at an infinite distance from the scene. In
    practice, it can be used to compute the radiance leaving a scene at the
    top of the atmosphere (or canopy if there is no atmosphere). Coupled to
    appropriate post-processing operations, scene reflectance can be derived
    from the radiance values it produces.

    Notes
    -----
    * Setting the ``target`` parameter is required to get meaningful results.
      Experiment classes should take care of setting it appropriately.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    hplane: t.Optional[pint.Quantity] = documented(
        pinttr.ib(default=None, units=ucc.deferred("angle")),
        doc="If all directions are expected to be within a hemisphere plane cut, "
        "the azimuth value of that plane. Unitless values are converted to "
        "``ucc['angle']``.",
        type="quantity or None",
        init_type="float or quantity, optional",
        default="None",
    )

    @hplane.validator
    def hplane_validator(self, attribute, value):
        if value is None:
            return

        # Check that all specified directions are in the requested plane
        angles = self.viewing_angles.m_as(ureg.deg)
        try:
            angles_in_hplane(
                value.m_as(ureg.deg),
                angles[:, :, 0],
                angles[:, :, 1],
                raise_exc=True,
            )
        except ValueError as e:
            raise ValueError(
                f"while validating '{attribute.name}': 'directions' are not all "
                "part of the same hemisphere plane cut"
            ) from e

    directions: np.ndarray = documented(
        attr.ib(
            default=np.array([[0.0, 0.0, -1.0]]),
            converter=np.array,
        ),
        doc="A sequence of 3-vectors specifying distant sensing directions.",
        type="ndarray",
        init_type="array-like",
        default="[[0, 0, -1]]",
    )

    target: t.Optional[Target] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(Target.convert),
            validator=attr.validators.optional(
                attr.validators.instance_of(
                    (
                        TargetPoint,
                        TargetRectangle,
                    )
                )
            ),
            on_setattr=attr.setters.pipe(attr.setters.convert, attr.setters.validate),
        ),
        doc="Target specification. The target can be specified using an "
        "array-like with 3 elements (which will be converted to a "
        ":class:`.TargetPoint`) or a dictionary interpreted by "
        ":meth:`Target.convert() <.Target.convert>`. If set to "
        "``None`` (not recommended), the default target point selection "
        "method is used: rays will not target a particular region of the "
        "scene.",
        type=":class:`.Target` or None",
        init_type=":class:`.Target` or dict or array-like, optional",
    )

    @property
    def viewing_angles(self) -> pint.Quantity:
        """
        quantity: Viewing angles computed from stored `directions` as a
            (N, 1, 2) array, where N is the number of directions. The last
            dimension is ordered as (zenith, azimuth).
        """
        angle_units = ucc.get("angle")
        angles = direction_to_angles(-self.directions).to(angle_units)

        # Normalise azimuth to [0, 2π]
        angles[:, 1] %= 360.0 * ureg.deg
        return angles.reshape((-1, 1, 2))

    @property
    def film_resolution(self) -> t.Tuple[int, int]:
        return (self.directions.shape[0], 1)

    flags: MeasureFlags = documented(
        attr.ib(default=MeasureFlags.DISTANT, converter=MeasureFlags, init=False),
        doc=get_doc(Measure, "flags", "doc"),
        type=get_doc(Measure, "flags", "type"),
    )

    # --------------------------------------------------------------------------
    #                         Additional constructors
    # --------------------------------------------------------------------------

    @classmethod
    def from_viewing_angles(
        cls,
        zeniths: np.typing.ArrayLike,
        azimuths: np.typing.ArrayLike,
        auto_hplane: bool = True,
        **kwargs,
    ):
        """
        Construct a :class:`.MultiDistantMeasure` using viewing angles instead
        of raw directions.

        Parameters
        ----------
        zeniths : array-like
            List of zenith values (can be a quantity). Scalar values are
            broadcast to the same shape as `azimuths`. Unitless values are
            converted to ``ucc['angle']``.

        azimuths : array-like
            List of azimuth values (can be a quantity). Scalar values are
            broadcast to the same shape as `zeniths`. Unitless values are
            converted to ``ucc['angle']``.

        auto_hplane : bool, optional
            If ``True``, passing a scalar as `azimuths` will automatically set
            the measure's `hplane` parameter, unless an `hplane` keyword
            argument is also passed.

        **kwargs
            Any keyword argument (except `direction`) to be forwarded to
            :class:`MultiDistantMeasure() <.MultiDistantMeasure>`. The `hplane`
            keyword argument takes precedence over `auto_hplane`.

        Returns
        -------
        MultiDistantMeasure
        """
        if "directions" in kwargs:
            raise TypeError(
                "from_viewing_angles() got an unexpected keyword argument 'directions'"
            )

        angle_units = ucc.get("angle")

        # Basic unit conversion and array reshaping
        zeniths = pinttr.util.ensure_units(
            np.atleast_1d(zeniths).reshape((-1, 1)), default_units=angle_units
        ).m_as(angle_units)
        azimuths = pinttr.util.ensure_units(
            np.atleast_1d(azimuths).reshape((-1, 1)), default_units=angle_units
        ).m_as(angle_units)

        # Broadcast arrays if relevant
        if len(zeniths) == 1:
            zeniths = np.full_like(azimuths, zeniths[0])
        if len(azimuths) == 1:
            azimuths = np.full_like(zeniths, azimuths[0])

            # Auto-set 'hplane' if relevant
            if auto_hplane and "hplane" not in kwargs:
                kwargs["hplane"] = azimuths[0] * angle_units

        # Compute directions
        angles = np.hstack((zeniths, azimuths)) * angle_units
        directions = -angles_to_direction(angles)

        # Create instance
        return cls(directions=directions, **kwargs)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _kernel_dict(self, sensor_id, spp):
        result = {
            "type": "mdistant",
            "id": sensor_id,
            "directions": ",".join(map(str, self.directions.ravel(order="C"))),
            "sampler": {
                "type": "independent",
                "sample_count": spp,
            },
            "film": {
                "type": "hdrfilm",
                "width": self.film_resolution[0],
                "height": self.film_resolution[1],
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"},
            },
        }

        if self.target is not None:
            result["target"] = self.target.kernel_item()

        return result

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        sensor_ids = self._sensor_ids()
        sensor_spps = self._sensor_spps()
        result = KernelDict()

        for spp, sensor_id in zip(sensor_spps, sensor_ids):
            result.data[sensor_id] = self._kernel_dict(sensor_id, spp)

        return result

    # --------------------------------------------------------------------------
    #                        Post-processing information
    # --------------------------------------------------------------------------

    @property
    def var(self) -> str:
        return "radiance"
