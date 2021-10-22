from __future__ import annotations

import typing as t
import warnings

import attr
import numpy as np
import pint
import pinttr
from pinttr.util import ensure_units

import eradiate

from ._core import Measure, SensorInfo
from ._distant import TargetOrigin, TargetOriginPoint, TargetOriginRectangle
from ..core import KernelDict
from ... import validators
from ..._mode import ModeFlags
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext
from ...frame import angles_in_hplane, angles_to_direction, direction_to_angles
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg


@parse_docs
@attr.s
class MultiDistantMeasure(Measure):

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
                angles[:, 0],
                angles[:, 1],
                raise_exc=True,
            )
        except ValueError as e:
            raise ValueError(
                f"while validating '{attribute.name}': 'directions' are not all "
                "part of the same hemisphere plane cut"
            ) from e

    split_spp: t.Optional[int] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(int),
            validator=attr.validators.optional(validators.is_positive),
        ),
        type="int",
        init_type="int, optional",
        doc="If set, this measure will be split into multiple sensors, each "
        "with a sample count lower or equal to `split_spp`. This parameter "
        "should be used in single-precision modes when the sample count is "
        "higher than 100,000 (very high sample count might result in floating "
        "point number precision issues otherwise).",
    )

    @split_spp.validator
    def _split_spp_validator(self, attribute, value):
        if (
            eradiate.mode().has_flags(ModeFlags.ANY_SINGLE)
            and self.spp > 1e5
            and self.split_spp is None
        ):
            warnings.warn(
                "In single-precision modes, setting a sample count ('spp') to "
                "values greater than 100,000 may result in floating point "
                "precision issues: using the measure's 'split_spp' parameter is "
                "recommended."
            )

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

    target: t.Optional[TargetOrigin] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(TargetOrigin.convert),
            validator=attr.validators.optional(
                attr.validators.instance_of(
                    (
                        TargetOriginPoint,
                        TargetOriginRectangle,
                    )
                )
            ),
            on_setattr=attr.setters.pipe(attr.setters.convert, attr.setters.validate),
        ),
        doc="Target specification. The target can be specified using an "
        "array-like with 3 elements (which will be converted to a "
        ":class:`.TargetOriginPoint`) or a dictionary interpreted by "
        ":meth:`TargetOrigin.convert() <.TargetOrigin.convert>`. If set to "
        "``None`` (not recommended), the default target point selection "
        "method is used: rays will not target a particular region of the "
        "scene.",
        type=":class:`.TargetOrigin` or None",
        init_type=":class:`.TargetOrigin` or dict or array-like, optional",
    )

    @property
    def viewing_angles(self) -> pint.Quantity:
        """
        quantity: Viewing angles computed from stored `directions` as a (N, 2)
            array, where N is the number of directions. The second dimension
            is ordered as (zenith, azimuth).
        """
        return direction_to_angles(-self.directions).to(ucc.get("angle"))

    @property
    def film_resolution(self) -> t.Tuple[int, int]:
        return (self.directions.shape[0], 1)

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

    def _sensor_spps(self) -> t.List[int]:
        if self.split_spp is not None and self.spp > self._spp_splitting_threshold:
            spps = [self.split_spp] * int(self.spp / self.split_spp)

            if self.spp % self.split_spp:
                spps.append(self.spp % self.split_spp)

            return spps

        else:
            return [self.spp]

    def _sensor_id(self, i_spp=None):
        """
        Assemble a sensor ID from indexes on sensor coordinates.
        """
        components = [self.id]

        if i_spp is not None:
            components.append(f"spp{i_spp}")

        return "_".join(components)

    def _sensor_ids(self) -> t.List[str]:
        if self.split_spp is not None and self.spp > self._spp_splitting_threshold:
            return [self._sensor_id(i) for i, _ in enumerate(self._sensor_spps())]

        else:
            return [self._sensor_id()]

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

    def sensor_infos(self) -> t.List[SensorInfo]:
        return [
            SensorInfo(id=id, spp=spp)
            for id, spp in zip(self._sensor_ids(), self._sensor_spps())
        ]

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        sensor_ids, sensor_spps = [], []

        for x in self.sensor_infos():
            sensor_ids.append(x.id)
            sensor_spps.append(x.spp)

        result = KernelDict()

        for spp, sensor_id in zip(sensor_spps, sensor_ids):
            result.data[sensor_id] = self._kernel_dict(sensor_id, spp)

        return result

    def _base_dicts(self) -> t.List[t.Dict]:
        raise NotImplementedError
