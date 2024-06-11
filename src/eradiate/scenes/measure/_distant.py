from __future__ import annotations

import typing as t
from abc import ABC
from copy import deepcopy

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttr
from pinttr.util import ensure_units

from ._core import Measure
from ... import converters, frame, validators
from ...attrs import documented, parse_docs
from ...config import settings
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...util.misc import is_vector3

# ------------------------------------------------------------------------------
#                             Measure target interface
# ------------------------------------------------------------------------------


@attrs.define
class Target:
    """
    Interface for target selection objects used by distant measure classes.
    """

    def kernel_item(self) -> dict:
        """Return kernel item."""
        raise NotImplementedError

    @staticmethod
    def new(target_type, *args, **kwargs) -> Target:
        """
        Instantiate one of the supported child classes. This factory requires
        manual class registration. All position and keyword arguments are
        forwarded to the constructed type.

        Currently supported classes:

        * ``point``: :class:`.TargetPoint`
        * ``rectangle``: :class:`.TargetRectangle`

        Parameters
        ----------
        target_type : {"point", "rectangle"}
            Identifier of one of the supported child classes.

        Returns
        -------
        :class:`.Target`
        """
        if target_type == "point":
            return TargetPoint(*args, **kwargs)
        elif target_type == "rectangle":
            return TargetRectangle(*args, **kwargs)
        else:
            raise ValueError(f"unknown target type {target_type}")

    @staticmethod
    def convert(value) -> t.Any:
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`new` to
        instantiate a :class:`Target` child class based on the ``"type"`` entry
        it contains.

        If ``value`` is a 3-vector, this method returns a :class:`.TargetPoint`
        instance.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            d = deepcopy(value)
            try:
                target_type = d.pop("type")
            except KeyError:
                raise ValueError("cannot convert dict, missing 'type' entry")

            return Target.new(target_type, **d)

        if is_vector3(value):
            return Target.new("point", xyz=value)

        return value


def _target_point_rectangle_xyz_converter(x):
    return converters.on_quantity(float)(
        pinttr.converters.to_units(ucc.deferred("length"))(x)
    )


@parse_docs
@attrs.define
class TargetPoint(Target):
    """
    Point target or origin specification.
    """

    # Target point in config units
    xyz: pint.Quantity = documented(
        pinttr.field(units=ucc.deferred("length")),
        doc="Point coordinates.\n\nUnit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
    )

    @xyz.validator
    def _xyz_validator(self, attribute, value):
        if not is_vector3(value):
            raise ValueError(
                f"while validating {attribute.name}: must be a "
                f"3-element vector of numbers"
            )

    def kernel_item(self) -> dict:
        """Return kernel item."""
        return self.xyz.m_as(uck.get("length"))


@parse_docs
@attrs.define
class TargetRectangle(Target):
    """
    Rectangle target origin specification.

    This class defines an axis-aligned rectangular zone where ray targets will
    be sampled or ray origins will be projected.
    """

    xmin: pint.Quantity = documented(
        pinttr.field(
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Lower bound on the X axis.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="quantity or float",
    )

    xmax: pint.Quantity = documented(
        pinttr.field(
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Upper bound on the X axis.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="quantity or float",
    )

    ymin: pint.Quantity = documented(
        pinttr.field(
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Lower bound on the Y axis.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="quantity or float",
    )

    ymax: pint.Quantity = documented(
        pinttr.field(
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Upper bound on the Y axis.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="quantity or float",
    )

    z: pint.Quantity = documented(
        pinttr.field(
            default=0.0,
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Altitude of the plane enclosing the rectangle.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="quantity or float",
        default="0.0",
    )

    @xmin.validator
    @xmax.validator
    @ymin.validator
    @ymax.validator
    @z.validator
    def _xyz_validator(self, attribute, value):
        validators.on_quantity(validators.is_number)(self, attribute, value)

    @xmin.validator
    @xmax.validator
    def _x_validator(self, attribute, value):
        if not self.xmin < self.xmax:
            raise ValueError(
                f"while validating {attribute.name}: 'xmin' must "
                f"be lower than 'xmax"
            )

    @ymin.validator
    @ymax.validator
    def _y_validator(self, attribute, value):
        if not self.ymin < self.ymax:
            raise ValueError(
                f"while validating {attribute.name}: 'ymin' must "
                f"be lower than 'ymax"
            )

    def kernel_item(self) -> dict:
        # Inherit docstring

        kernel_length = uck.get("length")
        xmin = self.xmin.m_as(kernel_length)
        xmax = self.xmax.m_as(kernel_length)
        ymin = self.ymin.m_as(kernel_length)
        ymax = self.ymax.m_as(kernel_length)
        z = self.z.m_as(kernel_length)

        dx = xmax - xmin
        dy = ymax - ymin

        to_world = mi.ScalarTransform4f.translate(
            [0.5 * dx + xmin, 0.5 * dy + ymin, z]
        ) @ mi.ScalarTransform4f.scale([0.5 * dx, 0.5 * dy, 1.0])

        return {"type": "rectangle", "to_world": to_world}


# ------------------------------------------------------------------------------
#                             Distant measure interface
# ------------------------------------------------------------------------------


@parse_docs
@attrs.define(eq=False, slots=False)
class AbstractDistantMeasure(Measure, ABC):
    """
    Abstract interface of all distant measure classes.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    target: Target | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(Target.convert),
            validator=attrs.validators.optional(attrs.validators.instance_of(Target)),
            on_setattr=attrs.setters.pipe(
                attrs.setters.convert, attrs.setters.validate
            ),
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

    ray_offset: pint.Quantity | None = documented(
        pinttr.field(default=None, units=ucc.deferred("length")),
        doc="Manually control the distance between the target and ray origins. "
        "If unset, ray origins are positioned outside of the scene and this "
        "measure is rigorously distant.",
        type="quantity or None",
        init_type="float or quantity, optional",
        default="None",
    )

    @ray_offset.validator
    def _ray_offset_validator(self, attribute, value):
        if value is None:
            return

        if value.magnitude <= 0:
            raise ValueError(
                f"while validating '{attribute.name}': only positive values "
                f"are allowed, got {value}"
            )

    # --------------------------------------------------------------------------
    #                             Flag-style queries
    # --------------------------------------------------------------------------

    def is_distant(self) -> bool:
        # Inherit docstring
        return self.ray_offset is None


@parse_docs
@attrs.define(eq=False, slots=False)
class DistantMeasure(AbstractDistantMeasure):
    """
    Single-pixel distant measure scene element [``distant``]

    This scene element records radiance leaving the scene in a single direction
    defined by its ``direction`` parameter. Most users will however find the
    :class:`.MultiDistantMeasure` class more flexible.
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
                else frame.AzimuthConvention.convert(x)
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
        doc="A 3-vector defining the direction observed by the sensor, pointing "
        "outwards the target.",
        type="ndarray",
        init_type="array-like",
        default="[0, 0, 1]",
    )

    @property
    def film_resolution(self) -> tuple[int, int]:
        return 1, 1

    @property
    def viewing_angles(self) -> pint.Quantity:
        """
        quantity: Viewing angles computed from the `direction` parameter as
            (1, 1, 2) array. The last dimension is ordered as (zenith, azimuth).
        """
        angles = frame.direction_to_angles(
            self.direction,
            azimuth_convention=self.azimuth_convention,
            normalize=True,
        ).to(ucc.get("angle"))  # Convert to default angle units
        return np.reshape(angles, (1, 1, 2))

    # --------------------------------------------------------------------------
    #                         Additional constructors
    # --------------------------------------------------------------------------

    @classmethod
    def from_angles(cls, angles: pint.Quantity, **kwargs) -> DistantMeasure:
        """
        Construct using a direction layout defined by explicit (zenith, azimuth)
        pairs.

        Parameters
        ----------
        angles : array-like
            A (zenith, azimuth) pair, either as a quantity or a unitless
            array-like. In the latter case, the default angle units are applied.

        azimuth_convention : .AzimuthConvention or str, optional
            The azimuth convention applying to the viewing direction layout.
            If unset, the global default convention is used.

        **kwargs
            Remaining keyword arguments are forwarded to the
            :class:`.DistantMeasure` constructor.

        Returns
        -------
        DistantMeasure
        """
        azimuth_convention = kwargs.pop("azimuth_convention", None)
        if azimuth_convention is None:
            azimuth_convention = settings.azimuth_convention

        angles = ensure_units(angles, default_units=ucc.get("angle")).m_as(ureg.rad)
        direction = np.squeeze(
            frame.angles_to_direction(
                angles=angles, azimuth_convention=azimuth_convention
            )
        )
        return cls(direction=direction, **kwargs)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def kernel_type(self) -> str:
        # Inherit docstring
        return "distant"

    @property
    def template(self) -> dict:
        # Inherit docstring
        result = super().template
        result["direction"] = mi.ScalarVector3f(-self.direction)

        if self.target is not None:
            result["target"] = self.target.kernel_item()

        if self.ray_offset is not None:
            result["ray_offset"] = self.ray_offset.m_as(uck.get("length"))

        return result

    @property
    def var(self) -> tuple[str, dict]:
        # Inherit docstring
        return "radiance", {
            "standard_name": "radiance",
            "long_name": "radiance",
            "units": symbol(uck.get("radiance")),
        }


@parse_docs
@attrs.define(eq=False, slots=False)
class MultiPixelDistantMeasure(AbstractDistantMeasure):
    """
    Multi-pixel distant measure scene element [``mpdistant``, ``multipixel_distant``]

    This scene element records radiance leaving the scene in a single direction
    defined by its ``direction`` parameter. Most users will however find the
    :class:`.MultiDistantMeasure` class more flexible.
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
                else frame.AzimuthConvention.convert(x)
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
        doc="A 3-vector defining the direction observed by the sensor, pointing "
        "outwards the target.",
        type="ndarray",
        init_type="array-like",
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
        angles: pint.Quantity = frame.direction_to_angles(
            self.direction, azimuth_convention=self.azimuth_convention
        ).squeeze()
        shape = (*self.film_resolution, 2)
        return np.broadcast_to(angles.m, shape) * angles.u

    # --------------------------------------------------------------------------
    #                         Additional constructors
    # --------------------------------------------------------------------------

    @classmethod
    def from_angles(cls, angles: pint.Quantity, **kwargs) -> MultiPixelDistantMeasure:
        """
        Construct using a direction layout defined by explicit (zenith, azimuth)
        pairs.

        Parameters
        ----------
        angles : array-like
            A (zenith, azimuth) pair, either as a quantity or a unitless
            array-like. In the latter case, the default angle units are applied.

        azimuth_convention : .AzimuthConvention or str, optional
            The azimuth convention applying to the viewing direction layout.
            If unset, the global default convention is used.

        **kwargs
            Remaining keyword arguments are forwarded to the
            :class:`.DistantMeasure` constructor.

        Returns
        -------
        DistantMeasure
        """
        azimuth_convention = kwargs.pop("azimuth_convention", None)
        if azimuth_convention is None:
            azimuth_convention = settings.azimuth_convention

        angles = ensure_units(angles, default_units=ucc.get("angle")).m_as(ureg.rad)
        direction = np.squeeze(
            frame.angles_to_direction(
                angles=angles, azimuth_convention=azimuth_convention
            )
        )
        return cls(direction=direction, **kwargs)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def kernel_type(self) -> str:
        # Inherit docstring
        return "mpdistant"

    @property
    def template(self) -> dict:
        # Inherit docstring
        result = super().template
        result["direction"] = mi.ScalarVector3f(-self.direction)

        if self.target is not None:
            result["target"] = self.target.kernel_item()

        if self.ray_offset is not None:
            result["ray_offset"] = self.ray_offset.m_as(uck.get("length"))

        return result

    @property
    def var(self) -> tuple[str, dict]:
        # Inherit docstring
        return "radiance", {
            "standard_name": "radiance",
            "long_name": "radiance",
            "units": symbol(uck.get("radiance")),
        }
