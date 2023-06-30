from __future__ import annotations

import typing as t
from abc import ABC
from copy import deepcopy

import attrs
import mitsuba as mi
import pint
import pinttr

from ._core import Measure
from ... import converters, validators
from ...attrs import documented, parse_docs
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
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
class DistantMeasure(Measure, ABC):
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
