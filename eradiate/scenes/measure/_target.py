from __future__ import annotations

import typing as t
from copy import deepcopy

import attr
import pint
import pinttr

from ... import converters, validators
from ..._util import is_vector3
from ...attrs import documented, parse_docs
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck


@attr.s
class Target:
    """
    Interface for target selection objects used by distant measure classes.
    """

    def kernel_item(self) -> t.Dict:
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
@attr.s
class TargetPoint(Target):
    """
    Point target or origin specification.
    """

    # Target point in config units
    xyz: pint.Quantity = documented(
        pinttr.ib(units=ucc.deferred("length")),
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

    def kernel_item(self) -> t.Dict:
        """Return kernel item."""
        return self.xyz.m_as(uck.get("length"))


@parse_docs
@attr.s
class TargetRectangle(Target):
    """
    Rectangle target origin specification.

    This class defines an axis-aligned rectangular zone where ray targets will
    be sampled or ray origins will be projected.
    """

    xmin: pint.Quantity = documented(
        pinttr.ib(
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
        pinttr.ib(
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
        pinttr.ib(
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
        pinttr.ib(
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
        pinttr.ib(
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

    def kernel_item(self) -> t.Dict:
        """Return kernel item."""
        from mitsuba.core import ScalarTransform4f

        xmin = self.xmin.m_as(uck.get("length"))
        xmax = self.xmax.m_as(uck.get("length"))
        ymin = self.ymin.m_as(uck.get("length"))
        ymax = self.ymax.m_as(uck.get("length"))
        z = self.z.m_as(uck.get("length"))

        dx = xmax - xmin
        dy = ymax - ymin

        to_world = ScalarTransform4f.translate(
            [0.5 * dx + xmin, 0.5 * dy + ymin, z]
        ) * ScalarTransform4f.scale([0.5 * dx, 0.5 * dy, 1.0])

        return {"type": "rectangle", "to_world": to_world}
