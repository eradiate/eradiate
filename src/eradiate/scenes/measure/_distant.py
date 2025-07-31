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
from ..core import BoundingBox
from ... import converters, frame, validators
from ...attrs import define, documented
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


@attrs.define
class TargetPoint(Target):
    """
    Point target specification.

    Parameters
    ----------
    xyz : quantity or array-like
        Point coordinates. Unit-enabled field (default: ucc['length']).
    """

    # Target point in config units
    xyz: pint.Quantity = pinttr.field(units=ucc.deferred("length"))

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


@attrs.define(init=False)
class TargetRectangle(Target):
    """
    Rectangle target specification.

    This class defines a rectangular zone where ray targets will be sampled or
    ray origins will be projected. It supports several parametrizations:

    * bounds and altitude (``xmin``, ``xmax``, ``ymin``, ``ymax``, ``z``):
      in that case, the rectangle is axis-aligned;
    * centre position, edge lengths, normal vector and orientation
      (``xyz``, ``size_x``, ``size_y``, ``n``, ``up``): in that case, the
      rectangle is scaled and positioned using a look-at transformation;
    * geometric transform (``to_world``): in that case, a geometric
      transformation can be passed directly.

    Parameters
    ----------
    to_world : mi.ScalarTransform4f
        If this parametrization is used, ``to_world`` must be supplied in kernel
        units.

    xmin, xmax, ymin, ymax, z, size_x, size_y : float or quantity
        Unit-enabled (default: ucc['length']). ``z`` may be omitted (if so, it
        defaults to 0).

    xyz : array-like or quantity
        Unit-enabled (default: ucc['length']).

    n, up : array-like
    """

    to_world: "mi.ScalarTransform4f" = attrs.field()

    _bbox: BoundingBox = attrs.field(repr=False)

    def __init__(self, **kwargs):
        config_length = ucc.get("length")
        kernel_length = uck.get("length")
        bounds_kwargs = {"xmin", "xmax", "ymin", "ymax", "z"}
        bounds_kwargs_no_z = {"xmin", "xmax", "ymin", "ymax"}
        normal_kwargs = {"size_x", "size_y", "xyz", "n", "up"}
        transform_kwargs = {"to_world"}

        if set(kwargs) == bounds_kwargs or set(kwargs) == bounds_kwargs_no_z:
            xmin = ensure_units(kwargs["xmin"], default_units=config_length).m_as(
                kernel_length
            )
            xmax = ensure_units(kwargs["xmax"], default_units=config_length).m_as(
                kernel_length
            )
            ymin = ensure_units(kwargs["ymin"], default_units=config_length).m_as(
                kernel_length
            )
            ymax = ensure_units(kwargs["ymax"], default_units=config_length).m_as(
                kernel_length
            )
            z = ensure_units(kwargs.get("z", 0.0), default_units=config_length).m_as(
                kernel_length
            )
            dx = xmax - xmin
            dy = ymax - ymin

            translate = [0.5 * dx + xmin, 0.5 * dy + ymin, z]
            scale = [0.5 * dx, 0.5 * dy, 1.0]

            to_world = mi.ScalarTransform4f.translate(
                translate
            ) @ mi.ScalarTransform4f.scale(scale)

        elif set(kwargs) == normal_kwargs:
            dx = ensure_units(kwargs["size_x"], default_units=config_length).m_as(
                kernel_length
            )
            dy = ensure_units(kwargs["size_y"], default_units=config_length).m_as(
                kernel_length
            )
            origin = ensure_units(kwargs["xyz"], default_units=config_length).m_as(
                kernel_length
            )
            direction = kwargs["n"]
            up = kwargs["up"]
            scale = [0.5 * dx, 0.5 * dy, 1.0]

            to_world = mi.ScalarTransform4f.look_at(
                origin=mi.ScalarPoint3f(origin),
                target=mi.ScalarVector3f(origin + direction),
                up=mi.ScalarVector3f(up),
            ) @ mi.ScalarTransform4f.scale(scale)

        elif set(kwargs) == transform_kwargs:
            to_world = mi.ScalarTransform4f(kwargs["to_world"])

        else:
            raise TypeError(
                f"Unhandled keyword argument combination {set(kwargs)} "
                f"(allowed: {bounds_kwargs = }, {normal_kwargs = }, {transform_kwargs = }"
            )

        bbox = mi.BoundingBox3f()
        for p in [
            to_world.transform_affine(mi.ScalarPoint3f(*x))
            for x in [[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]]
        ]:
            bbox.expand(p)
        bbox = BoundingBox(
            (bbox.min.numpy() * kernel_length).to(config_length),
            (bbox.max.numpy() * kernel_length).to(config_length),
        )

        self.__attrs_init__(to_world=to_world, bbox=bbox)

    @property
    def bbox(self) -> BoundingBox:
        """Bounding (in configuration units)."""
        return self._bbox

    @property
    def xmin(self) -> pint.Quantity:
        """
        .. deprecated:: 1.0.0

        Alias to ``self.bbox.min[0]`` (for compatibility).
        """
        return self.bbox.min[0]

    @property
    def xmax(self) -> pint.Quantity:
        """
        .. deprecated:: 1.0.0

        Alias to ``self.bbox.max[0]`` (for compatibility).
        """
        return self.bbox.max[0]

    @property
    def ymin(self) -> pint.Quantity:
        """
        .. deprecated:: 1.0.0

        Alias to ``self.bbox.min[1]`` (for compatibility).
        """
        return self.bbox.min[1]

    @property
    def ymax(self) -> pint.Quantity:
        """
        .. deprecated:: 1.0.0

        Alias to ``self.bbox.max[1]`` (for compatibility).
        """
        return self.bbox.max[1]

    @property
    def z(self) -> pint.Quantity:
        """
        .. deprecated:: 1.0.0

        Alias to ``0.5 * (self.bbox.min[2] + self.bbox.max[2])`` (for compatibility).
        """
        return 0.5 * (self.bbox.min[2] + self.bbox.max[2])

    def kernel_item(self) -> dict:
        # Inherit docstring

        return {"type": "rectangle", "to_world": self.to_world}


# ------------------------------------------------------------------------------
#                             Distant measure interface
# ------------------------------------------------------------------------------


@define(eq=False, slots=False)
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


@define(eq=False, slots=False)
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


@define(eq=False, slots=False)
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
        azimuth_convention = kwargs.get("azimuth_convention", None)
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
