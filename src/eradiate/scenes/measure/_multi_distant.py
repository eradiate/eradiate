from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import attrs
import numpy as np
import pint
import pinttr

from ._distant import AbstractDistantMeasure
from ... import converters, frame
from ...attrs import define, documented
from ...config import settings
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...util.misc import flatten

# ------------------------------------------------------------------------------
#                               Layout framework
# ------------------------------------------------------------------------------


@define
class Layout(ABC):
    """
    Abstract base class for all viewing direction layouts.
    """

    azimuth_convention: frame.AzimuthConvention = documented(
        attrs.field(
            default=None,
            converter=lambda x: (
                settings.azimuth_convention
                if x is None
                else frame.AzimuthConvention.convert(x)
            ),
            validator=attrs.validators.instance_of(frame.AzimuthConvention),
            kw_only=True,
        ),
        doc="Azimuth convention used by this layout. If ``None``, the global "
        "default configuration is used (see :ref:`sec-user_guide-config`).",
        type=".AzimuthConvention",
        init_type=".AzimuthConvention or str, optional",
        default="None",
    )

    @staticmethod
    def convert(value: t.Any) -> t.Any:
        """
        Attempt to instantiate a :class:`Layout` concrete class from an object.

        This conversion protocol accepts:

        * a dictionary of the form ``{"type": type_name, **kwargs}``;
        * a (N, 2)-array or a (2,)-array;
        * a (N, 3)-array or a (3,)-array.

        Other values pass through the converter.

        Dictionaries have their parameters forwarded to the type selected by the
        ``type`` parameter.
        A (N, 2) or (2,)-array is passed to an :class:`.AngleLayout`.
        A (N, 3) or (3,)-array is passed to a :class:`.DirectionLayout`.

        .. list-table::
           :header-rows: 1

           * - Type key
             - Class
           * - angles
             - :class:`.AngleLayout`
           * - aring
             - :class:`.AzimuthRingLayout`
           * - directions
             - :class:`.DirectionLayout`
           * - grid
             - :class:`.GridLayout`
           * - hplane
             - :class:`.HemispherePlaneLayout`
        """
        if isinstance(value, Layout):
            return value

        if isinstance(value, dict):
            d = pinttr.interpret_units(value, ureg=ureg)
            type_key = d.pop("type")
            cls = {
                "angles": AngleLayout,
                "aring": AzimuthRingLayout,
                "directions": DirectionLayout,
                "grid": GridLayout,
                "hplane": HemispherePlaneLayout,
            }[type_key]
            return cls(**d)

        if np.ndim(value) == 2:
            if np.shape(value)[1] == 2:
                return AngleLayout(angles=value)
            if np.shape(value)[1] == 3:
                return DirectionLayout(directions=value)

        if np.ndim(value) == 1:
            if np.shape(value) == (2,):
                return AngleLayout(angles=value)
            if np.shape(value) == (3,):
                return DirectionLayout(directions=value)

        return value

    @property
    def n_directions(self) -> int:
        """
        int: Number of viewing directions defined by this layout.
        """
        return len(self.angles)

    @property
    @abstractmethod
    def angles(self) -> pint.Quantity:
        """
        quantity: A sequence of viewing angles, corresponding to the direction
            sequence produced by :attr:`directions`, as a (N, 2) array.
            The last dimension is ordered as (zenith, azimuth).
        """
        pass

    @property
    def directions(self) -> np.narray:
        """
        ndarray: A sequence of viewing directions, pointing *outwards* the
            observed target, as a (N, 3) array.
        """
        # Default implementation computes directions from angles
        return frame.angles_to_direction(
            self.angles, azimuth_convention=self.azimuth_convention
        )


def _angles_converter(value):
    value = pinttr.util.ensure_units(value, ucc.deferred("angle"))
    angle_units = value.u
    magnitude = np.reshape(value.m_as(ureg.deg), (-1, 2))
    zeniths = magnitude[:, 0]
    azimuths = magnitude[:, 1]
    return (np.stack((zeniths, azimuths % 360), axis=1) * ureg.deg).to(angle_units)


@define
class AngleLayout(Layout):
    """
    A viewing direction layout directly defined by explicit (zenith, azimuth)
    pairs.
    """

    _angles: pint.Quantity = documented(
        pinttr.ib(
            converter=_angles_converter,
            units=ucc.deferred("angle"),
        ),
        doc="A sequence of viewing angles, corresponding to the direction "
        "sequence produced by :attr:`directions`, as a (N, 2) array. "
        "The last dimension is ordered as (zenith, azimuth). Zenith values "
        "must be between 0 and 180°. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['angle']).",
        type="quantity",
        init_type="array-like",
    )

    @_angles.validator
    def _angles_validator(self, attribute, value):
        zeniths = value[:, 0].m_as(ureg.deg)
        if np.any((zeniths < 0) | (zeniths > 180)):
            raise ValueError(
                f"while validating {attribute.name}: zenith values must be in "
                "[0°, 180°]"
            )

    @property
    def angles(self) -> pint.Quantity:
        # Inherit docstring
        return self._angles


@define
class AzimuthRingLayout(Layout):
    """
    A viewing direction layout defined by a single zenith and a vector of
    explicit azimuth values.
    """

    zenith: pint.Quantity = documented(
        pinttr.field(
            converter=attrs.converters.pipe(
                pinttr.converters.to_units(ucc.deferred("angle")),
                lambda x: converters.on_quantity(float)(x),
            ),
            units=ucc.deferred("angle"),
        ),
        doc="A single zenith value. Must be in [0°, 180°]. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['angle']).",
        type="quantity",
        init_type="float or quantity",
    )

    @zenith.validator
    def _zenith_validator(self, attribute, value):
        zenith = value.m_as(ureg.deg)
        if zenith < 0 or zenith > 180:
            raise ValueError(
                f"while validating {attribute.name}: zenith value must be in [0°, 180°]"
            )

    azimuths: pint.Quantity = documented(
        pinttr.field(
            converter=attrs.converters.pipe(
                pinttr.converters.to_units(ucc.deferred("angle")),
                lambda x: np.reshape(x, (-1,)),
                lambda x: x % (360.0 * ureg.deg),
            ),
            units=ucc.deferred("angle"),
        ),
        doc="A vector of azimuth values. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['angle']).",
        type="quantity",
        init_type="array-like",
    )

    @property
    def angles(self) -> pint.Quantity:
        # Inherit docstring
        # Basic unit conversion and broadcasting
        angle_units = ucc.get("angle")
        azimuths = np.reshape(self.azimuths.m_as(angle_units), (-1, 1))
        zeniths = np.full_like(azimuths, self.zenith.m_as(angle_units))

        # Assemble angles
        return np.hstack((zeniths, azimuths)) * angle_units


@define
class DirectionLayout(Layout):
    """
    A viewing direction layout directly defined by explicit (zenith, azimuth)
    pairs.
    """

    _directions: np.ndarray = documented(
        attrs.field(converter=lambda x: np.reshape(x, (-1, 3))),
        doc="A sequence of 3-vectors specifying distant sensing directions. "
        "Note that directions point outward the target. **Required, no default**.",
        type="ndarray",
        init_type="array-like",
    )

    @property
    def angles(self) -> pint.Quantity:
        # Inherit docstring

        return frame.direction_to_angles(
            self.directions,
            azimuth_convention=self.azimuth_convention,
            normalize=True,
        ).to(ucc.get("angle"))  # Convert to default angle units

    @property
    def n_directions(self) -> int:
        # Inherit docstring
        return len(self._directions)

    @property
    def directions(self) -> np.narray:
        # Inherit docstring
        return self._directions


@define
class HemispherePlaneLayout(Layout):
    """
    A viewing direction layout defined by a single azimuth and a vector of
    zenith values. Negative zenith values are mapped to (azimuth + 180°).
    """

    zeniths: pint.Quantity = documented(
        pinttr.field(
            converter=attrs.converters.pipe(
                pinttr.converters.to_units(ucc.deferred("angle")),
                lambda x: np.reshape(x, (-1,)),
            ),
            units=ucc.deferred("angle"),
        ),
        doc="A vector of zenith values. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['angle']).",
        type="quantity",
        init_type="array-like",
    )

    azimuth: pint.Quantity = documented(
        pinttr.field(
            converter=attrs.converters.pipe(
                pinttr.converters.to_units(ucc.deferred("angle")),
                lambda x: x % (360 * ureg.deg),
            ),
            units=ucc.deferred("angle"),
        ),
        doc="A single zenith value. **Required, no default**.",
        type="quantity",
        init_type="float or quantity",
    )

    @property
    def angles(self) -> pint.Quantity:
        # Inherit docstring
        # Basic unit conversion and broadcasting
        angle_units = ucc.get("angle")
        zeniths = np.reshape(self.zeniths.m_as(angle_units), (-1, 1))
        azimuths = np.full_like(zeniths, self.azimuth.m_as(angle_units))

        # Assemble angles
        return np.hstack((zeniths, azimuths)) * angle_units


@define
class GridLayout(Layout):
    """
    A viewing direction layout defined as the Cartesian product of an azimuth
    and zenith vectors.
    """

    zeniths: pint.Quantity = documented(
        pinttr.field(
            converter=attrs.converters.pipe(
                pinttr.converters.to_units(ucc.deferred("angle")),
                lambda x: np.reshape(x, (-1,)),
            ),
            units=ucc.deferred("angle"),
        ),
        doc="A vector of zenith values. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['angle']).",
        type="quantity",
        init_type="array-like",
    )

    @zeniths.validator
    def _zeniths_validator(self, attribute, value):
        zeniths = value.m_as(ureg.deg)
        if np.any((zeniths < 0) | (zeniths > 180)):
            raise ValueError(
                f"while validating {attribute.name}: zenith values must be in "
                "[0°, 180°]"
            )

    azimuths: pint.Quantity = documented(
        pinttr.field(
            converter=attrs.converters.pipe(
                pinttr.converters.to_units(ucc.deferred("angle")),
                lambda x: np.reshape(x, (-1,)),
            ),
            units=ucc.deferred("angle"),
        ),
        doc="A vector of azimuth values. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['angle']).",
        type="quantity",
        init_type="array-like",
    )

    @property
    def angles(self) -> pint.Quantity:
        # Inherit docstring
        #  Basic unit conversion and broadcasting
        angle_units = ucc.get("angle")
        zeniths = self.zeniths.m_as(angle_units)
        azimuths = self.azimuths.m_as(angle_units)

        # Assemble angles
        # This effectively produces the Cartesian product of the zeniths and
        # azimuths arrays (see https://stackoverflow.com/a/11146645/3645374)
        return np.dstack(np.meshgrid(zeniths, azimuths)).reshape(-1, 2) * angle_units


# ------------------------------------------------------------------------------
#                       MultiDistantMeasure implementation
# ------------------------------------------------------------------------------


def _extract_kwargs(kwargs: dict, keys: list[str]) -> dict:
    # Helper function to collect common layout keyword arguments
    # (mutates the param dictionary)
    # Used in MultiDistantMeasure constructors
    return {key: kwargs.pop(key) for key in keys if key in kwargs}


@define(eq=False, slots=False)
class MultiDistantMeasure(AbstractDistantMeasure):
    """
    Multi-distant radiance measure scene element [``mdistant``, ``multi_distant``].

    This scene element creates a measure consisting of an array of
    radiancemeters positioned at an infinite distance from the scene. In
    practice, it can be used to compute the radiance leaving a scene at the
    top of the atmosphere (or canopy if there is no atmosphere). Coupled to
    appropriate post-processing operations, scene reflectance can be derived
    from the radiance values it produces.

    .. admonition:: Class method constructors

       .. autosummary::

          aring
          grid
          hplane
          from_angles
          from_directions

    Notes
    -----
    * Setting the ``target`` parameter is required to get meaningful results.
      Experiment classes should take care of setting it appropriately.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    direction_layout: Layout = documented(
        attrs.field(
            kw_only=True,
            factory=lambda: DirectionLayout(directions=[0, 0, 1]),
            converter=Layout.convert,
            validator=attrs.validators.instance_of(Layout),
        ),
        doc="A viewing direction layout. Specification through a dictionary or "
        "arrays, as documented by :meth:`Layout.convert`, is also possible. "
        "The constructor methods provide a convenient interface to configure "
        "this parameter automatically.",
        type=".Layout",
        init_type="dict or array-like or .Layout, optional",
        default="DirectionLayout(directions=[0, 0, 1])",
    )

    @property
    def viewing_angles(self) -> pint.Quantity:
        """
        quantity: Viewing angles computed from stored `directions` as a
            (N, 1, 2) array, where N is the number of directions. The last
            dimension is ordered as (zenith, azimuth).
        """
        # Note: The middle dimension in (N, 1, 2) is the film height
        return self.direction_layout.angles.reshape(-1, 1, 2)

    @property
    def film_resolution(self) -> tuple[int, int]:
        # Inherit docstring
        return (self.direction_layout.n_directions, 1)

    # --------------------------------------------------------------------------
    #                         Additional constructors
    # --------------------------------------------------------------------------

    @classmethod
    def hplane(
        cls,
        zeniths: np.typing.ArrayLike,
        azimuth: float | pint.Quantity,
        **kwargs,
    ) -> MultiDistantMeasure:
        """
        Construct using a hemisphere plane cut viewing direction layout.

        Parameters
        ----------
        zeniths : array-like
            List of zenith values. Negative values are mapped to the
            `azimuth + 180°` half-plane. Unitless values are converted to
            ``ucc['angle']``.

        azimuth : float or quantity
            Hemisphere plane cut azimuth value. Unitless values are converted to
            ``ucc['angle']``.

        azimuth_convention : .AzimuthConvention or str, optional
            The azimuth convention applying to the viewing direction layout.
            If unset, the global default convention is used.

        **kwargs
            Remaining keyword arguments are forwarded to the
            :class:`.MultiDistantMeasure` constructor.

        Returns
        -------
        MultiDistantMeasure
        """
        layout = HemispherePlaneLayout(
            zeniths=zeniths,
            azimuth=azimuth,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)

    @classmethod
    def aring(
        cls,
        zenith: float | pint.Quantity,
        azimuths: np.typing.ArrayLike,
        **kwargs,
    ) -> MultiDistantMeasure:
        """
        Construct using an azimuth ring viewing direction layout.

        Parameters
        ----------
        zenith : float or quantity
            Azimuth ring zenith value. Unitless values are converted to
            ``ucc['angle']``.

        azimuths : array-like
            List of azimuth values. Unitless values are converted to
            ``ucc['angle']``.

        azimuth_convention : .AzimuthConvention or str, optional
            The azimuth convention applying to the viewing direction layout.
            If unset, the global default convention is used.

        **kwargs
            Remaining keyword arguments are forwarded to the
            :class:`.MultiDistantMeasure` constructor.

        Returns
        -------
        MultiDistantMeasure
        """
        layout = AzimuthRingLayout(
            zenith=zenith,
            azimuths=azimuths,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)

    @classmethod
    def grid(
        cls, zeniths: np.typing.ArrayLike, azimuths: np.typing.ArrayLike, **kwargs
    ) -> MultiDistantMeasure:
        """
        Construct using a gridded viewing direction layout, defined as the
        Cartesian product of zenith and azimuth arrays.

        Parameters
        ----------
        azimuths : array-like
            List of azimuth values.

        zeniths : array-like
            List of zenith values.

        azimuth_convention : .AzimuthConvention or str, optional
            The azimuth convention applying to the viewing direction layout.
            If unset, the global default convention is used.

        **kwargs
            Remaining keyword arguments are forwarded to the
            :class:`.MultiDistantMeasure` constructor.

        Returns
        -------
        MultiDistantMeasure
        """
        layout = GridLayout(
            zeniths=zeniths,
            azimuths=azimuths,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)

    @classmethod
    def from_angles(cls, angles: np.typing.ArrayLike, **kwargs) -> MultiDistantMeasure:
        """
        Construct using a direction layout defined by explicit (zenith, azimuth)
        pairs.

        Parameters
        ----------
        angles : array-like
            A sequence of (zenith, azimuth), interpreted as (N, 2)-shaped array.

        azimuth_convention : .AzimuthConvention or str, optional
            The azimuth convention applying to the viewing direction layout.
            If unset, the global default convention is used.

        **kwargs
            Remaining keyword arguments are forwarded to the
            :class:`.MultiDistantMeasure` constructor.

        Returns
        -------
        MultiDistantMeasure
        """
        layout = AngleLayout(
            angles=angles,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)

    @classmethod
    def from_directions(
        cls, directions: np.typing.ArrayLike, **kwargs
    ) -> MultiDistantMeasure:
        """
        Construct using a direction layout defined by explicit direction
        vectors.

        Parameters
        ----------
        directions : array-like
            A sequence of direction vectors, interpreted as (N, 3)-shaped array.

        azimuth_convention : .AzimuthConvention or str, optional
            The azimuth convention applying to the viewing direction layout.
            If unset, the global default convention is used.

        **kwargs
            Remaining keyword arguments are forwarded to the
            :class:`.MultiDistantMeasure` constructor.

        Returns
        -------
        MultiDistantMeasure

        Warnings
        --------
        Viewing directions are defined pointing *outwards* the target location.
        """
        layout = DirectionLayout(
            directions=directions,
            **_extract_kwargs(kwargs, ["azimuth_convention"]),
        )
        return cls(direction_layout=layout, **kwargs)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def kernel_type(self) -> str:
        return "mdistant"

    @property
    def template(self) -> dict:
        result = super().template
        result["directions"] = ",".join(
            map(str, -self.direction_layout.directions.ravel(order="C"))
        )

        if self.target is not None:
            result.update(flatten({"target": self.target.kernel_item()}))

        if self.ray_offset is not None:
            result["ray_offset"] = self.ray_offset.m_as(uck.get("length"))

        return result

    # --------------------------------------------------------------------------
    #                        Post-processing information
    # --------------------------------------------------------------------------

    @property
    def var(self) -> tuple[str, dict]:
        # Inherit docstring
        return "radiance", {
            "standard_name": "radiance",
            "long_name": "radiance",
            "units": symbol(uck.get("radiance")),
        }
