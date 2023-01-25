from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import attrs
import numpy as np
import pint
import pinttr

from ... import converters, frame
from ..._config import config
from ...attrs import documented, parse_docs
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg

# ------------------------------------------------------------------------------
#                               Layout framework
# ------------------------------------------------------------------------------


@parse_docs
@attrs.define
class Layout(ABC):
    """
    Abstract base class for all viewing direction layouts.
    """

    azimuth_convention: frame.AzimuthConvention = documented(
        attrs.field(
            default=None,
            converter=lambda x: config.azimuth_convention
            if x is None
            else (frame.AzimuthConvention[x.upper()] if isinstance(x, str) else x),
            validator=attrs.validators.instance_of(frame.AzimuthConvention),
            kw_only=True,
        ),
        doc="Azimuth convention used by this layout. If ``None``, the global "
        "default configuration is used (see :class:`.EradiateConfig`).",
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


@parse_docs
@attrs.define
class AngleLayout(Layout):
    """
    A viewing direction layout directly defined by explicit (zenith, azimuth)
    pairs.
    """

    _angles: pint.Quantity = documented(
        pinttr.ib(
            converter=lambda x: np.reshape(
                pinttr.converters.to_units(ucc.deferred("angle"))(x), (-1, 2)
            )
            % (360.0 * ureg.deg),
            units=ucc.deferred("angle"),
        ),
        doc="A sequence of viewing angles, corresponding to the direction "
        "sequence produced by :attr:`directions`, as a (N, 2) array. "
        "The last dimension is ordered as (zenith, azimuth). "
        "**Required, no default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['angle']).",
        type="quantity",
        init_type="array-like",
    )

    @property
    def angles(self) -> pint.Quantity:
        # Inherit docstring
        return self._angles


@parse_docs
@attrs.define
class AzimuthRingLayout(Layout):
    """
    A viewing direction layout defined by a single zenith and a vector of
    explicit azimuth values.
    """

    zenith: pint.Quantity = documented(
        pinttr.field(
            converter=lambda x: converters.on_quantity(float)(
                pinttr.converters.to_units(ucc.deferred("angle"))(x)
            ),
            units=ucc.deferred("angle"),
        ),
        doc="A single zenith value. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['angle']).",
        type="quantity",
        init_type="float or quantity",
    )

    azimuths: pint.Quantity = documented(
        pinttr.field(
            converter=lambda x: np.reshape(
                pinttr.converters.to_units(ucc.deferred("angle"))(x), (-1,)
            )
            % (360.0 * ureg.deg),
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


@parse_docs
@attrs.define
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
        ).to(
            ucc.get("angle")
        )  # Convert to default angle units

    @property
    def n_directions(self) -> int:
        # Inherit docstring
        return len(self._directions)

    @property
    def directions(self) -> np.narray:
        # Inherit docstring
        return self._directions


@parse_docs
@attrs.define
class HemispherePlaneLayout(Layout):
    """
    A viewing direction layout defined by a single azimuth and a vector of
    zenith values. Negative zenith values are mapped to (azimuth + 180°).
    """

    zeniths: pint.Quantity = documented(
        pinttr.field(
            converter=lambda x: np.reshape(
                pinttr.converters.to_units(ucc.deferred("angle"))(x), (-1,)
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
        pinttr.field(units=ucc.deferred("angle")),
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


@parse_docs
@attrs.define
class GridLayout(Layout):
    """
    A viewing direction layout defined as the Cartesian product of an azimuth
    and zenith vectors.
    """

    zeniths: pint.Quantity = documented(
        pinttr.field(
            converter=lambda x: np.reshape(
                pinttr.converters.to_units(ucc.deferred("angle"))(x), (-1,)
            ),
            units=ucc.deferred("angle"),
        ),
        doc="A vector of zenith values. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['angle']).",
        type="quantity",
        init_type="array-like",
    )

    azimuths: pint.Quantity = documented(
        pinttr.field(
            converter=lambda x: np.reshape(
                pinttr.converters.to_units(ucc.deferred("angle"))(x), (-1,)
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
