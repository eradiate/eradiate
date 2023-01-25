from __future__ import annotations

import typing as t

import attrs
import numpy as np
import pint
import pinttr

from ._distant import DistantMeasure, Target, TargetPoint, TargetDisk
from ._helpers import AngleLayout, AzimuthRingLayout, DirectionLayout, GridLayout, HemispherePlaneLayout, Layout
from ...attrs import documented, parse_docs
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...util.deprecation import deprecated

# ------------------------------------------------------------------------------
#                       MultiDistantMeasure implementation
# ------------------------------------------------------------------------------


def _extract_kwargs(kwargs: dict, keys: t.List[str]) -> dict:
    # Helper function to collect common layout keyword arguments
    # (mutates the param dictionary)
    # Used in MultiDistantMeasure constructors
    return {key: kwargs.pop(key) for key in keys if key in kwargs}


@parse_docs
@attrs.define
class MultiDistantMeasureGonio(DistantMeasure):
    """
    Multi-distant radiance measure scene element, goniometer variant [``distant_gonio``, ``mdistant_gonio``, \
    ``multi_distant_gonio``].

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
    * Setting the ``target`` and ``target_radius`` parameters is required to get meaningful results.
      Experiment classes should take care of setting it appropriately.
    * This measure defines its target by a point in space and a radius. Rays will be sampled to hit a disk perpendicular
      to their direction at the target point and with target radius.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    target: t.Optional[Target] = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(Target.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(
                    (
                        TargetPoint,
                    )
                )
            ),
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
        type=":class:`.TargetPoint` or None",
        init_type=":class:`.TargetPoint` or dict or array-like, optional",
    )

    target_radius: pint.Quantity = documented(
        pinttr.field(
            factory=lambda: 1.0 * ucc.get("length"), units=ucc.deferred("length")
        ),
        doc='Radius of the targeted region. Unit-enabled field (default: ``ucc["length"]``).',
        type="quantity",
        init_type="quantity or float, optional",
        default="1.0",
    )

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
    def film_resolution(self) -> t.Tuple[int, int]:
        # Inherit docstring
        return (self.direction_layout.n_directions, 1)

    # --------------------------------------------------------------------------
    #                         Additional constructors
    # --------------------------------------------------------------------------

    @classmethod
    def hplane(
        cls,
        zeniths: np.typing.ArrayLike,
        azimuth: t.Union[float, pint.Quantity],
        **kwargs,
    ) -> MultiDistantMeasureGonio:
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
        zenith: t.Union[float, pint.Quantity],
        azimuths: np.typing.ArrayLike,
        **kwargs,
    ) -> MultiDistantMeasureGonio:
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

    # TODO: MultiDistantGonio should accept only a point target and additionally have a field of view!!!

    @classmethod
    def grid(
        cls, zeniths: np.typing.ArrayLike, azimuths: np.typing.ArrayLike, **kwargs
    ) -> MultiDistantMeasureGonio:
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
    def from_angles(cls, angles: np.typing.ArrayLike, **kwargs) -> MultiDistantMeasureGonio:
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
    ) -> MultiDistantMeasureGonio:
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

    @classmethod
    @deprecated(
        deprecated_in="0.22.6",
        removed_in="0.23.1",
        details="Transition to using the direction layout or other class method "
        "constructors.",
    )
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
        .MultiDistantMeasure
        """
        if "directions" in kwargs:
            raise TypeError(
                "from_viewing_angles() got an unexpected keyword argument 'directions'"
            )

        if "hplane" in kwargs:
            raise TypeError(
                "from_viewing_angles() got an unexpected keyword argument 'hplane'"
            )

        # Basic unit conversion and array reshaping
        angle_units = ucc.get("angle")
        zeniths = pinttr.util.ensure_units(
            np.atleast_1d(zeniths).reshape((-1, 1)), default_units=angle_units
        ).m_as(angle_units)
        azimuths = pinttr.util.ensure_units(
            np.atleast_1d(azimuths).reshape((-1, 1)), default_units=angle_units
        ).m_as(angle_units)

        # Detect layout
        if len(zeniths) == 1 and len(azimuths) != 1:
            layout = AzimuthRingLayout(
                zenith=zeniths[0],
                azimuths=azimuths,
                **_extract_kwargs(kwargs, ["azimuth_convention"]),
            )

        elif len(zeniths) != 1 and len(azimuths) == 1 and auto_hplane:
            layout = HemispherePlaneLayout(
                zeniths=zeniths,
                azimuth=azimuths[0],
                **_extract_kwargs(kwargs, ["azimuth_convention"]),
            )

        else:
            layout = AngleLayout(
                angles=np.hstack((zeniths, azimuths)),
                **_extract_kwargs(kwargs, ["azimuth_convention"]),
            )

        # Create instance
        return cls(direction_layout=layout, **kwargs)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _kernel_dict_impl(self, sensor_id, spp):
        result = {
            "type": "mdistant_gonio",
            "id": sensor_id,
            "directions": ",".join(
                map(str, -self.direction_layout.directions.ravel(order="C"))
            ),
            "sampler": {
                "type": self.sampler,
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
            target = TargetDisk(center=self.target.xyz, radius=self.target_radius)
            result["target"] = target.kernel_item()

        if self.ray_offset is not None:
            result["ray_offset"] = self.ray_offset.m_as(uck.get("length"))

        return result

    # --------------------------------------------------------------------------
    #                        Post-processing information
    # --------------------------------------------------------------------------

    @property
    def var(self) -> t.Tuple[str, t.Dict]:
        # Inherit docstring
        return "radiance", {
            "standard_name": "radiance",
            "long_name": "radiance",
            "units": symbol(uck.get("radiance")),
        }
