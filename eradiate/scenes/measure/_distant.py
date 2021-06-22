from copy import deepcopy
from typing import Dict, List, Tuple

import attr
import numpy as np
import pinttr
import xarray as xr

import eradiate

from ._core import Measure, MeasureFactory
from ..illumination import DirectionalIllumination
from ... import converters, validators
from ..._attrs import documented, parse_docs
from ..._util import is_vector3
from ...frame import direction_to_angles
from ...units import symbol
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...warp import square_to_uniform_hemisphere


@attr.s
class TargetOrigin:
    """
    Interface for target and origin selection classes used by :class:`DistantMeasure`.
    """

    def kernel_item(self):
        """Return kernel item."""
        raise NotImplementedError

    @staticmethod
    def new(target_type, *args, **kwargs):
        """
        Instantiate one of the supported child classes. This factory requires
        manual class registration. All position and keyword arguments are
        forwarded to the constructed type.

        Currently supported classes:

        * ``point``: :class:`TargetOriginPoint`
        * ``rectangle``: :class:`TargetOriginRectangle`
        * ``sphere``: :class:`TargetOriginSphere`

        Parameter ``target_type`` (str):
            Identifier of one of the supported child classes.
        """
        if target_type == "point":
            return TargetOriginPoint(*args, **kwargs)
        elif target_type == "rectangle":
            return TargetOriginRectangle(*args, **kwargs)
        elif target_type == "sphere":
            return TargetOriginSphere(*args, **kwargs)
        else:
            raise ValueError(f"unknown target type {target_type}")

    @staticmethod
    def convert(value):
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`new` to
        instantiate a :class:`Target` child class based on the ``"type"`` entry
        it contains.

        If ``value`` is a 3-vector, this method returns a :class:`TargetPoint`
        instance.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            d = deepcopy(value)
            try:
                target_type = d.pop("type")
            except KeyError:
                raise ValueError("cannot convert dict, missing 'type' entry")

            return TargetOrigin.new(target_type, **d)

        if is_vector3(value):
            return TargetOrigin.new("point", xyz=value)

        return value


def _target_point_rectangle_xyz_converter(x):
    return converters.on_quantity(float)(
        pinttr.converters.to_units(ucc.deferred("length"))(x)
    )


@parse_docs
@attr.s
class TargetOriginPoint(TargetOrigin):
    """
    Point target or origin specification.
    """

    # Target point in config units
    xyz = documented(
        pinttr.ib(units=ucc.deferred("length")),
        doc="Point coordinates.\n\nUnit-enabled field (default: cdu[length]).",
        type="array-like",
    )

    @xyz.validator
    def _xyz_validator(self, attribute, value):
        if not is_vector3(value):
            raise ValueError(
                f"while validating {attribute.name}: must be a "
                f"3-element vector of numbers"
            )

    def kernel_item(self):
        """Return kernel item."""
        return self.xyz.m_as(uck.get("length"))


@parse_docs
@attr.s
class TargetOriginRectangle(TargetOrigin):
    """
    Rectangle target origin specification.

    This class defines an axis-aligned rectangular zone where ray targets will
    be sampled or ray origins will be projected.
    """

    xmin = documented(
        pinttr.ib(
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Lower bound on the X axis.\n"
        "\n"
        "Unit-enabled field (default: cdu[length]).",
        type="float",
    )

    xmax = documented(
        pinttr.ib(
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Upper bound on the X axis.\n"
        "\n"
        "Unit-enabled field (default: cdu[length]).",
        type="float",
    )

    ymin = documented(
        pinttr.ib(
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Lower bound on the Y axis.\n"
        "\n"
        "Unit-enabled field (default: cdu[length]).",
        type="float",
    )

    ymax = documented(
        pinttr.ib(
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Upper bound on the Y axis.\n"
        "\n"
        "Unit-enabled field (default: cdu[length]).",
        type="float",
    )

    z = documented(
        pinttr.ib(
            default=0.0,
            converter=_target_point_rectangle_xyz_converter,
            units=ucc.deferred("length"),
        ),
        doc="Altitude of the plane enclosing the rectangle.\n"
        "\n"
        "Unit-enabled field (default: cdu[length]).",
        type="float",
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

    def kernel_item(self):
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


@parse_docs
@attr.s
class TargetOriginSphere(TargetOrigin):
    """
    Sphere target or origin specification.
    """

    center = documented(
        pinttr.ib(units=ucc.deferred("length")),
        doc="Center coordinates.\n" "\n" "Unit-enabled field (default: cdu[length]).",
        type="array-like",
    )

    @center.validator
    def _center_validator(self, attribute, value):
        if not is_vector3(value):
            raise ValueError(
                f"while validating {attribute.name}: must be a "
                f"3-element vector of numbers"
            )

    radius = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            validator=[pinttr.validators.has_compatible_units, validators.is_positive],
        ),
        doc="Sphere radius.\n\nUnit-enabled field (default: cdu[length]).",
        type="float",
    )

    def kernel_item(self):
        """Return kernel item."""
        center = self.center.m_as(uck.get("length"))
        radius = self.radius.m_as(uck.get("length"))

        return {"type": "sphere", "center": center, "radius": radius}


@MeasureFactory.register("distant")
@parse_docs
@attr.s
class DistantRadianceMeasure(DistantMeasure):
    """
    Distant measure scene element [:factorykey:`distant`].

    This measure records radiance leaving the scene at infinite distance.
    Depending on film resolution (*i.e.* storage discretisation), radiance is
    recorded for a single direction, in a plane or in an entire hemisphere.

    When used with a backward tracing algorithm, rays traced by the sensor
    target a shape which can be controlled through the ``target`` parameter.
    This feature is useful if one wants to compute the average radiance leaving
    a particular subset of the scene.

    .. note:: This scene element is a thin wrapper around the ``distant`` sensor
       kernel plugin.
    """

    _film_resolution = documented(
        attr.ib(
            default=(32, 32),
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(int),
                iterable_validator=validators.has_len(2),
            ),
        ),
        doc="Film resolution as a (width, height) 2-tuple. "
        "If the height is set to 1, direction sampling will be restricted to a "
        "plane.",
        type="array-like",
        default="(32, 32)",
    )

    target = documented(
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
        doc="Target specification. If set to ``None``, default target point "
        "selection is used: rays will not target a particular region of the "
        "scene. The target can be specified using an array-like with 3 "
        "elements (which will be converted to a :class:`TargetPoint`) or a "
        "dictionary interpreted by :meth:`Target.convert`.",
        type=":class:`TargetOrigin` or None",
        default="None",
    )

    origin = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(TargetOrigin.convert),
            validator=attr.validators.optional(
                attr.validators.instance_of((TargetOriginSphere,))
            ),
            on_setattr=attr.setters.pipe(attr.setters.convert, attr.setters.validate),
        ),
        doc="Ray origin specification. If set to ``None``, the default origin "
        "point selection strategy is used: ray origins will be projected to "
        "the scene's bounding sphere. Otherwise, ray origins are projected "
        "to the shape specified as origin. The origin can be specified using "
        "a dictionary interpreted by :meth:`TargetOrigin.convert`.",
        type=":class:`TargetOriginSphere` or None",
        default="None",
    )

    orientation = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.deg),
            validator=validators.is_positive,
            units=ucc.deferred("angle"),
        ),
        doc="Azimuth angle defining the orientation of the sensor in the "
        "horizontal plane.\n"
        "\n"
        "Unit-enabled field (default: cdu[angle]).",
        type="float",
        default="0.0 deg",
    )

    direction = documented(
        attr.ib(
            default=[0, 0, 1],
            converter=np.array,
            validator=validators.is_vector3,
        ),
        doc="A 3-vector orienting the hemisphere mapped by the measure.",
        type="array-like",
        default="[0, 0, 1]",
    )

    flip_directions = documented(
        attr.ib(default=None, converter=attr.converters.optional(bool)),
        doc=" If ``True``, sampled directions will be flipped.",
        type="bool",
        default="False",
    )

    @property
    def film_resolution(self):
        return self._film_resolution

    def _base_dicts(self) -> List[Dict]:
        result = []

        for sensor_info in self.sensor_infos():
            d = {
                "type": "distant",
                "id": sensor_info.id,
                "direction": self.direction,
                "orientation": [
                    np.cos(self.orientation.m_as(ureg.rad)),
                    np.sin(self.orientation.m_as(ureg.rad)),
                    0.0,
                ],
            }

            if self.target is not None:
                d["ray_target"] = self.target.kernel_item()

            if self.origin is not None:
                d["ray_origin"] = self.origin.kernel_item()

            if self.flip_directions is not None:
                d["flip_directions"] = self.flip_directions

            result.append(d)

        return result

    def postprocess(self) -> xr.Dataset:
        # Fetch results (SPP-split aggregated) as a Dataset
        result = self._postprocess_fetch_results()

        # Attach viewing angle coordinates
        result = self._postprocess_add_viewing_angles(result)

        return result

    def _postprocess_fetch_results(self) -> xr.Dataset:
        # Collect results and add appropriate metadata

        ds = self.results.to_dataset(aggregate_spps=True)

        # Rename raw field to outgoing radiance
        ds = ds.rename(raw="lo")
        ds["lo"].attrs = {
            "standard_name": "toa_outgoing_radiance_per_unit_wavelength",
            "long_name": "top-of-atmosphere outgoing spectral radiance",
            "units": symbol(uck.get("radiance")),
        }

        return ds

    def _postprocess_add_viewing_angles(self, ds: xr.Dataset) -> xr.Dataset:
        # Compute viewing angles at pixel locations
        # Angle computation must match the kernel plugin's direction sampling
        # routine
        xs = ds.coords["x"].data
        ys = ds.coords["y"].data
        if not np.allclose((len(xs), len(ys)), self.film_resolution):
            raise ValueError(
                f"raw data width and height ({len(xs)}, {len(ys)}) does not "
                f"match film size ({self.film_resolution})"
            )

        if self.film_resolution[1] == 1:  # Plane case
            theta = (90.0 - 180.0 * xs).reshape(1, len(xs))
            phi = np.full_like(theta, self.orientation.m_as("deg"))

        else:  # Hemisphere case
            xy = np.array([(x, y) for y in ys for x in xs])
            angles = direction_to_angles(square_to_uniform_hemisphere(xy)).m_as("deg")
            theta = angles[:, 0].reshape((len(xs), len(ys))).T
            phi = angles[:, 1].reshape((len(xs), len(ys))).T

        # Assign angles as non-dimension coords
        ds = ds.assign_coords(
            {
                "vza": (
                    ("y", "x"),
                    theta,
                    {
                        "standard_name": "viewing_zenith_angle",
                        "long_name": "viewing zenith angle",
                        "units": symbol("deg"),
                    },
                ),
                "vaa": (
                    ("y", "x"),
                    phi,
                    {
                        "standard_name": "viewing_azimuth_angle",
                        "long_name": "viewing azimuth angle",
                        "units": symbol("deg"),
                    },
                ),
            }
        )

        return ds


@MeasureFactory.register("distant_reflectance")
@parse_docs
@attr.s
class DistantReflectanceMeasure(DistantRadianceMeasure):
    """
    Distant reflectance  measure scene element
    [:factorykey:`distant_reflectance`].

    This measure is a specialised version of the :class:`.DistantMeasure`. It
    implements similar functionality, with extra post-processing features to
    derive reflectance values from the recorded radiance.
    """

    def postprocess(self, illumination=None) -> xr.Dataset:
        """
        Return post-processed raw sensor results.

        Parameter ``illumination`` (:class:`DirectionalIllumination`):
            Incoming radiance value. *This keyword argument is required.*

        Returns → :class:`~xarray.Dataset`:
            Post-processed results.
        """
        if illumination is None:
            raise TypeError("missing required keyword argument 'illumination'")

        if not isinstance(illumination, DirectionalIllumination):
            raise ValueError(
                "keyword argument 'illumination' must be a "
                "DirectionalIllumination instance, got a "
                f"{illumination.__class__.__name__}"
            )

        # Get radiance data
        result = super(DistantReflectanceMeasure, self).postprocess()

        # Add illumination data
        result = self._postprocessing_add_illumination(result, illumination)

        # Compute reflectance data
        result = self._postprocessing_add_reflectance(result)

        return result

    def _postprocessing_add_illumination(
        self, ds: xr.Dataset, illumination: DirectionalIllumination
    ) -> xr.Dataset:
        # Collect illumination angular data
        saa = illumination.azimuth.m_as(ureg.deg)
        sza = illumination.zenith.m_as(ureg.deg)
        cos_sza = np.cos(np.deg2rad(sza))

        # Add angular dimensions
        ds = ds.expand_dims({"sza": [sza], "saa": [saa]}, axis=(0, 1))
        ds.coords["sza"].attrs = {
            "standard_name": "solar_zenith_angle",
            "long_name": "solar zenith angle",
            "units": symbol("deg"),
        }
        ds.coords["saa"].attrs = {
            "standard_name": "solar_azimuth_angle",
            "long_name": "solar azimuth angle",
            "units": symbol("deg"),
        }

        # Collect illumination spectral data
        k_irradiance_units = uck.get("irradiance")
        irradiances = (
            np.array(
                [
                    illumination.irradiance.eval(spectral_ctx=spectral_ctx).m_as(
                        k_irradiance_units
                    )
                    for spectral_ctx in self.spectral_cfg.spectral_ctxs()
                ]
            )
            * k_irradiance_units
        )
        spectral_coord_label = eradiate.mode().spectral_coord_label

        # Add irradiance variable
        ds["irradiance"] = (
            ("sza", "saa", spectral_coord_label),
            np.array(irradiances.magnitude * cos_sza).reshape((1, 1, len(irradiances))),
        )
        ds["irradiance"].attrs = {
            "standard_name": "horizontal_solar_irradiance_per_unit_wavelength",
            "long_name": "horizontal spectral irradiance",
            "units": symbol(k_irradiance_units),
        }

        return ds

    def _postprocessing_add_reflectance(self, ds: xr.Dataset) -> xr.Dataset:
        # Compute BRDF and BRF
        # We assume that all quantities are stored in kernel units
        ds["brdf"] = ds["lo"] / ds["irradiance"]
        ds["brdf"].attrs = {
            "standard_name": "brdf",
            "long_name": "bi-directional reflection distribution function",
            "units": symbol("1/sr"),
        }

        ds["brf"] = ds["brdf"] * np.pi
        ds["brf"].attrs = {
            "standard_name": "brf",
            "long_name": "bi-directional reflectance factor",
            "units": symbol("dimensionless"),
        }

        return ds
