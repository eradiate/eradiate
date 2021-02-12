"""Measurement-related scene generation facilities.

.. admonition:: Registered factory members [:class:`MeasureFactory`]
    :class: hint

    .. factorytable::
       :factory: MeasureFactory
"""
from abc import ABC
from abc import abstractmethod
from copy import deepcopy

import attr
import numpy as np

import eradiate.kernel
from .core import SceneElement
from .._mode import ModePrecision
from ..util.attrs import (
    attrib_quantity,
    converter_quantity,
    documented, get_doc, parse_docs, unit_enabled,
    validator_has_len,
    validator_is_number,
    validator_is_positive,
    validator_is_vector3,
    validator_quantity,
)
from ..util.collections import is_vector3
from ..util.factory import BaseFactory
from ..util.frame import angles_to_direction
from ..util.misc import always_iterable
from ..util.units import config_default_units as cdu
from ..util.units import kernel_default_units as kdu
from ..util.units import ureg


@parse_docs
@attr.s
class Measure(SceneElement, ABC):
    """Abstract class for all measure scene elements.
    """

    id = documented(
        attr.ib(
            default="measure",
            validator=attr.validators.optional((attr.validators.instance_of(str))),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default="\"measure\"",
    )

    @abstractmethod
    def postprocess_results(self, sensor_id, sensor_spp, results):
        """Process sensor data to extract measure results. These post-processing
        operations can include (but are not limited to) array reshaping and
        sensor data aggregation and data transformation operations.

        Parameter ``sensor_id`` (list):
            List of sensor_ids that belong to this measure

        Parameter ``sensor_spp`` (list):
            List of spp values that belong to this measure's sensors.

        Parameter ``results`` (dict):
            Dictionary, mapping sensor IDs to their respective results.

        Returns → dict:
            Recombined an reshaped results.
        """
        pass

    @abstractmethod
    def sensor_info(self):
        """This method returns a tuple of sensor IDs and the corresponding SPP
        values. If applicable this method will ensure sensors do not exceed
        SPP levels that lead to numerical precision loss in results."""
        pass


class MeasureFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`Measure`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: MeasureFactory
    """

    _constructed_type = Measure
    registry = {}


@unit_enabled
@attr.s
class Target:
    """Abstract interface for target selection classes used by :class:`DistantMeasure`."""

    def kernel_item(self):
        """Return kernel item."""
        raise NotImplementedError

    @staticmethod
    def new(target_type, *args, **kwargs):
        """Instantiate one of the supported child classes. This factory requires
        manual class registration. All position and keyword arguments are
        forwarded to the constructed type.

        Currently supported classes:

        * ``point``: :class:`TargetPoint`
        * ``rectangle``: :class:`TargetRectangle`

        Parameter ``target_type`` (str):
            Identifier of one of the supported child classes.
        """
        if target_type == "point":
            return TargetPoint(*args, **kwargs)
        elif target_type == "rectangle":
            return TargetRectangle(*args, **kwargs)
        else:
            raise ValueError(f"unknown target type {target_type}")

    @staticmethod
    def convert(value):
        """Object converter method.

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

            return Target.new(target_type, **d)

        if is_vector3(value):
            return Target.new("point", xyz=value)

        return value


@parse_docs
@attr.s
class TargetPoint(Target):
    """Point target specification.
    """

    # Target point in CDU
    xyz = documented(
        attrib_quantity(units_compatible=cdu.generator("length")),
        doc="Target point coordinates.\n"
            "\n"
            "Unit-enabled field (default: cdu[length]).",
        type="array-like[float, float, float]",
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
        return self.xyz.to(kdu.get("length")).magnitude


@parse_docs
@attr.s
class TargetRectangle(Target):
    """Rectangle target specification.

    This target spec defines an rectangular, axis-aligned zone where ray targets
    will be sampled.
    """

    # fmt: off
    # Corners of an axis-aligned rectangle in CDU
    xmin = documented(
        attrib_quantity(
            converter=converter_quantity(float),
            validator=validator_quantity(validator_is_number),
            units_compatible=cdu.generator("length")
        ),
        doc="Lower bound on the X axis.\n"
            "\n"
            "Unit-enabled field (default: cdu[length]).",
        type="float",
    )

    xmax = documented(
        attrib_quantity(
            converter=converter_quantity(float),
            validator=validator_quantity(validator_is_number),
            units_compatible=cdu.generator("length")
        ),
        doc="Upper bound on the X axis.\n"
            "\n"
            "Unit-enabled field (default: cdu[length]).",
        type="float",
    )

    ymin = documented(
        attrib_quantity(
            converter=converter_quantity(float),
            validator=validator_quantity(validator_is_number),
            units_compatible=cdu.generator("length")
        ),
        doc="Lower bound on the Y axis.\n"
            "\n"
            "Unit-enabled field (default: cdu[length]).",
        type="float",
    )

    ymax = documented(
        attrib_quantity(
            converter=converter_quantity(float),
            validator=validator_quantity(validator_is_number),
            units_compatible=cdu.generator("length"),
        ),
        doc="Upper bound on the Y axis.\n"
            "\n"
            "Unit-enabled field (default: cdu[length]).",
        type="float",
    )

    # fmt: on

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
        from eradiate.kernel.core import ScalarTransform4f

        xmin = self.xmin.to(kdu.get("length")).magnitude
        xmax = self.xmax.to(kdu.get("length")).magnitude
        ymin = self.ymin.to(kdu.get("length")).magnitude
        ymax = self.ymax.to(kdu.get("length")).magnitude

        dx = xmax - xmin
        dy = ymax - ymin

        to_world = ScalarTransform4f.translate(
            [0.5 * dx + xmin, 0.5 * dy + ymin, 0]
        ) * ScalarTransform4f.scale([0.5 * dx, 0.5 * dy, 1])

        return {"type": "rectangle", "to_world": to_world}


@MeasureFactory.register("distant")
@parse_docs
@attr.s
class DistantMeasure(Measure):
    """Distant measure scene element [:factorykey:`distant`].

    This scene element is a thin wrapper around the ``distant`` sensor kernel
    plugin. It parametrises the sensor is oriented based on the a pair of zenith
    and azimuth angles, following the convention used in Earth observation.
    """

    # fmt: off
    film_resolution = documented(
        attr.ib(
            default=(32, 32),
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(int),
                iterable_validator=validator_has_len(2)
            ),
        ),
        doc="Film resolution as a (height, width) 2-tuple. "
            "If the height is set to 1, direction sampling will be restricted to a "
            "plane.",
        type="array-like[int, int]",
        default="(32, 32)"
    )

    spp = documented(
        attr.ib(
            default=32,
            converter=int,
            validator=validator_is_positive
        ),
        doc="Number of samples per pixel.",
        type="int",
        default="32"
    )

    target = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(Target.convert),
            validator=attr.validators.optional(attr.validators.instance_of(Target)),
            on_setattr=attr.setters.pipe(attr.setters.convert, attr.setters.validate)
        ),
        doc="Target specification. If set to ``None``, default target point selection "
            "is used. The target can be specified using an array-like with 3 "
            "elements (which will be converted to a :class:`TargetPoint`) or a "
            "dictionary interpreted by :meth:`Target.convert`.",
        type=":class:`Target` or None",
        default="None",
    )

    orientation = documented(
        attrib_quantity(
            default=ureg.Quantity(0., ureg.deg),
            validator=validator_is_positive,
            units_compatible=cdu.generator("angle"),
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
            validator=validator_is_vector3,
        ),
        doc="Vector orienting the hemisphere mapped by the measure.",
        type="arraylike[float, float, float]",
        default="[0, 0, 1]"
    )

    flip_directions = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(bool)
        ),
        doc=" If ``True``, sampled directions will be flipped.",
        type="bool",
        default="False",
    )

    # fmt: on

    def sensor_info(self):
        """This method returns a tuple of sensor IDs and the corresponding SPP
        values. If applicable this method will ensure sensors do not exceed
        SPP levels that lead to numerical precision loss in results."""
        return [(self.id, self.spp)]

    def postprocess_results(self, sensor_ids, sensor_spp, results):
        """Process sensor data to extract measure results. These post-processing
        operations can include (but are not limited to) array reshaping and
        sensor data aggregation and data transformation operations.

        Parameter ``sensor_id`` (list):
            List of sensor_ids that belong to this measure

        Parameter ``sensor_spp`` (list):
            List of spp values that belong to this measure's sensors.

        Parameter ``results`` (dict):
            Dictionary, mapping sensor IDs to their respective results.

        Returns → dict:
            Recombined an reshaped results.
        """
        if isinstance(sensor_ids, list):
            sensor_ids = sensor_ids[0]

        data = results[sensor_ids]

        return np.reshape(data, (data.shape[0], data.shape[1]))

    # fmt: off
    def kernel_dict(self, ref=True):
        result = {
            "type": "distant",
            "id": self.id,
            "direction": self.direction,
            "orientation": [
                np.cos(self.orientation.to(ureg.rad).m),
                np.sin(self.orientation.to(ureg.rad).m),
                0.
            ],
            "sampler": {
                "type": "independent",
                "sample_count": self.spp
            },
            "film": {
                "type": "hdrfilm",
                "width": self.film_resolution[0],
                "height": self.film_resolution[1],
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"}
            },
        }

        if self.target is not None:
            target = self.target
            result["ray_target"] = target.kernel_item()

        if self.flip_directions is not None:
            result["flip_directions"] = self.flip_directions

        return {self.id: result}
    # fmt: on


@MeasureFactory.register("perspective")
@parse_docs
@attr.s
class PerspectiveCameraMeasure(Measure):
    """Perspective camera scene element [:factorykey:`perspective`].

    This scene element is a thin wrapper around the ``perspective`` sensor
    kernel plugin. It positions a perspective camera based on a set of vectors,
    specifying the origin, viewing direction and 'up' direction of the camera.
    """

    # fmt: off
    target = documented(
        attrib_quantity(
            default=ureg.Quantity([0, 0, 0], ureg.m),
            validator=validator_has_len(3),
            units_compatible=cdu.generator("length"),
        ),
        doc="A 3-element vector specifying the location targeted by the camera.\n"
            "\n"
            "Unit-enabled field (default: cdu[length]).",
        type="array-like[float, float, float]",
        default="[0, 0, 0] m"
    )

    origin = documented(
        attrib_quantity(
            default=ureg.Quantity([1, 1, 1], ureg.m),
            validator=validator_has_len(3),
            units_compatible=cdu.generator("length"),
        ),
        doc="A 3-element vector specifying the position of the camera.\n"
            "\n"
            "Unit-enabled field (default: cdu[length]).",
        type="array-like[float, float, float]",
        default="[1, 1, 1] m"
    )

    up = documented(
        attr.ib(
            default=[0, 0, 1],
            validator=validator_has_len(3)
        ),
        doc="A 3-element vector specifying the up direction of the camera.\n"
            "This vector must be different from the camera's viewing direction,\n"
            "which is given by ``target - origin``.",
        type="array-like[float, float, float]",
        default="[0, 0, 1]",
    )

    film_width = documented(
        attr.ib(
            default=64,
            converter=int,
            validator=validator_is_positive
        ),
        doc="Horizontal resolution of the film in pixels.",
        type="int",
        default="64",
    )

    film_height = documented(
        attr.ib(
            default=64,
            converter=int,
            validator=validator_is_positive
        ),
        doc="Vertical resolution of the film in pixels.",
        type="int",
        default="64",
    )

    spp = documented(
        attr.ib(
            default=32,
            converter=int,
            validator=validator_is_positive
        ),
        doc="Number of samples per pixel.",
        type="int",
        default="32",
    )

    # fmt: on

    @target.validator
    @origin.validator
    def _target_origin_validator(self, attribute, value):
        if np.allclose(self.target, self.origin):
            raise ValueError(
                f"While initializing {attribute}:"
                f"Origin and target must not be equal,"
                f"got target = {self.target}, origin = {self.origin}"
            )

    @up.validator
    def _up_validator(self, attribute, value):
        direction = self.target - self.origin
        if np.allclose(np.cross(direction, value), 0):
            raise ValueError(
                f"While initializing {attribute}:"
                f"Up direction must differ from viewing direction,"
                f"got up = {self.up}, view direction = {direction}."
            )

    def postprocess_results(self, sensor_ids, sensor_spp, results):
        """Process sensor data to extract measure results. These post-processing
        operations can include (but are not limited to) array reshaping and
        sensor data aggregation and data transformation operations.

        Parameter ``sensor_ids`` (list):
            List of sensor_ids that belong to this measure

        Parameter ``sensor_spp`` (list):
            List of spp values that belong to this measure's sensors.

        Parameter ``results`` (dict):
            Dictionary, mapping sensor IDs to their respective results.

        Returns → dict:
            Recombined an reshaped results.
        """
        if isinstance(sensor_ids, list):
            sensor_ids = sensor_ids[0]

        return results[sensor_ids]

    def sensor_info(self):
        """This method returns a tuple of sensor IDs and the corresponding SPP
        values. If applicable this method will ensure sensors do not exceed
        SPP levels that lead to numerical precision loss in results."""
        return [(self.id, self.spp)]

    # fmt: off
    def kernel_dict(self, ref=True):
        from eradiate.kernel.core import ScalarTransform4f

        target = self.target.to(kdu.get("length")).magnitude
        origin = self.origin.to(kdu.get("length")).magnitude

        return {
            self.id: {
                "type": "perspective",
                "far_clip": 1e7,
                "to_world": ScalarTransform4f.look_at(origin=origin, target=target, up=self.up),
                "sampler": {
                    "type": "independent",
                    "sample_count": self.spp
                },
                "film": {
                    "type": "hdrfilm",
                    "width": self.film_width,
                    "height": self.film_height,
                    "pixel_format": "luminance",
                    "component_format": "float32",
                    "rfilter": {"type": "box"}
                }
            }
        }
    # fmt: on


@MeasureFactory.register("radiancemeter_hsphere")
@parse_docs
@attr.s
class RadianceMeterHsphereMeasure(Measure):
    """
    Hemispherical radiancemeter measure scene element
    [:factorykey:`radiancemeter_hsphere`].

    This scene element creates a ``radiancemeterarray`` sensor kernel plugin
    covering the hemisphere defined by an ``origin`` point and a ``direction``
    vector.
    """

    # fmt: off
    id = documented(
        attr.ib(
            default="radiancemeter_hsphere",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(Measure, "id", "doc"),
        type=get_doc(Measure, "id", "type"),
        default="radiancemeter_hsphere",
    )

    zenith_res = documented(
        attrib_quantity(
            default=ureg.Quantity(10., ureg.deg),
            validator=validator_is_positive,
            units_compatible=cdu.generator("angle"),
        ),
        doc="Zenith angle resolution.\n"
            "\n"
            "Unit-enabled field (default unit: cdu[angle]).",
        type="float",
        default="10.0 deg"
    )

    azimuth_res = documented(
        attrib_quantity(
            default=ureg.Quantity(10., ureg.deg),
            validator=validator_is_positive,
            units_compatible=cdu.generator("angle"),
        ),
        doc="Azimuth angle resolution.\n"
            "\n"
            "Unit-enabled field (default unit: cdu[angle]).",
        type="float",
        default="10.0 deg",
    )

    origin = documented(
        attrib_quantity(
            default=ureg.Quantity([0, 0, 0], ureg.m),
            validator=validator_has_len(3),
            units_compatible=cdu.generator("length"),
        ),
        doc="Position of the sensor.\n"
            "\n"
            "Unit-enabled field (default unit: cdu[length]).",
        type="array-like[float, float, float]",
        default="[0, 0, 0] m",
    )

    direction = documented(
        attr.ib(
            default=[0, 0, 1],
            validator=validator_has_len(3)
        ),
        doc="Direction of the hemisphere's zenith.",
        type="array-like[float, float, float]",
        default="[0, 0, 1]",
    )

    orientation = documented(
        attr.ib(
            default=[1, 0, 0],
            validator=validator_has_len(3),
        ),
        doc="Direction with which azimuth origin is aligned.",
        type="array-like[float, float, float]",
        default="[1, 0, 0]"
    )

    hemisphere = documented(
        attr.ib(
            default="front",
            validator=attr.validators.in_(("front", "back")),
        ),
        doc="If set to ``\"front\"``, the created radiancemeter array directions "
            "will point to the hemisphere defined by ``direction``. If set to "
            "``\"back\"``, the created radiancemeter array directions will "
            "point to the hemisphere defined by ``-direction``.\n"
            "\n"
            ".. only:: latex\n"
            "\n"
            "   .. figure:: ../../../fig/radiancemeter_hsphere.png\n"
            "\n"
            ".. only:: not latex\n"
            "\n"
            "   .. figure:: ../../../fig/radiancemeter_hsphere.svg",
        type="\"front\" or \"back\"",
        default="\"front\"",
    )

    spp = documented(
        attr.ib(
            default=32,
            converter=int,
            validator=validator_is_positive
        ),
        doc="Number of samples per (zenith, azimuth) pair.",
        type="int",
        default="32",
    )

    # Private attributes
    _spp_max_single = attr.ib(
        default=1e5,
        converter=int,
        validator=validator_is_positive,
        repr=False
    )

    _zenith_angles = attr.ib(default=None, init=False)  # Set during post-init
    _azimuth_angles = attr.ib(default=None, init=False)  # Set during post-init

    # fmt: on

    def __attrs_post_init__(self):
        self._zenith_angles = ureg.Quantity(
            np.arange(0, 90.0, self.zenith_res.to(ureg.deg).magnitude), ureg.deg
        )
        self._azimuth_angles = ureg.Quantity(
            np.arange(0, 360.0, self.azimuth_res.to(ureg.deg).magnitude), ureg.deg
        )

    def postprocess_results(self, sensor_ids, sensor_spp, results):
        """This method reshapes the 1D results returned by the
        ``radiancemeterarray`` kernel plugin into the shape implied by the
        azimuth and zenith angle resolutions, such that the result complies with
        the format required to further process the results.

        Additionally, if the measure's sensor was split up in order to limit
        the SPP per sensor, this method will recombine the results from each
        sensor.

        Parameter ``sensor_ids`` (list):
            List of sensor_ids that belong to this measure

        Parameter ``sensor_spp`` (list):
            List of spp values that belong to this measure's sensors.

        Parameter ``results`` (dict):
            Dictionary, mapping sensor IDs to their respective results.

        Returns → dict:
            Recombined and reshaped results.
        """
        sensors = np.array([results[x] for x in sensor_ids])
        spp_sum = np.sum(sensor_spp)

        # multiply each sensor's result by its relative SPP and sum all results
        results = np.dot(sensors.transpose(), sensor_spp / spp_sum).transpose()

        return np.reshape(
            results, (len(self._zenith_angles), len(self._azimuth_angles))
        )

    def _orientation_transform(self):
        """Compute matrix that transforms vectors between object and world
        space.
        """
        from eradiate.kernel.core import Transform4f, Vector3f, Point3f

        origin = Point3f(self.origin.to(kdu.get("length")).magnitude)
        zenith_direction = Vector3f(self.direction)
        orientation = Vector3f(self.orientation)
        up = Transform4f.rotate(zenith_direction, 90).transform_vector(orientation)

        return Transform4f.look_at(origin, origin + zenith_direction, up)

    def _directions(self):
        """Generate the array of direction vectors to configure the kernel
        plugin. Directions are returned as a flattened list of 3-component
        vectors.
        """
        hemisphere_transform = self._orientation_transform()

        directions = []
        for theta in self._zenith_angles.to(ureg.rad).magnitude:
            for phi in self._azimuth_angles.to(ureg.rad).magnitude:
                directions.append(
                    hemisphere_transform.transform_vector(
                        angles_to_direction(theta=theta, phi=phi)
                    )
                )

        return (
            -np.array(directions) if self.hemisphere == "back" else np.array(directions)
        )

    def sensor_info(self):
        """This method generates the sensor_id for the kernel_scene sensor
        implementation. On top of that, it will perform the SPP-split if
        conditions are met. In single precision computation the SPP should not
        become too large, otherwise the results will degrade due to precision
        limitations. In this case, this method will create multiple sensor_ids.
        It returns a list of tuples, each holding a sensor_id and the
        corresponding SPP value. In the case of a SPP-split, none of the SPP
        values will exceed the threshold.
        """
        mode = eradiate.mode()

        if (
            mode.is_single_precision()
            and self.spp > self._spp_max_single
        ):
            spps = [
                self._spp_max_single
                for i in range(int(self.spp / self._spp_max_single))
            ]
            if self.spp % self._spp_max_single:
                spps.append(self.spp % self._spp_max_single)

            return [(f"{self.id}_{i}", spp) for i, spp in enumerate(spps)]

        else:
            return [(self.id, self.spp)]

    # fmt: off
    def kernel_dict(self, **kwargs):
        directions = self._directions()
        origin = always_iterable(self.origin.to(kdu.get("length")).magnitude)
        kernel_dict = {}

        base_dict = {
            "type": "radiancemeterarray",
            "directions": ", ".join([str(x) for x in directions.flatten()]),
            "origins": ", ".join([str(x) for x in origin] * len(directions)),
            "id": self.id,
            "sampler": {
                "type": "independent",
                "sample_count": 0
            },
            "film": {
                "type": "hdrfilm",
                "width": len(directions),
                "height": 1,
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"}
            }
        }

        for (sensor_id, spp) in self.sensor_info():
            sensor_dict = deepcopy(base_dict)
            sensor_dict["id"] = sensor_id
            sensor_dict["sampler"]["sample_count"] = spp
            kernel_dict[sensor_id] = sensor_dict

        return kernel_dict
    # fmt: on


@MeasureFactory.register("radiancemeter_plane")
@parse_docs
@attr.s
class RadianceMeterPlaneMeasure(Measure):
    """
    Plane radiancemeter measure scene element
    [:factorykey:`radiancemeter_plane`].

    This scene element creates a ``radiancemeterarray`` sensor kernel plugin
    covering a plane defined by an ``origin`` point, a ``direction`` vector and
    an ``orientation`` vector.
    """

    # fmt: off
    id = documented(
        attr.ib(
            default="radiancemeter_plane",
            validator=attr.validators.optional((attr.validators.instance_of(str))),
        ),
        doc=get_doc(Measure, "id", "doc"),
        type=get_doc(Measure, "id", "type"),
        default="\"radiancemeter_plane\"",
    )

    zenith_res = documented(
        attrib_quantity(
            default=ureg.Quantity(10., ureg.deg),
            validator=validator_is_positive,
            units_compatible=cdu.generator("angle"),
        ),
        doc="Zenith angle resolution.\n"
            "\n"
            "Unit-enabled field (default unit: cdu[angle]).",
        type=float,
        default="10.0 deg"
    )

    origin = documented(
        attrib_quantity(
            default=ureg.Quantity([0, 0, 0], ureg.m),
            validator=validator_has_len(3),
            units_compatible=cdu.generator("length"),
        ),
        doc="Position of the sensor.\n"
            "\n"
            "Unit-enabled field (default unit: cdu[length]).",
        type="array-like[float, float, float]",
        default="[0, 0, 0] m.",
    )

    direction = documented(
        attr.ib(
            default=[0, 0, 1],
            validator=validator_has_len(3)
        ),
        doc="Direction of the hemisphere's zenith.",
        type="array-like[float, float, float]",
        default="[0, 0, 1]",
    )

    orientation = documented(
        attr.ib(
            default=[1, 0, 0],
            validator=validator_has_len(3),
        ),
        doc="Direction with which azimuth origin is aligned.",
        type="array-like[float, float, float]",
        default="[1, 0, 0]",
    )

    hemisphere = documented(
        attr.ib(
            default="front",
            validator=attr.validators.in_(("front", "back")),
        ),
        doc="If set to ``\"front\"``, the created radiancemeter array "
            "directions will point to the hemisphere defined by ``direction``. "
            " If set to ``\"back\"``, the created radiancemeter array "
            "directions will point to the hemisphere defined by ``-direction``.\n"
            "\n"
            ".. only:: latex\n"
            "\n"
            "   .. figure:: ../../../fig/radiancemeter_plane.png\n"
            "\n"
            ".. only:: not latex\n"
            "\n"
            "   .. figure:: ../../../fig/radiancemeter_plane.svg",
        type="\"front\" or \"back\"",
        default="\"front\""
    )

    spp = documented(
        attr.ib(
            default=32,
            converter=int,
            validator=validator_is_positive
        ),
        doc="Number of samples per (zenith, azimuth) pair.",
        type="int",
        default="32",
    )

    # Private attributes
    _spp_max_single = attr.ib(
        default=1e5,
        converter=int,
        validator=validator_is_positive,
        repr=False
    )

    _zenith_angles = attr.ib(default=None, init=False)  # Set during post-init
    _azimuth_angles = attr.ib(default=None, init=False)  # Set during post-init

    # fmt: on

    def __attrs_post_init__(self):
        self._zenith_angles = ureg.Quantity(
            np.arange(0, 90.0, self.zenith_res.to(ureg.deg).magnitude), ureg.deg
        )
        self._azimuth_angles = ureg.Quantity(np.array([0, 180]), ureg.deg)

    def postprocess_results(self, sensor_ids, sensor_spps, runner_results):
        """This method reshapes the 1D results returned by the
        ``radiancemeterarray`` kernel plugin into the shape implied by the
        azimuth and zenith angle resolutions, such that the result complies with
        the format required to further process the results.

        Additionally, if the measure's sensor was split up in order to limit
        the SPP per sensor, this method will recombine the results from each
        sensor.

        Parameter ``sensor_ids`` (list):
            List of sensor IDs that belong to this measure.

        Parameter ``sensor_spps`` (list):
            List of SPP values that belong to this measure's sensors.

        Parameter ``runner_results`` (dict):
            Dictionary mapping sensor IDs to their respective results.

        Returns → dict:
            Recombined and reshaped results.
        """
        sensor_values = np.array([runner_results[x] for x in sensor_ids])
        spp_sum = np.sum(sensor_spps)

        # Compute weighted sum of sensor contributions
        # The transpose() is required to correctly position the dimension on
        # which dot() operates
        runner_results = np.dot(sensor_values.transpose(), sensor_spps).transpose()
        runner_results /= spp_sum

        return np.reshape(runner_results, (len(self._zenith_angles), 2))

    def _orientation_transform(self):
        """Compute matrix that transforms vectors between object and world space."""
        from eradiate.kernel.core import Transform4f, Point3f, Vector3f

        origin = Point3f(self.origin.to(kdu.get("length")).magnitude)
        zenith_direction = Vector3f(self.direction)
        orientation = Vector3f(self.orientation)

        up = Transform4f.rotate(zenith_direction, 90).transform_vector(orientation)
        if not np.any(np.cross(zenith_direction, up)):
            raise ValueError("Zenith direction and orientation must not be parallel!")

        return Transform4f.look_at(
            origin, [sum(x) for x in zip(origin, zenith_direction)], up
        )

    def _directions(self):
        """Generate the array of direction vectors to configure the kernel
        plugin. Directions are returned as a flattened list of 3-component
        vectors.
        """
        hemisphere_transform = self._orientation_transform()

        directions = []
        for theta in self._zenith_angles.to(ureg.rad).magnitude:
            for phi in self._azimuth_angles.to(ureg.rad).magnitude:
                directions.append(
                    hemisphere_transform.transform_vector(
                        angles_to_direction(theta=theta, phi=phi)
                    )
                )

        return (
            -np.array(directions) if self.hemisphere == "back" else np.array(directions)
        )

    def sensor_info(self):
        """This method generates the sensor_id for the kernel_scene sensor
        implementation. On top of that, it will perform the SPP-split if
        conditions are met. In single precision computation the SPP should not
        become too large, otherwise the results will degrade due to precision
        limitations. In this case, this method will create multiple sensor_ids.
        It returns a list of tuples, each holding a sensor_id and the
        corresponding SPP value. In the case of a SPP-split, none of the SPP
        values will exceed the threshold.
        """
        mode = eradiate.mode()

        if (
            mode.is_single_precision()
            and self.spp > self._spp_max_single
        ):
            spps = [
                self._spp_max_single
                for i in range(int(self.spp / self._spp_max_single))
            ]
            if self.spp % self._spp_max_single:
                spps.append(self.spp % self._spp_max_single)

            return [(f"{self.id}_{i}", spp) for i, spp in enumerate(spps)]

        else:
            return [(self.id, self.spp)]

    # fmt: off
    def kernel_dict(self, **kwargs):
        directions = self._directions()
        origin = always_iterable(self.origin.to(kdu.get("length")).magnitude)
        kernel_dict = {}

        base_dict = {
            "type": "radiancemeterarray",
            "directions": ", ".join([str(x) for x in directions.flatten()]),
            "origins": ", ".join([str(x) for x in origin] * len(directions)),
            "id": self.id,
            "sampler": {
                "type": "independent",
                "sample_count": 0
            },
            "film": {
                "type": "hdrfilm",
                "width": len(directions),
                "height": 1,
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"}
            }
        }

        for (sensor_id, spp) in self.sensor_info():
            sensor_dict = deepcopy(base_dict)
            sensor_dict["id"] = sensor_id
            sensor_dict["sampler"]["sample_count"] = spp
            kernel_dict[sensor_id] = sensor_dict

        return kernel_dict
    # fmt: on
