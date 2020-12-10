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
from ..util.attrs import (
    attrib_quantity, validator_has_len, validator_is_positive
)
from ..util.factory import BaseFactory
from ..util.frame import angles_to_direction, spherical_to_cartesian
from ..util.misc import always_iterable
from ..util.units import config_default_units as cdu
from ..util.units import kernel_default_units as kdu
from ..util.units import ureg


@attr.s
class Measure(SceneElement, ABC):
    """Abstract class for all measure scene elements.

    See :class:`.SceneElement` for undocumented members.
    """

    id = attr.ib(
        default="measure",
        validator=attr.validators.optional((attr.validators.instance_of(str))),
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


@MeasureFactory.register(name="distant")
@attr.s
class DistantMeasure(Measure):
    """Distant measure scene element [:factorykey:`distant`].

    This scene element is a thin wrapper around the ``distant`` sensor kernel
    plugin. It parametrises the sensor is oriented based on the a pair of zenith
    and azimuth angles, following the convention used in Earth observation.

    .. rubric:: Constructor arguments / instance attributes

    ``zenith`` (float):
        Zenith angle. Default value: 0 deg.

        Unit-enabled field (default: cdu[angle]).

    ``azimuth`` (float):
        Azimuth angle value. Default value: 0 deg.

        Unit-enabled field (default: cdu[angle]).

    ``spp`` (int):
        Number of samples. Default: 10000.
    """

    zenith = attrib_quantity(
        default=ureg.Quantity(0., ureg.deg),
        validator=validator_is_positive,
        units_compatible=cdu.generator("angle"),
    )

    azimuth = attrib_quantity(
        default=ureg.Quantity(0., ureg.deg),
        validator=validator_is_positive,
        units_compatible=cdu.generator("angle"),
    )

    spp = attr.ib(
        default=10000,
        converter=int,
        validator=validator_is_positive
    )

    def sensor_info(self):
        """This method returns a tuple of sensor IDs and the corresponding SPP
        values. If applicable this method will ensure sensors do not exceed
        SPP levels that lead to numerical precision loss in results."""
        return [(self.id, self.spp)]

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
        return results["sensor_id"]

    def kernel_dict(self, ref=True):
        return {
            self.id: {
                "type": "distant",
                "direction": list(-angles_to_direction(
                    theta=self.zenith.to(ureg.rad).magnitude,
                    phi=self.azimuth.to(ureg.rad).magnitude
                )),
                "target": [0, 0, 0],
                "sampler": {
                    "type": "independent",
                    "sample_count": self.spp
                },
                "film": {
                    "type": "hdrfilm",
                    "width": 1,
                    "height": 1,
                    "pixel_format": "luminance",
                    "component_format": "float32",
                    "rfilter": {"type": "box"}
                }
            }
        }


@MeasureFactory.register(name="perspective")
@attr.s
class PerspectiveCameraMeasure(Measure):
    """Perspective camera scene element [:factorykey:`perspective`].

    This scene element is a thin wrapper around the ``perspective`` sensor
    kernel plugin. It positions a perspective camera based on the a pair of
    zenith and azimuth angles, following the convention used in Earth
    observation. The film is a square.

    .. rubric:: Constructor arguments / instance attributes

    ``target`` (array[float]):
        A 3-element vector specifying the location targeted by the camera.
        Default: [0, 0, 0] m.

        Unit-enabled field (default: cdu[length]).

    ``zenith`` (float):
        Zenith angle. Default value: 0 deg.

        Unit-enabled field (default: cdu[angle]).

    ``azimuth`` (float):
        Azimuth angle value. Default value: 0 deg.

        Unit-enabled field (default: cdu[angle]).

    ``distance`` (float):
        Distance from the ``target`` point to the camera.
        Default: 1 km.

        Unit-enabled field (default: cdu[length]).

    ``res`` (int):
        Resolution of the film in pixels. Default: 64.

    ``spp`` (int):
        Number of samples per pixel. Default: 32.
    """

    target = attrib_quantity(
        default=ureg.Quantity([0, 0, 0], ureg.m),
        validator=validator_has_len(3),
        units_compatible=cdu.generator("length"),
    )

    zenith = attrib_quantity(
        default=ureg.Quantity(0., ureg.deg),
        validator=validator_is_positive,
        units_compatible=cdu.generator("angle"),
    )

    azimuth = attrib_quantity(
        default=ureg.Quantity(0., ureg.deg),
        validator=validator_is_positive,
        units_compatible=cdu.generator("angle"),
    )

    distance = attrib_quantity(
        default=ureg.Quantity(1., ureg.km),
        validator=validator_is_positive,
        units_compatible=cdu.generator("length"),
    )

    res = attr.ib(
        default=64,
        converter=int,
        validator=validator_is_positive
    )

    spp = attr.ib(
        default=32,
        converter=int,
        validator=validator_is_positive
    )

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
        return results["sensor_id"]

    def sensor_info(self):
        """This method returns a tuple of sensor IDs and the corresponding SPP
        values. If applicable this method will ensure sensors do not exceed
        SPP levels that lead to numerical precision loss in results."""
        return [(self.id, self.spp)]

    def kernel_dict(self, ref=True):
        from eradiate.kernel.core import ScalarTransform4f

        target = self.target.to(kdu.get("length")).magnitude
        distance = self.distance.to(kdu.get("length")).magnitude
        zenith = self.zenith.to("rad").magnitude
        azimuth = self.azimuth.to("rad").magnitude

        origin = spherical_to_cartesian(distance, zenith, azimuth)
        direction = origin / np.linalg.norm(origin)

        if np.allclose(origin, target):
            raise ValueError("target is too close to the camera")

        up = [np.cos(azimuth), np.sin(azimuth), 0] \
            if np.allclose(direction, [0, 0, 1]) \
            else [0, 0, 1]

        return {
            self.id: {
                "type": "perspective",
                "far_clip": 1e7,
                "to_world": ScalarTransform4f.look_at(origin=origin, target=target, up=up),
                "sampler": {
                    "type": "independent",
                    "sample_count": self.spp
                },
                "film": {
                    "type": "hdrfilm",
                    "width": self.res,
                    "height": self.res,
                    "pixel_format": "luminance",
                    "component_format": "float32",
                    "rfilter": {"type": "box"}
                }
            }
        }


@MeasureFactory.register(name="radiancemeter_hsphere")
@attr.s
class RadianceMeterHsphereMeasure(Measure):
    """Hemispherical radiancemeter measure scene element
    [:factorykey:`radiancemeter_hsphere`].

    This scene element creates a ``radiancemeterarray`` sensor kernel plugin
    covering the hemisphere defined by an ``origin`` point and a ``direction``
    vector.

    See :class:`Measure` for undocumented members.

    .. rubric:: Constructor arguments / instance attributes

    ``zenith_res`` (float):
        Zenith angle resolution. Default:  10 deg.

        Unit-enabled field (default unit: cdu[angle]).

    ``azimuth_res`` (float):
        Azimuth angle resolution. Default: 10 deg.

        Unit-enabled field (default unit: cdu[angle]).

    ``origin`` (list[float]):
        Position of the sensor. Default: [0, 0, 0] m.

        Unit-enabled field (default unit: cdu[length]).

    ``direction`` (list[float]):
        Direction of the hemisphere's zenith. Default: [0, 0, 1].

    ``orientation`` (list[float]):
        Direction with which azimuth origin is aligned.
        Default value: [1, 0, 0].

    ``hemisphere`` ("front" or "back"):
        If set to ``"front"``, the created radiancemeter array directions will
        point to the hemisphere defined by ``direction``.
        If set to ``"back"``, the created radiancemeter array directions will
        point to the hemisphere defined by ``-direction``.
        Default value: ``"front"``.

        .. only:: latex

           .. figure:: ../../../fig/radiancemeter_hsphere.png

        .. only:: not latex

           .. figure:: ../../../fig/radiancemeter_hsphere.svg


    ``spp`` (int):
        Number of samples per (zenith, azimuth) pair.
        Default: 32.

    ``id`` (str):
        Identifier to allow mapping of results to the measure inside an application.
        Default value: "radiancemeter_hsphere".

    """

    zenith_res = attrib_quantity(
        default=ureg.Quantity(10., ureg.deg),
        validator=validator_is_positive,
        units_compatible=cdu.generator("angle"),
    )

    azimuth_res = attrib_quantity(
        default=ureg.Quantity(10., ureg.deg),
        validator=validator_is_positive,
        units_compatible=cdu.generator("angle"),
    )

    origin = attrib_quantity(
        default=ureg.Quantity([0, 0, 0], ureg.m),
        validator=validator_has_len(3),
        units_compatible=cdu.generator("length"),
    )

    direction = attr.ib(
        default=[0, 0, 1],
        validator=validator_has_len(3)
    )

    orientation = attr.ib(
        default=[1, 0, 0],
        validator=validator_has_len(3),
    )

    hemisphere = attr.ib(
        default="front",
        validator=attr.validators.in_(("front", "back")),
    )

    id = attr.ib(
        default="radiancemeter_hsphere",
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )

    spp = attr.ib(
        default=32,
        converter=int,
        validator=validator_is_positive
    )

    _zenith_angles = attr.ib(default=None, init=False)  # Set during post-init
    _azimuth_angles = attr.ib(default=None, init=False)  # Set during post-init

    def __attrs_post_init__(self):
        self._zenith_angles = ureg.Quantity(
            np.arange(0, 90., self.zenith_res.to(ureg.deg).magnitude),
            ureg.deg
        )
        self._azimuth_angles = ureg.Quantity(
            np.arange(0, 360., self.azimuth_res.to(ureg.deg).magnitude),
            ureg.deg
        )

    def postprocess_results(self, sensor_id, sensor_spp, results):
        """This method reshapes the 1D results returned by the
        :class:`~mitsuba.sensors.radiancemeterarray` kernel plugin into the shape
        implied by the azimuth and zenith angle resolutions, such that
        the result complies with the format required to further process the results.

        Additionally, if the measure's sensor was split up in order to limit
        the SPP per sensor, this method will recombine the results from each
        sensor.

        Parameter ``sensor_id`` (list):
            List of sensor_ids that belong to this measure

        Parameter ``sensor_spp`` (list):
            List of spp values that belong to this measure's sensors.

        Parameter ``results`` (dict):
            Dictionary, mapping sensor IDs to their respective results.

        Returns → dict:
            Recombined an reshaped results.
        """
        sensors = np.array([results[x] for x in sensor_id])
        spp_sum = np.sum(sensor_spp)

        # multiply each sensor's result by its relative SPP and sum all results
        results = np.dot(sensors.transpose(), sensor_spp / spp_sum).transpose()

        return np.reshape(results, (len(self._zenith_angles), len(self._azimuth_angles)))

    def _orientation_transform(self):
        """Compute matrix that transforms vectors between object and world space."""
        from eradiate.kernel.core import Transform4f, Vector3f, Point3f
        origin = Point3f(self.origin.to(kdu.get("length")).magnitude)
        zenith_direction = Vector3f(self.direction)
        orientation = Vector3f(self.orientation)
        up = Transform4f.rotate(zenith_direction, 90).transform_vector(orientation)

        return Transform4f.look_at(origin, origin + zenith_direction, up)

    def _directions(self):
        """Generate the array of direction vectors to configure the kernel plugin.
        Directions are returned as a flattened list of 3-component vectors."""
        hemisphere_transform = self._orientation_transform()

        directions = []
        for theta in self._zenith_angles.to(ureg.rad).magnitude:
            for phi in self._azimuth_angles.to(ureg.rad).magnitude:
                directions.append(hemisphere_transform.transform_vector(
                    angles_to_direction(theta=theta, phi=phi))
                )

        return -np.array(directions) if self.hemisphere == "back" \
            else np.array(directions)

    def sensor_info(self):
        """This method generates the sensor_id for the kernel_scene sensor
        implementation. On top of that, it will perform the SPP-split if conditions
        are met. In single precision computation the SPP should not become too
        large, otherwise the results will degrade due to precision limitations.
        In this case, this method will create multiple sensor_ids.
        It returns a list of tuples, each holding a sensor_id and the corresponding
        SPP value. In the case of a SPP-split, none of the SPP values will
        exceed the threshold."""
        if eradiate.mode.precision == eradiate.ModePrecision.SINGLE and self.spp > 1e5:
            sensor_info = [(f"{self.id}_{i}", int(1e5)) for i in range(int(self.spp / 1e5))]
            if self.spp % 1e5 != 0:
                sensor_info += [(f"{self.id}_{int(self.spp / 1e5)}", self.spp % 1e5)]
            return sensor_info
        else:
            return [(self.id, self.spp)]

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


@attr.s
@MeasureFactory.register(name="radiancemeter_pplane")
class RadianceMeterPPlaneMeasure(Measure):
    """Distant principal plane measure scene generation helper [:factorykey:`radiancemeter_pplane`].

    This creates a :class:`~mitsuba.sensors.radiancemeterarray` kernel plugin,
    covering the plane defined by the "origin" point, the "direction" vector as the apex
    of the hemisphere and the "orientation" vector to select the plane inside this hemisphere.

    The "hemisphere" parameter can be used to invert the sensor's orientation: Setting it to
    "front" lets the sensor observe the hemisphere defined by the origin and the direction,
    which can be thought of as looking upwards. Setting the parameter to "back" lets the sensor
    observe the opposite hemisphere, or look downwards.

    The sensor is oriented based on the classical angular convention used
    in Earth observation.

    .. admonition:: Configuration format
        :class: hint

        ``zenith_res`` (float):
            Zenith angle resolution. Default. 10.

            Unit-enabled field (default unit: cdu[angle])

        ``azimuth_res`` (float):
            Azimuth angle resolution. Default: 10.

            Unit-enabled field (default unit: cdu[angle])

        ``origin`` (list[float]):
            Position of the sensor. Default: [0, 0, 0]

            Unit-enabled field (default unit: cdu[length])

        ``direction`` (list[float]):
            Direction of the hemisphere's zenith

            Default value: [0, 0, 1]

        ``orientation`` (list[float]):
            Direction with which azimuth origin is aligned

            Default value: [1, 0, 0]

        ``hemisphere`` (str):
            "front" sets the sensors to point into the hemisphere that holds
            the "direction" vector, while "back" sets them to point into the
            opposite hemisphere.

            Default value: "front"

        ``spp`` (int):
            Number of samples per (zenith, azimuth) pair.

            Default: 32.

        ``id`` (str):
            Identifier to allow mapping of results to the measure inside an application.

            Default: "radiancemeter_pplane"
    """
    zenith_res = attrib_quantity(
        default=ureg.Quantity(10., ureg.deg),
        validator=validator_is_positive,
        units_compatible=cdu.generator("angle"),
    )

    origin = attrib_quantity(
        default=ureg.Quantity([0, 0, 0], ureg.m),
        validator=validator_has_len(3),
        units_compatible=cdu.generator("length"),
    )

    direction = attr.ib(
        default=[0, 0, 1],
        validator=validator_has_len(3)
    )

    orientation = attr.ib(
        default=[1, 0, 0],
        validator=validator_has_len(3),
    )

    hemisphere = attr.ib(
        default="front",
        validator=attr.validators.in_(("front", "back")),
    )

    id = attr.ib(
        default="radiancemeter_pplane",
        validator=attr.validators.optional((attr.validators.instance_of(str))),
    )

    spp = attr.ib(
        default=32,
        converter=int,
        validator=validator_is_positive
    )

    _zenith_angles = attr.ib(default=None, init=False)  # Set during post-init
    _azimuth_angles = attr.ib(default=None, init=False)  # Set during post-init

    def __attrs_post_init__(self):
        self._zenith_angles = ureg.Quantity(
            np.arange(0, 90., self.zenith_res.to(ureg.deg).magnitude),
            ureg.deg
        )
        self._azimuth_angles = ureg.Quantity(
            np.array([0, 180]),
            ureg.deg
        )

    def postprocess_results(self, sensor_id, sensor_spp, results):
        """This method reshapes the 1D results returned by the
        :class:`~mitsuba.sensors.radiancemeterarray` kernel plugin into the shape
        implied by the azimuth and zenith angle resolutions, such that
        the result complies with the format required to further process the results.

        Additionally, if the measure's sensor was split up in order to limit
        the SPP per sensor, this method will recombine the results from each
        sensor.

        Parameter ``sensor_id`` (list):
            List of sensor_ids that belong to this measure

        Parameter ``sensor_spp`` (list):
            List of spp values that belong to this measure's sensors.

        Parameter ``results`` (dict):
            Dictionary, mapping sensor IDs to their respective results.

        Returns → dict:
            Recombined an reshaped results.
        """

        sensors = np.array([results[x] for x in sensor_id])
        spp_sum = np.sum(sensor_spp)

        # multiply each sensor's result by its relative SPP and sum all results
        results = np.dot(sensors.transpose(), sensor_spp / spp_sum).transpose()
        print(results)

        return np.reshape(results, (len(self._zenith_angles), 2))

    def _orientation_transform(self):
        """Compute matrix that transforms vectors between object and world space."""
        from eradiate.kernel.core import Transform4f, Point3f, Vector3f
        origin = Point3f(self.origin.to(kdu.get("length")).magnitude)
        zenith_direction = Vector3f(self.direction)
        orientation = Vector3f(self.orientation)

        up = Transform4f.rotate(zenith_direction, 90).transform_vector(orientation)
        if not np.any(np.cross(zenith_direction, up)):
            raise ValueError("Zenith direction and orientation must not be parallel!")

        return Transform4f.look_at(origin, [sum(x) for x in zip(origin, zenith_direction)], up)

    def _directions(self):
        """Generate the array of direction vectors to configure the kernel plugin.
        Directions are returned as a flattened list of 3-component vectors."""
        hemisphere_transform = self._orientation_transform()

        directions = []
        for theta in self._zenith_angles.to(ureg.rad).magnitude:
            for phi in self._azimuth_angles.to(ureg.rad).magnitude:
                directions.append(hemisphere_transform.transform_vector(
                    angles_to_direction(theta=theta, phi=phi))
                )

        return -np.array(directions) if self.hemisphere == "back" \
            else np.array(directions)

    def sensor_info(self):
        """This method generates the sensor_id for the kernel_scene sensor
        implementation. On top of that, it will perform the SPP-split if conditions
        are met. In single precision computation the SPP should not become too
        large, otherwise the results will degrade due to precision limitations.
        In this case, this method will create multiple sensor_ids.
        It returns a list of tuples, each holding a sensor_id and the corresponding
        SPP value. In the case of a SPP-split, none of the SPP values will
        exceed the threshold."""
        if eradiate.mode.precision == eradiate.ModePrecision.SINGLE and self.spp > 1e5:
            sensor_info = [(f"{self.id}_{i}", int(1e5)) for i in range(int(self.spp / 1e5))]
            if self.spp % 1e5 != 0:
                sensor_info += [(f"{self.id}_{int(self.spp / 1e5)}", self.spp % 1e5)]
            return sensor_info
        else:
            return [(self.id, self.spp)]

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
