"""Measurement-related scene generation facilities.

.. admonition:: Registered factory members
    :class: hint

    .. factorytable::
       :factory: SceneElementFactory
       :modules: eradiate.scenes.measure
"""
from abc import ABC
from abc import abstractmethod

import attr
import numpy as np

from .core import SceneElement, SceneElementFactory
from ..util import always_iterable
from ..util.attrs import (
    attrib_quantity, validator_has_len, validator_is_positive
)
from ..util.frame import angles_to_direction, spherical_to_cartesian
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
    def repack_results(self, results):
        """Pack the results into a format that is understood by the runner and application.

        .. admonition:: Example

            The 1D application expects results to be packed such that the zenith and
            azimuth angles form one dimension on the data each. Scene elements
            based on the :class:`~mitsuba.sensors.radiancemeterarray` sensor however
            store the results in a one-dimensional array, ignoring the arrangement of
            sensors.

        Parameter ``results`` (array):
            The data array as it is returned by the kernel plugin underlying the
            scene element

        Returns → array:
            The data reshaped to dimensions as expected by the application in which
            the scene element is used.
        """
        pass


@SceneElementFactory.register(name="distant")
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

    def repack_results(self, results):
        return results

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


@SceneElementFactory.register(name="perspective")
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

    def repack_results(self, results):
        return results

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


@SceneElementFactory.register(name="radiancemeter_hsphere")
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

        Default: "radiancemeter_hsphere"

    """
    # TODO: add figure to explain what "hemisphere" does

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

    def repack_results(self, results):
        """This method reshapes the 1D results returned by the
        :class:`~mitsuba.sensors.radiancemeterarray` kernel plugin into the shape
        implied by the azimuth and zenith angle resolutions, such that
        the result complies with the format required to further process the results."""

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

    def kernel_dict(self, ref=True):
        spp = self.spp
        directions = self._directions()
        origin = self.origin.to(kdu.get("length")).magnitude

        return {
            self.id: {
                "type": "radiancemeterarray",
                "directions": ", ".join([str(x) for x in directions.flatten()]),
                "origins": ", ".join([str(x) for x in origin] * len(directions)),
                "id": self.id,
                "sampler": {
                    "type": "independent",
                    "sample_count": spp
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
        }


@attr.s
@SceneElementFactory.register(name="radiancemeter_pplane")
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

    def repack_results(self, results):
        """This method reshapes the 1D results returned by the
        :class:`~mitsuba.sensors.radiancemeterarray` kernel plugin into the shape
        implied by the azimuth and zenith angle resolutions, such that
        the result complies with the format required to further process the results."""

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

    def kernel_dict(self, **kwargs):
        spp = self.spp
        directions = self._directions()
        origin = always_iterable(self.origin.to(kdu.get("length")).magnitude)

        return {
            self.id: {
                "type": "radiancemeterarray",
                "directions": ", ".join([str(x) for x in directions.flatten()]),
                "origins": ", ".join([str(x) for x in origin] * len(directions)),
                "id": self.id,
                "sampler": {
                    "type": "independent",
                    "sample_count": spp
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
        }
