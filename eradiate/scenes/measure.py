"""Measurement-related scene generation facilities.

.. admonition:: Factory-enabled scene elements
    :class: hint

    .. factorytable::
        :modules: measure
"""
from abc import ABC
from abc import abstractmethod

import attr
import numpy as np

from .core import SceneElement, SceneElementFactory
from ..util.attrs import (
    attrib, attrib_float_positive, attrib_int_positive, attrib_units,
    validator_has_len
)
from ..util.frame import angles_to_direction, spherical_to_cartesian
from ..util.units import config_default_units as cdu, ureg
from ..util.units import kernel_default_units as kdu


@attr.s
class Measure(SceneElement, ABC):
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

    The sensor is oriented based on the classical angular convention used
    in Earth observation.

    .. admonition:: Configuration example
        :class: hint

        Default:
            .. code:: python

               {
                   "zenith": 0.,
                   "azimuth": 0.,
                   "spp": 10000,
               }

    .. admonition:: Configuration format
        :class: hint

        ``zenith`` (float):
            Zenith angle [deg].

            Default value: 0.

        ``azimuth`` (float):
            Azimuth angle value [deg].

            Default value: 0.

        ``spp`` (int):
            Number of samples.

            Default: 10000.
    """

    zenith = attrib_float_positive(
        default=0.,
        has_units=True
    )
    zenith_units = attrib_units(
        default=attr.Factory(lambda: cdu.get("angle")),
        compatible_units=ureg.deg,
    )

    azimuth = attrib_float_positive(
        default=0.,
        has_units=True
    )
    azimuth_units = attrib_units(
        default=attr.Factory(lambda: cdu.get("angle")),
        compatible_units=ureg.deg,
    )

    spp = attrib_int_positive(default=10000)

    def repack_results(self, results):
        return results

    def kernel_dict(self, **kwargs):
        zenith = self.get_quantity("zenith")
        azimuth = self.get_quantity("azimuth")
        spp = self.spp

        return {
            self.id: {
                "type": "distant",
                "direction": list(-angles_to_direction(
                    theta=zenith.to(ureg.rad).magnitude,
                    phi=azimuth.to(ureg.rad).magnitude
                )),
                "target": [0, 0, 0],
                "sampler": {
                    "type": "independent",
                    "sample_count": spp
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

    The sensor is oriented based on the classical angular convention used
    in Earth observation.

    The film is a square.

    .. admonition:: Configuration example
        :class: hint

        Default:
            .. code:: python

               {
                   "target": [0, 0, 0],
                   "zenith": 0.,
                   "azimuth": 0.,
                   "distance": 1.,
                   "res": 64,
                   "spp": 32,
               }

    .. admonition:: Configuration format
        :class: hint

        ``target`` (list[float]):
            A 3-element vector specifying the location targeted by the camera
            [u_length].

            Default: [0, 0, 0].

        ``zenith`` (float):
            Zenith angle [deg].

            Default value: 0.

        ``azimuth`` (float):
            Azimuth angle value [deg].

            Default value: 0.

        ``distance`` (float):
            Distance from the ``target`` point to the camera [u_length].

            Default: 1.

        ``res`` (int):
            Resolution of the film in pixels.

            Default: 64.

        ``spp`` (int):
            Number of samples per pixel.

            Default: 32.
    """

    target = attrib(
        default=[0, 0, 0],
        validator=validator_has_len(3),
        has_units=True
    )
    target_units = attrib_units(
        default=attr.Factory(lambda: cdu.get("length")),
        compatible_units=ureg.m,
    )

    zenith = attrib_float_positive(
        default=0.,
        has_units=True
    )
    zenith_units = attrib_units(
        default=attr.Factory(lambda: cdu.get("angle")),
        compatible_units=ureg.deg,
    )

    azimuth = attrib_float_positive(
        default=0.,
        has_units=True
    )
    azimuth_units = attrib_units(
        default=attr.Factory(lambda: cdu.get("angle")),
        compatible_units=ureg.deg,
    )

    distance = attrib_float_positive(
        default=1.,
        has_units=True
    )
    distance_units = attrib_units(
        default=attr.Factory(lambda: cdu.get("length")),
        compatible_units=ureg.m
    )

    res = attrib_int_positive(default=64)

    spp = attrib_int_positive(default=32)

    def repack_results(self, results):
        return results

    def kernel_dict(self, **kwargs):
        from eradiate.kernel.core import ScalarTransform4f

        target = self.get_quantity("target").to(kdu.get("length")).magnitude
        distance = self.get_quantity("distance").to(kdu.get("length")).magnitude
        zenith = self.get_quantity("zenith").to("rad").magnitude
        azimuth = self.get_quantity("azimuth").to("rad").magnitude

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
    """Distant hemispherical measure scene element [:factorykey:`radiancemeter_hsphere`].

    This creates a :class:`~mitsuba.sensors.radiancemeterarray` kernel plugin,
    covering the hemisphere defined by the "origin" point and the "direction" vector.

    The sensor is oriented based on the classical angular convention used
    in Earth observation.

    .. admonition:: Configuration format
        :class: hint

        ``zenith_res`` (float):
            Zenith angle resolution. Default. 10.

            Unit-enabled field (default unit: cdu[angle])

        ``azimuth_res`` (float):
            Azimuth angle resolution. Default: 10.

            <Unit-enabled field (default unit: cdu[angle])>

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

            Default: "radiancemeter_hemisphere"
    """

    zenith_res = attrib_float_positive(
        default=10.,
        has_units=True
    )
    zenith_res_units = attrib_units(
        default=attr.Factory(lambda: cdu.get("angle")),
        compatible_units=ureg.deg,
    )

    azimuth_res = attrib_float_positive(
        default=10.,
        has_units=True
    )
    azimuth_res_units = attrib_units(
        default=attr.Factory(lambda: cdu.get("angle")),
        compatible_units=ureg.deg,
    )

    origin = attrib(
        default=[0, 0, 0],
        validator=validator_has_len(3),
        has_units=True
    )
    origin_units = attrib_units(
        default=attr.Factory(lambda: cdu.get("length")),
        compatible_units=ureg.m,
    )

    direction = attrib(
        default=[0, 0, 1],
        validator=validator_has_len(3),
        has_units=False
    )

    orientation = attrib(
        default=[1, 0, 0],
        validator=validator_has_len(3),
        has_units=False
    )

    hemisphere = attrib(
        default="front",
        validator=attr.validators.in_(("front", "back")),
    )

    id = attr.ib(
        default="radiancemeter_hsphere",
        validator=attr.validators.optional((attr.validators.instance_of(str))),
    )

    spp = attrib_int_positive(default=32)

    _zenith_angles = attrib(default=None, init=False)  # Set during post-init
    _azimuth_angles = attrib(default=None, init=False)  # Set during post-init

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        self._zenith_angles = ureg.Quantity(np.arange(0, 90, self.zenith_res), ureg.deg)
        self._azimuth_angles = ureg.Quantity(np.arange(0, 360.001, self.azimuth_res), ureg.deg)

    def repack_results(self, results):
        """This method reshapes the 1D results returned by the
        :class:`~mitsuba.sensors.radiancemeterarray` kernel plugin into the shape
        implied by the azimuth and zenith angle resolutions, such that
        the result complies with the format required to further process the results."""

        return np.reshape(results, (len(self._zenith_angles), len(self._azimuth_angles)))

    def _orientation_transform(self):
        from eradiate.kernel.core import Transform4f, Vector3f, Point3f
        origin = Point3f(self.get_quantity("origin").to(kdu.get("length")).magnitude)
        zenith_direction = Vector3f(self.direction)
        orientation = Vector3f(self.orientation)
        up = Transform4f.rotate(zenith_direction, 90).transform_vector(orientation)

        return Transform4f.look_at(origin, origin + zenith_direction, up)

    def directions(self):
        hemisphere_transform = self._orientation_transform()

        directions = []
        for theta in self._zenith_angles.to(ureg.rad).magnitude:
            for phi in self._azimuth_angles.to(ureg.rad).magnitude:
                directions.append(hemisphere_transform.transform_vector(
                    angles_to_direction(theta=np.deg2rad(theta), phi=np.deg2rad(phi)))
                )

        return -np.array(directions) if self.hemisphere == "back" \
            else np.array(directions)

    def kernel_dict(self, **kwargs):
        spp = self.spp
        directions = self.directions()
        origin = self.get_quantity("origin").to(kdu.get("length")).magnitude

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
