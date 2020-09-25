"""Measurement-related scene generation facilities.

.. admonition:: Factory-enabled scene generation helpers
    :class: hint

    .. factorytable::
        :modules: measure
"""

import attr
import numpy as np

from .core import Factory, SceneHelper
from ..util.frame import angles_to_direction, spherical_to_cartesian
from ..util.units import config_default_units as cdu
from ..util.units import kernel_default_units as kdu


class Measure(SceneHelper):
    @classmethod
    def config_schema(cls):
        d = super(Measure, cls).config_schema()
        d["id"]["default"] = "measure"
        return d


@attr.s
@Factory.register(name="distant")
class DistantMeasure(Measure):
    """Distant measure scene generation helper [:factorykey:`distant`].

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

    @classmethod
    def config_schema(cls):
        d = super(DistantMeasure, cls).config_schema()
        d.update({
            "zenith": {
                "type": "number",
                "min": 0.,
                "max": 90.,
                "default": 0.0,
            },
            "zenith_unit": {
                "type": "string",
                "default": cdu.get_str("angle")
            },
            "azimuth": {
                "type": "number",
                "min": 0.,
                "max": 360.,
                "default": 0.0,
            },
            "azimuth_unit": {
                "type": "string",
                "default": cdu.get_str("angle")
            },
            "spp": {
                "type": "integer",
                "min": 0,
                "default": 10000
            }
        })
        return d

    def kernel_dict(self, **kwargs):
        zenith = self.config.get_quantity("zenith").to(kdu.get("angle")).magnitude
        azimuth = self.config.get_quantity("azimuth").to(kdu.get("angle")).magnitude
        spp = self.config.get_quantity("spp")
        return {
            self.id: {
                "type": "distant",
                "direction": list(-angles_to_direction(
                    theta=np.deg2rad(zenith),
                    phi=np.deg2rad(azimuth)
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


@attr.s
@Factory.register(name="perspective")
class PerspectiveCameraMeasure(Measure):
    """Perspective camera scene generation helper [:factorykey:`perspective`].

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

    @classmethod
    def config_schema(cls):
        d = super(PerspectiveCameraMeasure, cls).config_schema()
        d.update({
            "target": {
                "type": "list",
                "items": [{"type": "number"}] * 3,
                "default": [0, 0, 0]
            },
            "target_unit": {
                "type": "string",
                "default": cdu.get_str("length")
            },
            "zenith": {
                "type": "number",
                "min": 0.,
                "max": 90.,
                "default": 45.
            },
            "zenith_unit": {
                "type": "string",
                "default": cdu.get_str("angle")
            },
            "azimuth": {
                "type": "number",
                "min": 0.,
                "max": 360.,
                "default": 180.
            },
            "azimuth_unit": {
                "type": "string",
                "default": cdu.get_str("angle")
            },
            "distance": {
                "type": "number",
                "min": 0.,
                "default": 1.,
            },
            "distance_unit": {
                "type": "string",
                "default": cdu.get_str("length")
            },
            "res": {
                "type": "integer",
                "min": 0,
                "default": 64
            },
            "spp": {
                "type": "integer",
                "min": 0,
                "default": 32
            }
        })
        return d

    def kernel_dict(self, **kwargs):
        from eradiate.kernel.core import ScalarTransform4f

        target = self.config.get_quantity("target").to(kdu.get("length")).magnitude
        distance = self.config.get_quantity("distance").to(kdu.get("length")).magnitude
        res = self.config["res"]
        spp = self.config["spp"]
        zenith = self.config.get_quantity("zenith").to("rad").magnitude
        azimuth = self.config.get_quantity("azimuth").to("rad").magnitude

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
                    "sample_count": spp
                },
                "film": {
                    "type": "hdrfilm",
                    "width": res,
                    "height": res,
                    "pixel_format": "luminance",
                    "component_format": "float32",
                    "rfilter": {"type": "box"}
                }
            }
        }


@attr.s
@Factory.register(name="radiance_hemi")
class RadianceMeterHemisphere(Measure):
    """Distant hemispherical measure scene generation helper [:factorykey:`radiance_hemi`].

    This creates a :class:`~mitsuba.sensors.radiancemeterarray` kernel plugin,
    covering the hemisphere defined by the "origin" point and the "direction" vector.

    The sensor is oriented based on the classical angular convention used
    in Earth observation.

    .. admonition:: Configuration example
        :class: hint

        Default:
            .. code:: python

               {
                   "zenith_res": 10.,
                   "azimuth_res": 10.,
                   "origin": [0, 0, 0],
                   "zenith_direction": [0, 0, 1],
                   "hemisphere": "front"
                   "spp": 32,
               }

    .. admonition:: Configuration format
        :class: hint

        ``zenith_res`` (float):
            Zenith angle resolution

            Default value: 10 degrees

        ``azimuth_res`` (float):
            Azimuth angle resolution

            Default value: 10 degrees

        ``origin`` (list[float]):
            Position of the RadiancemeterArray

            Default value: [0, 0, 0]

        ``direction`` (list[float]):
            Direction of the hemisphere's zenith

            Default value: [0, 0, 1]

        ``orientation`` (list[float]):
            Direction of the hemisphere's principal plane

            Default value: [1, 0, 0]

        ``hemisphere`` (string):
            "front" sets the sensors to point into the hemisphere that holds
            the "direction" vector, while "back" sets them to point into the
            opposite hemisphere.

            Default value: "front"

        ``spp`` (int):
            Number of samples.

            Default: 32.
    """

    @classmethod
    def config_schema(cls):
        d = super(RadianceMeterHemisphere, cls).config_schema()
        d.update({
            "zenith_res": {
                "type": "number",
                "min": 1,
                "default": 10,
            },
            "zenith_res_unit": {
                "type": "string",
                "default": str(cdu.get("angle"))
            },
            "azimuth_res": {
                "type": "number",
                "min": 1,
                "default": 10,
            },
            "azimuth_res_unit": {
                "type": "string",
                "default": str(cdu.get("angle"))
            },
            "origin": {
                "type": "list",
                "items": [{"type": "number"}] * 3,
                "default": [0, 0, 0]
            },
            "direction": {
                "type": "list",
                "items": [{"type": "number"}] * 3,
                "default": [0, 0, 1]
            },
            "hemisphere": {
                "type": "string",
                "allowed": ["front", "back"],
                "default": "front"
            },
            "orientation": {
                "type": "list",
                "items": [{"type": "number"}] * 3,
                "default": [1, 0, 0]
            },
            "spp": {
                "type": "integer",
                "min": 1,
                "default": 32
            }
        })
        
        return d

    zenith_angles = attr.ib(default=[])
    azimuth_angles = attr.ib(default=[])

    def init(self):
        """(Re)initialise internal state.

        This method is automatically called by the constructor to initialise the
        object."""

        zenith_res = self.config.get_quantity("zenith_res").to("deg").magnitude
        azimuth_res = self.config.get_quantity("azimuth_res").to("deg").magnitude
        
        self.zenith_angles = np.arange(0, 90, zenith_res)
        self.azimuth_angles = np.arange(0, 360, azimuth_res)

    def repack_results(self, results):
        """This method reshapes the 1D results returned by the
        :class:`~mitsuba.sensors.radiancemeterarray` kernel plugin into the shape
        implied by the azimuth and zenith angle resolutions, such that
        the result complies with the format required to further process the results."""

        return np.reshape(results, (len(self.zenith_angles), len(self.azimuth_angles)))

    def get_orientation_transform(self):
        from eradiate.kernel.core import Transform4f
        origin = self.config.get_quantity("origin")
        zenith_direction = self.config.get_quantity("direction")
        orientation = self.config.get_quantity("orientation")

        return Transform4f.look_at(origin, zenith_direction, orientation)

    def generate_directions(self):
        hemisphere_transform = self.get_orientation_transform()

        directions = []
        for theta in self.zenith_angles:
            for phi in self.azimuth_angles:
                directions.append(hemisphere_transform.transform_vector(
                    angles_to_direction(theta=theta, phi=phi))
                )

        return -np.array(directions) if self.config.get("hemisphere") == "back" \
            else np.array(directions)

    def kernel_dict(self, **kwargs):
        spp = self.config.get_quantity("spp")
        directions = self.generate_directions()
        origin = self.config.get_quantity("origin")

        return {
            self.id: {
                "type": "radiancemeterarray",
                "directions": ", ".join([str(x) for x in directions.flatten()]),
                "origins": ", ".join([str(x) for x in origin]*len(directions)),
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
