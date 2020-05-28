import attr
import eradiate
import numpy as np

from ..scenes.atmosphere import RayleighHomogeneous
from ..util.frame import angles_to_direction
from ..util import ensure_array, ureg


@ureg.wraps(None, (ureg.deg, ureg.deg, None), strict=False)
def _make_distant(zenith=0., azimuth=0., spp=10000):
    """Create a `distant` plugin interface instance.

    Parameter ``zenith`` (float)
        Zenith angle [deg].

    Parameter ``azimuth`` (float)
        Azimuth angle [deg].

    Parameter ``spp`` (int)
        Number of samples used from this sensor.

    Returns → Distant
        A Distant sensor plugin interface facing the direction
        specified by the angular configuration and pointing towards the origin
        :math:`(0, 0, 0)` in world coordinates.
    """

    dict_sensor = {
        "type": "distant",
        "direction": list(-angles_to_direction(theta=np.deg2rad(zenith),
                                               phi=np.deg2rad(azimuth))),
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
            "rfilter": {"type": "box"}
        }
    }

    return dict_sensor


def _make_default_scene():
    dict_scene = {
        "type": "scene",
        "bsdf_surface": {
            "type": "diffuse",
            "reflectance": {"type": "uniform", "value": 0.5}
        },
        "surface": {
            "type": "rectangle",
            "bsdf": {"type": "ref", "id": "bsdf_surface"}
        },
        "illumination": {
            "type": "directional",
            "direction": [0, 0, -1],
            "irradiance": {"type": "uniform", "value": 1.0}
        },
        "integrator": {"type": "path"}
    }

    return dict_scene


@attr.s
class OneDimSolver:
    r"""This class executes simulations on one-dimensional scenes, *i.e.* with 
    2 translational invariances.

    Constructor arguments / public attributes:
        ``scene`` (:class:`eradiate.scenes.builder.Scene`):
            Scene for which simulations will be run. If no value is 
            passed, a default scene is constructed. It consists of a square 
            covering :math:`[-1, 1]^2` with normal vector :math:`+Z` 
            with a Lambertian BRDF (reflectance :math:`\rho = 0.5`) illuminated
            directional emitter with direction vector :math:`-Z` and constant 
            irradiance equal to 1.
    """

    dict_scene = attr.ib(default=_make_default_scene())

    def run(self, vza=0., vaa=0., spp=3200):
        """Run the simulation for a set of specified sensor angular
        configurations.

        Parameter ``vza`` (float or array-like):
            Viewing zenith angles [deg].
        
        Parameter ``vaa`` (float or array-like): 
            Viewing azimuth angles [deg].

        Parameter ``spp`` (int): 
            Number of samples taken for each angular configuration.

        Returns → float or array: 
            Recorded leaving radiance.
        """

        # Ensure that vza and vaa are numpy arrays
        vza = ensure_array(vza)
        vaa = ensure_array(vaa)

        # Basic setup
        eradiate.kernel.set_variant("scalar_mono")
        from eradiate.kernel.core import Thread
        from eradiate.kernel.core.xml import load_dict
        Thread.thread().logger().clear_appenders()

        reflected_radiance = np.empty((len(vza), len(vaa)))

        for i, theta in enumerate(vza):
            for j, phi in enumerate(vaa):
                # Adjust scene setup
                self.dict_scene["sensor_distant"] = _make_distant(theta, phi, spp)

                # Run computation
                scene = load_dict(self.dict_scene)
                sensor = scene.sensors()[0]
                scene.integrator().render(scene, sensor)

                # Collect results
                film = sensor.film()
                result = float(np.array(film.bitmap(), dtype=float))
                reflected_radiance[i, j] = result

        # Fix result dimensionality (remove useless dims)
        try:
            return float(reflected_radiance)
        except TypeError:
            return np.squeeze(reflected_radiance)


def add_rayleigh_atmosphere(dict_scene):
    atmosphere = RayleighHomogeneous()
    dict_scene["integrator"] = {"type": "volpath"}
    dict_scene["atmosphere"] = atmosphere.shapes()["shape_atmosphere"]
    return dict_scene
