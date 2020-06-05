""" This module contains basic facilities to run simulations on one-dimensional
scenes. """

import attr
import numpy as np

import eradiate
from ...scenes import measure
from ...util import ensure_array
from ...util.decorators import classproperty


@attr.s
class OneDimSolver(metaclass=classproperty.meta):
    r"""Execute simulations on one-dimensional scenes, *i.e.* with 2
    translational invariances.

    Constructor arguments / public attributes:
        ``dict_scene`` (dict):
            Dictionary used to generate the scene for which simulations will be
            run. If no value is passed, a default scene is constructed. It
            consists of a square covering :math:`[-1, 1]^2` with normal vector
            :math:`+Z` and a Lambertian BRDF (reflectance :math:`\rho = 0.5`)
            illuminated by a directional emitter with direction vector
            :math:`-Z` and constant irradiance equal to 1. If set to `None`,
            defaults to :data:`DEFAULT_DICT_SCENE`.
        ``variant`` (str):
            Kernel variant to use for the simulation. By default, the
            ``scalar_mono_double`` variant is used.

    """

    @classproperty
    def DEFAULT_DICT_SCENE(self):
        return {
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

    dict_scene = attr.ib(default=None)
    variant = attr.ib(default="scalar_mono_double")

    @variant.validator
    def _check_variant(self, attribute, value):
        if value not in ["scalar_mono", "scalar_mono_double"]:
            raise ValueError(f"unsupported kernel variant '{value}'")

    def __attrs_post_init__(self):
        if self.dict_scene is None:
            self.dict_scene = self.DEFAULT_DICT_SCENE

    def run(self, vza=0., vaa=0., spp=3200):
        """Run the simulation for a set of specified sensor angular
        configurations.

        The solver uses the variant stored in its :data:`variant` instance
        attribute.

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
        eradiate.kernel.set_variant(self.variant)
        from eradiate.kernel.core import Thread
        from eradiate.kernel.core.xml import load_dict
        Thread.thread().logger().clear_appenders()

        reflected_radiance = np.empty((len(vza), len(vaa)))

        for i, theta in enumerate(vza):
            for j, phi in enumerate(vaa):
                # Adjust scene setup
                self.dict_scene["sensor_distant"] = measure._distant(theta, phi, spp)

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
