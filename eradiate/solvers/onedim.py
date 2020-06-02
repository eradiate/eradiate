from copy import deepcopy

import attr
import eradiate
import numpy as np

import eradiate.scenes.atmosphere.rayleigh
from ..scenes import atmosphere, illumination, measure, lithosphere
from ..util import ensure_array
from ..util.decorators import classproperty


@attr.s
class OneDimSolver(metaclass=classproperty.meta):
    r"""This class executes simulations on one-dimensional scenes, *i.e.* with
    2 translational invariances.

    Constructor arguments / public attributes:
        ``dict_scene`` (dict):
            Dictionary used to generate the scene for which simulations will be 
            run. If no value is passed, a default scene is constructed. It 
            consists of a square covering :math:`[-1, 1]^2` with normal vector 
            :math:`+Z` and a Lambertian BRDF (reflectance :math:`\rho = 0.5`) 
            illuminated by a directional emitter with direction vector 
            :math:`-Z` and constant irradiance equal to 1.
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

    def __attrs_post_init__(self):
        if self.dict_scene is None:
            self.dict_scene = self.DEFAULT_DICT_SCENE

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
        eradiate.kernel.set_variant("scalar_mono_double")
        from eradiate.kernel.core import Thread
        from eradiate.kernel.core.xml import load_dict
        Thread.thread().logger().clear_appenders()

        reflected_radiance = np.empty((len(vza), len(vaa)))

        for i, theta in enumerate(vza):
            for j, phi in enumerate(vaa):
                # Adjust scene setup
                self.dict_scene["sensor_distant"] = measure.distant(theta, phi, spp)

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


@attr.s
class RayleighSolverApp(metaclass=classproperty.meta):
    # Static methods
    @staticmethod
    def _select(params):
        switcher = {
            "constant": illumination.constant,
            "directional": illumination.directional,
            "lambert": lithosphere.Lambertian
        }

        params = deepcopy(params)
        callable_ = switcher[params["type"]]
        del params["type"]
        return callable_(**params)

    # Class attributes
    @classproperty
    def DEFAULT_CONFIG(cls):
        return {
            "mode": {
                "type": "mono",
                "wavelength": 550.
            },
            "illumination": {
                "type": "directional",
                "zenith": 180.,
                "azimuth": 0.,
                "irradiance": 1.
            },
            "measure": {
                "type": "distant",
                "zenith": 30.,
                "azimuth": 180.
            },
            "surface": {
                "type": "lambert",
                "reflectance": 0.5
            }
        }

    # Instance attributes
    config = attr.ib(default=None)
    _solver = attr.ib(default=OneDimSolver())

    def __attrs_post_init__(self):
        self.init()

    def init(self):
        if self.config is None:
            self.config = self.DEFAULT_CONFIG

        self._solver.init()
        self._configure_scene()

    def _configure_scene(self):
        # Set illumination
        config_illumination = self.config["illumination"]
        self._solver.dict_scene["illumination"] = self._select(config_illumination)

        # Set atmosphere
        try:
            config_atmosphere = self.config["atmosphere"]
            rayleigh_parameters = {"wavelength": self.config["mode"]["wavelength"]}
            atmosphere_ = eradiate.scenes.atmosphere.rayleigh.RayleighHomogeneous(
                scattering_coefficient=None,
                rayleigh_parameters=rayleigh_parameters,
                width=None, height=1e5
            )
            atmosphere_.add_to(self._solver.dict_scene, inplace=True)
        except KeyError:
            atmosphere_ = None

        # Set surface
        config_surface = self.config["surface"]
        if atmosphere_ is not None:
            config_surface["width"] = atmosphere_.width
        surface = self._select(config_surface)
        surface.add_to(self._solver.dict_scene, inplace=True)

    def run(self):
        self._solver.run()
