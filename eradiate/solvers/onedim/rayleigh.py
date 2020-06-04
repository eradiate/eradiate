from copy import deepcopy

import attr

from . import OneDimSolver
from ...scenes import illumination, lithosphere
from ...scenes.atmosphere import RayleighHomogeneous
from ...util.decorators import classproperty


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
            atmosphere = RayleighHomogeneous(scattering_coefficient=None,
                                             rayleigh_parameters=rayleigh_parameters,
                                             width=None,
                                             height=1e5)
            atmosphere.add_to(self._solver.dict_scene, inplace=True)
        except KeyError:
            atmosphere = None

        # Set surface
        config_surface = self.config["surface"]
        if atmosphere is not None:
            config_surface["width"] = atmosphere.width
        surface = self._select(config_surface)
        surface.add_to(self._solver.dict_scene, inplace=True)

    def run(self):
        self._solver.run()
