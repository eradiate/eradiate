from copy import deepcopy

import attr

from . import OneDimSolver
from ...scenes import measure
from ...scenes.atmosphere import RayleighHomogeneous
from ...scenes.factory import Factory
from ...util.decorators import classproperty


@attr.s
class RayleighSolverApp(metaclass=classproperty.meta):
    r"""Application to run simulations in Rayleigh-scattering homogeneous
    one-dimensional scenes.

    Constructor arguments / public attributes:
        ``config`` (dict):
            Configuration dictionary (see specification below).

    Configuration format:
        This class is initialised with a configuration dictionary with the
        following keys:

            - ``mode``              (required)
            - ``illumination``      (required)
            - ``measure``           (required)
            - ``surface``           (required)
            - ``atmosphere``        (optional)

        For each of these keys, the corresponding value is a dictionary that
        specifies the configuration of the given element. That configuration
        includes at least a ``type`` entry and a number of additional optional
        parameters. This dictionary is merged with the default one
        (see :data:`DEFAULT_CONFIG`): if parameters are omitted, they will be
        set to their default values.

        .. admonition:: Example

            The following dictionary configures a Lambertian surface with a
            reflectance value of 0.3:

            .. code:: python

                {
                    "type": "lambertian",
                    "reflectance": 0.3
                }

        Supported configuration entries:
            ============== ===========================
             component     available ``type``values
            ============== ===========================
             mode           'mono'
             illumination   'constant', 'directional'
             measure        'distant'
             surface        'lambertian'
             atmosphere     'rayleigh_homogeneous'
            ============== ===========================

        Configuration parameters:
            ``mono``
                - ``wavelength`` (float or list): wavelength at which to run the
                  simulation.

            ``constant``
                Parameters of :func:`eradiate.scene.illumination.constant`

            ``directional``
                Parameters of :func:`eradiate.scenes.illumination.directional`

            ``distant``
                Parameters of :func:`eradiate.scenes.measure.distant`

            ``lambertian``
                Attributes of :class: eradiate.scenes.lithosphere.Lambert
                Note: if an atmosphere is set, `width` will be set to the
                atmosphere's width.

            ``rayleigh_homogeneous``
                Attributes of
                :class:`eradiate.scenes.atmosphere.rayleigh.RayleighHomogeneous`

    """

    # Static methods
    @staticmethod
    def _select(params):
        switcher = {
            "distant": measure._distant,
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
                "zenith": 0.,
                "azimuth": 0.,
                "irradiance": 1.
            },
            "measure": {
                "type": "distant",
                "zenith": 30.,
                "azimuth": 180.
            },
            "surface": {
                "type": "lambertian",
                "reflectance": 0.5
            }
        }

    # Instance attributes
    config = attr.ib(default=None)
    _solver = attr.ib(default=OneDimSolver())

    def __attrs_post_init__(self):
        self.init()

    def init(self):
        """(Re)initialise hidden internal state."""
        if self.config is None:
            self.config = self.DEFAULT_CONFIG

        #self._solver.init()
        self._configure_scene()

    def _configure_scene(self):
        factory = Factory()

        # Set illumination
        illumination = factory.create(self.config["illumination"])
        illumination.add_to(self._solver.dict_scene, inplace=True)

        # Set measure
        # TODO: refactor using factory
        config_measure = self.config["measure"]
        self._solver.dict_scene["measure"] = self._select(config_measure)

        # Set atmosphere
        # TODO: refactor using factory
        try:
            config_atmosphere = self.config["atmosphere"]

            if 'scattering_coefficient' not in config_atmosphere:
                try:
                    wavelength = self.config["mode"]["wavelength"]
                except KeyError:
                    wavelength = 550.
                config_atmosphere['wavelength'] = wavelength

            atmosphere = RayleighHomogeneous(**config_atmosphere)
            atmosphere.add_to(self._solver.dict_scene, inplace=True)
        except KeyError:
            atmosphere = None

        # Set surface
        if atmosphere is not None:
            self.config["surface"]["width"] = atmosphere.width
        surface = factory.create(self.config["surface"])
        surface.add_to(self._solver.dict_scene, inplace=True)

    def run(self):
        self._solver.run()
