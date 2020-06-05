import warnings
from copy import deepcopy

import attr
import eradiate.kernel

from . import OneDimSolver
from ...scenes import measure, SceneDict
from ...scenes.atmosphere import RayleighHomogeneous
from ...scenes.factory import Factory
from ...util.collections import frozendict
from ...util.exceptions import ConfigWarning


@attr.s
class RayleighSolverApp:
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
             component     available ``type`` values
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

    # Class attributes
    #: Default configuration
    DEFAULT_CONFIG = frozendict({
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
            "type": "lambertian",
            "reflectance": 0.5
        }
    })

    # Instance attributes
    config = attr.ib(default=None)
    _scene_dict = attr.ib(default=None)
    _solver = attr.ib(default=None)

    def __attrs_post_init__(self):
        if self.config is None:
            self.config = dict(self.DEFAULT_CONFIG)
        else:
            self.config = {**self.DEFAULT_CONFIG, **self.config}

        self.init()

    def init(self):
        r"""(Re)initialise hidden internal state.
        """
        # Select the kernel variant based on configuration
        self._set_kernel_variant()

        # Reinitialise scene
        self._scene_dict = SceneDict.empty()
        self._configure_scene()

        # Reinitialise solver
        self._solver = OneDimSolver(self._scene_dict)

    def _set_kernel_variant(self):
        """Set kernel variant according to scene dictionary. If scene dictionary
        has not been created, use mode information to set variant.
        """

        if self._scene_dict is not None:
            eradiate.kernel.set_variant(self._scene_dict.variant)
        else:
            mode = self.config["mode"]["type"]
            if mode == "mono":
                eradiate.kernel.set_variant("scalar_mono_double")
            else:
                raise ValueError(f"unsupported mode '{mode}'")

    def _configure_scene(self):
        factory = Factory()

        # Gather mode information
        wavelength = self.config["mode"]["wavelength"]

        # Set illumination
        illumination = factory.create(self.config["illumination"])
        illumination.add_to(self._scene_dict, inplace=True)

        # Set measure
        measure = factory.create(self.config["measure"])
        measure.add_to(self._scene_dict, inplace=True)

        # Set atmosphere
        try:
            config_atmosphere = self.config["atmosphere"]

            try:
                wavelength_atmosphere = config_atmosphere["sigmas_params"]["wavelength"]
                if wavelength_atmosphere != wavelength:
                    warnings.warn("overriding 'atmosphere.sigmas_params.wavelength' "
                                  "with 'mode.wavelength'", ConfigWarning)
                    config_atmosphere["sigmas_params"]["wavelength"] = wavelength
            except KeyError:
                pass
            atmosphere = RayleighHomogeneous(config_atmosphere)
            atmosphere.add_to(self._scene_dict, inplace=True)
        except KeyError:
            atmosphere = {}

        # Set surface
        if atmosphere:
            self.config["surface"]["width"] = atmosphere.config["width"]
        surface = factory.create(self.config["surface"])
        surface.add_to(self._scene_dict, inplace=True)

    def run(self):
        self._solver.run()
