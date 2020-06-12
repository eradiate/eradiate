import warnings

import attr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import eradiate.kernel
from . import OneDimSolver
from ...scenes import SceneDict
from ...scenes.atmosphere import RayleighHomogeneous
from ...scenes.factory import Factory
from ...util import brdf_viewer as bv, ensure_array
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
    })

    # Instance attributes
    config = attr.ib(default=None)
    _scene_dict = attr.ib(default=None)
    _solver = attr.ib(default=None)
    result = attr.ib(init=False)

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
        self._scene_dict.add(factory.create(self.config["illumination"]))

        # Set atmosphere
        try:
            config_atmosphere = self.config["atmosphere"]

            try:
                sigma_s_params = config_atmosphere["sigma_s_params"]

                try:
                    wavelength_atmosphere = sigma_s_params["wavelength"]
                    if wavelength_atmosphere != wavelength:
                        warnings.warn("overriding 'atmosphere.sigma_s_params.wavelength' "
                                    "with 'mode.wavelength'", ConfigWarning)
                    config_atmosphere["sigma_s_params"]["wavelength"] = wavelength
                except KeyError:
                    config_atmosphere["sigma_s_params"]["wavelength"] = wavelength

            except KeyError:
                if "sigma_s" not in config_atmosphere:
                    config_atmosphere["sigma_s_params"] = {"wavelength": wavelength}

            atmosphere = RayleighHomogeneous(config_atmosphere)
            self._scene_dict.add(atmosphere)
        except KeyError:
            atmosphere = {}

        # Set surface
        if atmosphere:
            self.config["surface"]["width"] = atmosphere.config["width"]
        self._scene_dict.add(factory.create(self.config["surface"]))

    def run(self):

        # DataArray coords must be arrays
        theta_i = ensure_array(self.config['illumination']['zenith'])
        phi_i = ensure_array(self.config['illumination']['azimuth'])
        theta_o = ensure_array(self.config['measure']['zenith'])
        phi_o = ensure_array(self.config['measure']['azimuth'])
        wavelength = ensure_array(self.config['mode']['wavelength'])

        data = self._solver.run(vza=theta_o, vaa=phi_o, squeeze=False)
        for dim in [0, 1, 4]:
            data = np.expand_dims(data, dim)

        self.result = xr.DataArray(
            data,
            coords=[theta_i, phi_i, theta_o, phi_o, wavelength],
            dims=["theta_i", "phi_i", "theta_o", "phi_o", "wavelength"]
        )

    def visualize(self, plot_type, ax_ext=None, fname=None):
        """Generate the requested plot type with the :class:`BRDFView` and store
        the resulting figure in a file under the given path.

        Parameter ``plot_type`` (str)
            Sets the plot type to request from the :class:`BRDFView`.
            Currently supported options are:

            - ``hemispherical``: Plot scattering into the hemisphere around the
              scattering surface normal
            - ``principal_plane``: Plot scattering into the plane defined by the
              surface normal and the incoming light direction

        Parameter ``fname`` (str or PathLike)
            Location and file name to store the plot. File type is inferred from the suffix of
            this parameter.
        """

        if plot_type == "hemispherical":
            fig, ax = plt.subplot(1, 1, subplot_kw={'projection': 'polar'})
            plt.title("Hemispherical view")

            viewer = bv.HemisphericalView()
            viewer.wavelength = self.config['mode']['wavelength']
            viewer.wi = [self.config['illumination']['zenith'],
                         self.config['illumination']['azimuth']]
            viewer.brdf = self.result

        elif plot_type == "principal_plane":
            fig, ax = plt.subplot(1, 1)
            plt.title("Principal plane view")

            viewer = bv.PrincipalPlaneView()
            viewer.wavelength = self.config['mode']['wavelength']
            viewer.wi = [self.config['illumination']['zenith'],
                         self.config['illumination']['azimuth']]
            viewer.brdf = self.result

        else:
            raise ValueError(f"unsupported plot type '{plot_type}'")

        viewer.evaluate()

        if ax_ext is None:
            viewer.plot(ax=ax)
            plt.savefig(fname)
        else:
            viewer.plot(ax=ax_ext)
            return ax
