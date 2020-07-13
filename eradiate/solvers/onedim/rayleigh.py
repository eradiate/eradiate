"""Rayleigh solver application class and related facilities."""

# TODO: refactor into apps module?

import warnings
from copy import deepcopy

import attr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import eradiate.kernel
from . import OneDimSolver
from ...scenes.core import Factory, KernelDict
from ...util import brdf_viewer as bv, ensure_array
from ...util.collections import frozendict
from ...util.config_object import ConfigObject
from ...util.exceptions import ConfigWarning


@attr.s
class RayleighSolverApp(ConfigObject):
    r"""Application to run simulations in Rayleigh-scattering homogeneous
    one-dimensional scenes.

    Constructor arguments / public attributes:
        ``config`` (dict):
            Configuration dictionary (see specification below).

    .. admonition:: Configuration format
        :class: hint

        ``mode`` (dict):
            Section dedicated to operation mode selection.

            ``type`` (str):
                Operational mode selection.

                Allowed values:
                ``mono``.

                Default: ``mono``.

            ``wavelength`` (float):
                Selected wavelength.

                Default: 550.0.

        ``surface`` (dict):
            Section dedicated to configuring the scene's surface.
            This section must be a factory configuration dictionary which will
            be passed to :meth:`eradiate.scenes.core.Factory.create`.

            Allowed scene generation helpers:
            :factorykey:`lambertian`,
            :factorykey:`rpv`

            Default:
            :factorykey:`lambertian`.

        ``atmosphere`` (dict or None):
            Section dedicated to configuring the scene's atmosphere.
            This section must be a factory configuration dictionary which will
            be passed to :meth:`eradiate.scenes.core.Factory.create`.
            If set to ``None``, no atmosphere is added to the scene.

            Allowed scene generation helpers:
            :factorykey:`rayleigh_homogeneous`.

            Default:
            :factorykey:`rayleigh_homogeneous`.

        ``illumination`` (dict):
            Section dedicated to configuring the scene's illumination.
            This section must be a factory configuration dictionary which will
            be passed to :meth:`eradiate.scenes.core.Factory.create`.

            Allowed scene generation helpers:
            :factorykey:`constant`,
            :factorykey:`directional`.

            Default:
            :factorykey:`directional`.

        ``measure`` (dict):
            Section dedicated to measure definition.

            ``type`` (str):
                Operational mode selection.

                Allowed values: ``mono``.

                Default: ``mono``.

            ``wavelength`` (float):
                Selected wavelength.

                Default: 550.0.

            ``type`` (str):
                Selected measure type.

                Allowed values:
                ``hemispherical`` (hemispherical),
                ``pplane`` (principal plane).

                Default: ``hemispherical``.

            ``zenith_res`` (float):
                Zenith angle grid resolution [deg].

                Default: 10.

            ``azimuth_res`` (float):
                Azimuth angle grid resolution [deg].

                Default: 10.

            ``spp`` (int):
                Number of samples taken for each viewing angle configuration.

                Default: 1000.
    """

    # Class attributes
    #: Configuration validation schema
    CONFIG_SCHEMA = frozendict({
        "mode": {
            "type": "dict",
            "default": {},
            "schema": {
                "type": {
                    "type": "string",
                    "allowed": ["mono"],
                    "default": "mono"
                },
                "wavelength": {"type": "number", "min": 0.0, "default": 550.0},
            }
        },
        "surface": {
            "type": "dict",
            "default": {},
            "allow_unknown": True,
            "schema": {
                "type": {
                    "type": "string",
                    "allowed": ["lambertian", "rpv"],
                    "default": "lambertian"
                },
            }
        },
        "atmosphere": {
            "type": "dict",
            "nullable": True,
            "default": {},
            "allow_unknown": True,
            "schema": {
                "type": {
                    "type": "string",
                    "allowed": ["rayleigh_homogeneous"],
                    "default": "rayleigh_homogeneous",
                },
            }
        },
        "illumination": {
            "type": "dict",
            "default": {},
            "allow_unknown": True,
            "schema": {
                "type": {
                    "type": "string",
                    "allowed": ["directional", "constant"],
                    "default": "directional"
                },
            },
        },
        "measure": {
            "type": "dict",
            "default": {},
            "schema": {
                "type": {
                    "type": "string",
                    "allowed": ["hemispherical", "pplane"],
                    "default": "hemispherical",
                },
                "zenith_res": {
                    "type": "number",
                    "min": 1.0,
                    "default": 10.0,
                },
                "azimuth_res": {
                    "type": "number",
                    "min": 1.0,
                    "default": 10.0,
                },
                "spp": {
                    "type": "integer",
                    "min": 1,
                    "default": 1000,
                }
            }
        }
    })

    # Instance attributes
    _kernel_dict = attr.ib(default=None)
    _helpers = attr.ib(default=None)
    _runner = attr.ib(default=None)
    results = attr.ib(init=False)

    def __attrs_post_init__(self):
        super(RayleighSolverApp, self).__attrs_post_init__()
        self.init()

    def init(self):
        r"""(Re)initialise hidden internal state.
        """
        # Select the kernel variant based on configuration
        self._set_kernel_variant()

        # Reinitialise scene
        self._configure_scene()

        # Reinitialise solver
        self._runner = OneDimSolver(self._kernel_dict)

    def _set_kernel_variant(self):
        """Set kernel variant according to kernel scene dictionary. If scene
        dictionary has not been created, use mode information to set variant.
        """

        if self._kernel_dict is not None:
            eradiate.kernel.set_variant(self._kernel_dict.variant)
        else:
            mode = self.config["mode"]["type"]
            if mode == "mono":
                eradiate.kernel.set_variant("scalar_mono_double")
            else:
                raise ValueError(f"unsupported mode '{mode}'")

    def _configure_scene(self):
        factory = Factory()
        config = deepcopy(self.config)
        self._helpers = {}
        self._kernel_dict = KernelDict.empty()

        # Gather mode information
        wavelength = config["mode"]["wavelength"]

        # Set illumination
        self._helpers["illumination"] = factory.create(config["illumination"])

        # Set atmosphere
        config_atmosphere = config.get("atmosphere", None)

        if config_atmosphere is not None:
            try:
                sigma_s = config_atmosphere["sigma_s"]

                try:
                    wavelength_atmosphere = sigma_s["wavelength"]
                    if wavelength_atmosphere != wavelength:
                        warnings.warn(
                            "overriding 'atmosphere.sigma_s.wavelength' "
                            "with 'mode.wavelength'",
                            ConfigWarning
                        )

                except KeyError:
                    # sigma_s does not contain wavelength spec:
                    # we add it
                    config_atmosphere["sigma_s"]["wavelength"] = wavelength

                except TypeError:
                    # sigma_s is a number: we leave it as it is
                    pass

            except KeyError:
                # config_atmosphere is missing a scattering coefficient
                # specification: just add it
                config_atmosphere["sigma_s"] = {"wavelength": wavelength}

            self._helpers["atmosphere"] = factory.create(config_atmosphere)

        # Set surface
        atmosphere = self._helpers.get("atmosphere", None)
        if atmosphere is not None:
            if "width" in config["surface"].keys():
                warnings.warn(
                    "overriding 'surface.width' with 'atmosphere.width'",
                    ConfigWarning
                )
            config["surface"]["width"] = atmosphere._width

        self._helpers["surface"] = factory.create(config["surface"])

        # Expand helpers to kernel scene dictionary
        self._kernel_dict.add(list(self._helpers.values()))

    def compute(self, quiet=False):
        # Ensure that scalar values used as xarray coordinates are arrays
        illumination = self._helpers["illumination"]

        theta_i = ensure_array(illumination.config["zenith"], dtype=float)
        phi_i = ensure_array(illumination.config["azimuth"], dtype=float)
        wavelength = ensure_array(self.config["mode"]["wavelength"], dtype=float)

        # Process measure angles
        measure_type = self.config["measure"]["type"]

        if measure_type == "hemispherical":
            theta_o = np.arange(0., 90., self.config["measure"]["zenith_res"])
            phi_o = np.arange(0., 360.001, self.config["measure"]["azimuth_res"])

        elif measure_type == "pplane":
            theta_o = np.arange(0., 90., self.config["measure"]["zenith_res"])
            phi_o = np.array([
                self.config["illumination"]["azimuth"],
                self.config["illumination"]["azimuth"] + 180.
            ] % 360.)

        else:
            raise ValueError(f"unsupported measure.type {measure_type}")

        # Run simulation
        data = self._runner.run(vza=theta_o, vaa=phi_o,
                                spp=self.config["measure"]["spp"],
                                squeeze=False,
                                show_progress=not quiet)
        for dim in [0, 1, 4]:
            data = np.expand_dims(data, dim)

        # Store results to an xarray.DataArray
        self.results = xr.DataArray(
            data,
            coords=[theta_i, phi_i, theta_o, phi_o, wavelength],
            dims=["theta_i", "phi_i", "theta_o", "phi_o", "wavelength"]
        )

    def plot(self, plot_type=None, ax=None):
        """Generate the requested plot type with the :class:`BRDFView` and store
        the resulting figure in a file under the given path.

        Parameter ``plot_type`` (str or None)
            Sets the plot type to request from the :class:`BRDFView`. If set to
            `None`, the plot type is selected based on the ``measure.type``
            configuration parameter.

            Currently supported options are:

            - ``hemispherical``: Plot scattering into the hemisphere around the
              scattering surface normal (available only if ``measure.type`` is
              ``hemispherical``).
            - ``pplane``: Plot scattering into the plane defined by the
              surface normal and the incoming light direction.

        Parameter ``fname`` (str or PathLike)
            Location and file name to store the plot. File type is inferred from
            the suffix of this parameter upon call of
            :func:`~matplotlib.pyplot.savefig`.
        """
        measure_type = self.config["measure"]["type"]

        if plot_type is None:
            plot_type = measure_type
        elif plot_type == "hemispherical" and measure_type != "hemispherical":
            raise ValueError("hemispherical plot type requires hemispherical measure type")

        # Select plot based on requested measure type
        if plot_type == "hemispherical":
            if ax is None:
                ax = plt.subplot(111, projection="polar")
            plt.title("Hemispherical view")

            viewer = bv.HemisphericalView()
            viewer.wavelength = self.config["mode"]["wavelength"]
            viewer.wi = [self.config["illumination"]["zenith"],
                         self.config["illumination"]["azimuth"]]
            viewer.brdf = self.results

        elif plot_type == "pplane":
            if ax is None:
                ax = plt.subplot(111)
            plt.title("Principal plane view")

            viewer = bv.PrincipalPlaneView()
            viewer.wavelength = self.config["mode"]["wavelength"]
            viewer.wi = [self.config["illumination"]["zenith"],
                         self.config["illumination"]["azimuth"]]
            viewer.brdf = self.results

        else:
            raise ValueError(f"unsupported measure.type {plot_type}")

        viewer.evaluate()
        viewer.plot(ax=ax)

        return ax
