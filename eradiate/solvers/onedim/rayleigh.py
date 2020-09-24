"""Rayleigh solver application class and related facilities."""

# TODO: refactor into apps module?

import warnings
from copy import deepcopy

import attr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import eradiate.kernel
from .runner import OneDimRunner
from ...scenes.core import Factory, KernelDict
from ...util.units import kernel_default_units as kdu, config_default_units as cdu
from ...util import ensure_array, view
from ...util.config_object import ConfigObject
from ...util.exceptions import ConfigWarning
from ...util.xarray import eo_dataarray


@attr.s
class RayleighSolverApp(ConfigObject):
    r"""Application to run simulations in Rayleigh-scattering homogeneous
    one-dimensional scenes.

    Constructor arguments / public attributes:
        ``config`` (dict):
            Configuration dictionary (see specification below).

    .. admonition:: Configuration examples
        :class: hint

        Default:
            .. code:: yaml

               {
                   "mode": {
                       "type": "mono",
                       "wavelength": 550.
                   },
                   "surface": {
                       "type": "lambertian"
                   },
                   "atmosphere": {
                       "rayleigh_homogeneous"
                   },
                   "illumination": {
                       "type": "directional"
                   },
                   "measure": {
                       "type": "hemispherical",
                       "zenith_res": 10.,
                       "azimuth_res": 10.,
                       "spp": 1000
                   }
               }

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
                Selected wavelength [nm].

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
            :factorykey:`directional`,

            Default:
            :factorykey:`directional`.

        ``measure`` (dict):
            Section dedicated to measure definition.

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
    @classmethod
    def config_schema(cls):
        return dict({
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
    results = attr.ib(init=False, factory=xr.Dataset)

    def __attrs_post_init__(self):
        super(RayleighSolverApp, self).__attrs_post_init__()
        self.init()

    def init(self):
        r"""(Re)initialise hidden internal state.
        """
        # Select spectral mode based on configuration
        self._set_mode()

        # Reinitialise scene
        self._configure_scene()

        # Reinitialise solver
        self._runner = OneDimRunner(self._kernel_dict)

    def _set_mode(self):
        """Set spectral mode according to configuration. If a scene dictionary
        is already loaded, check that it is compatible with the requested
        spectral mode.
        """

        mode_config = deepcopy(self.config["mode"])
        mode_type = mode_config.pop("type")
        eradiate.set_mode(mode_type, **mode_config)

        # Check that scene dictionary is compatible with requested spectral mode
        if self._kernel_dict is not None:
            self._kernel_dict.check()

    def _configure_scene(self):
        factory = Factory()
        config = deepcopy(self.config)
        self._helpers = {}
        self._kernel_dict = KernelDict.empty()

        with cdu.override({"length": "km"}):
            with kdu.override({"length": "km"}):
                # Gather mode information
                wavelength = config["mode"]["wavelength"]

                # Set illumination
                self._helpers["illumination"] = factory.create(self.config["illumination"])

                # Set atmosphere
                config_atmosphere = config.get("atmosphere", None)

                if config_atmosphere is not None:
                    self._helpers["atmosphere"] = factory.create(config_atmosphere)

                # Set surface
                atmosphere = self._helpers.get("atmosphere", None)
                if atmosphere is not None:
                    if "width" in config["surface"].keys():
                        warnings.warn(
                            "overriding 'surface.width' with 'atmosphere.width'",
                            ConfigWarning
                        )
                    config["surface"]["width"] = atmosphere._width.magnitude
                    config["surface"]["width_unit"] = str(atmosphere._width.units)

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
            phi_o = np.array(
                [
                    illumination.config["illumination"]["azimuth"],
                    illumination.config["illumination"]["azimuth"] + 180.
                ] % 360.
            )

        else:
            raise ValueError(f"unsupported measure.type {measure_type}")

        # Run simulation
        data = self._runner.run(vza=theta_o, vaa=phi_o,
                                spp=self.config["measure"]["spp"],
                                squeeze=False,
                                show_progress=not quiet)
        for dim in [0, 1, 4]:
            data = np.expand_dims(data, dim)

        self.results["lo"] = eo_dataarray(data, theta_i, phi_i, theta_o, phi_o, wavelength)
        self.results["irradiance"] = (
            ("sza", "saa", "wavelength"),
            np.array(self._kernel_dict["illumination"]["irradiance"]["value"]).reshape(1, 1, 1),
            {
                "long_name": "illumination spectral irradiance",
                "units": "W/km^2/nm",
                "angles_convention": "eo_scene"
            }
        )

        self.results.attrs = self.results["lo"].attrs

    def postprocess(self):
        """Compute the TOA BRDF and TOA BRF from the raw results.
        The BRDF is computed by dividing the raw result by the incident radiance, while
        the BRF is computed by further dividing that result by the BRDF of a
        homogeneously reflecting (lambertian) surface.
        """
        # TODO: make metadata handling more robust
        # TODO: add support of CF convention-style metadata (discuss fields to include with Yvan)
        self.results["brdf"] = self.results["lo"] / self.results["irradiance"]
        self.results["brdf"].attrs = self.results["lo"].attrs
        self.results["brf"] = self.results["brdf"] / np.pi
        self.results["brf"].attrs = self.results["lo"].attrs

    def plot(self, plot_type=None, ax=None, result_type="brdf"):
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

        Parameter ``ax`` (:class:`~matplotlib.Axes`)
            Optional Axes object to embed the plot in a separate plotting script

        Parameter ``result_type`` (str)
            Result type to plot

            Currently supported options are:

            - ``lo``: Plots the leaving radiance
            - ``brdf``: Plots the ToA BRDF
            - ``brf``: Plots the ToA BRF

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

            hdata = self.results[result_type].ert.sel(
                theta_i=self.config["illumination"]["zenith"],
                phi_i=self.config["illumination"]["azimuth"],
                wavelength=self.config["mode"]["wavelength"]
            )

            hdata.ert.plot("pcolormesh", ax=ax)

        elif plot_type == "pplane":
            if ax is None:
                ax = plt.subplot(111)
            plt.title("Principal plane view")

            bhdata = self.results[result_type].ert.sel(wavelength=self.config['mode']['wavelength'])
            plane = view.pplane(bhdata,
                                theta_i=self.config['illumination']['zenith'],
                                phi_i=self.config['illumination']['azimuth'])

            plane.ert.plot(plane, ax=ax)

        else:
            raise ValueError(f"unsupported measure.type {plot_type}")

        return ax
