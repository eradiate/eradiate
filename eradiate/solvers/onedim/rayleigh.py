"""Rayleigh solver application class and related facilities."""

# TODO: refactor into apps module?

import warnings
from copy import deepcopy

import attr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import eradiate.kernel

from ...scenes.core import SceneElementFactory, KernelDict
from ...util import ensure_array, view
from ...util.config_object import ConfigObject
from ...util.exceptions import ConfigWarning
from ...util.units import config_default_units as cdu
from ...util.units import kernel_default_units as kdu
from ...util.xarray import eo_dataarray
from .runner import OneDimRunner


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
                       "type": "rayleigh_homogeneous"
                   },
                   "illumination": {
                       "type": "directional"
                   },
                   "measure": {
                       "type": "toa_lo_hsphere",
                       "zenith_res": 10.,
                       "azimuth_res": 10.,
                       "spp": 32
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

            Allowed scene elements:
            :factorykey:`lambertian`,
            :factorykey:`rpv`

            Default:
            :factorykey:`lambertian`.

        ``atmosphere`` (dict or None):
            Section dedicated to configuring the scene's atmosphere.
            This section must be a factory configuration dictionary which will
            be passed to :meth:`eradiate.scenes.core.Factory.create`.
            If set to ``None``, no atmosphere is added to the scene.

            Allowed scene elements:
            :factorykey:`rayleigh_homogeneous`.

            Default:
            :factorykey:`rayleigh_homogeneous`.

        ``illumination`` (dict):
            Section dedicated to configuring the scene's illumination.
            This section must be a factory configuration dictionary which will
            be passed to :meth:`eradiate.scenes.core.Factory.create`.

            Allowed scene elements:
            :factorykey:`constant`,
            :factorykey:`directional`,

            Default:
            :factorykey:`directional`.

        ``measure`` (list[dict]):
            Section dedicated to measure definition. The rayleigh one-dimensional solver
            currently supports two kinds of measurements:

            - Top of atmosphere leaving radiance over hemisphere

            - Top of atmosphere leaving radiance in principal plane

            Both measures record the leaving radiance at the top of the configured
            atmosphere. Their central difference is that the principal plane measure
            records radiance for only one azimuth value, opposed to the hemispherical
            measure, which covers the entire hemisphere.

            .. admonition:: Note

                BRDF and BRF measures are created from the leaving radiance in
                post-procressing and do not require dedicated scene elements.


            ``type`` (str):
                Selected measure type.

                Allowed values:
                ``toa_lo_hsphere`` (hemispherical),
                ``toa_lo_pplane`` (principal plane).

                Default: ``toa_lo_hsphere``.

            ``zenith_res`` (float):
                Zenith angle grid resolution [deg]. Default: 10.

                Unit-enabled field (default unit: cdu[angle])

            ``azimuth_res`` (float):
                **Hemispherical only!**

                Azimuth angle grid resolution. Default: 10.

                Unit-enabled field (default unit: cdu[angle])

            ``orientation`` (float):
                **Principal plane only!**

                Azimuth angle along which to align the principal plane. [deg]
                If not set, will match the direction of incoming radiation.

                Default: None

            ``spp`` (int):
                Number of samples taken for each viewing angle configuration.

                Default: 32.
    """

    # Class attributes
    #: Configuration validation schema
    @classmethod
    def config_schema(cls):
        return {
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
                "type": "list",
                "default":
                    [{
                        "type": "toa_lo_hsphere",
                        "zenith_res": 10,
                        "azimuth_res": 10,
                        "spp": 32
                    }],
            }
        }

    # Instance attributes
    _kernel_dict = attr.ib(default=None)
    _elements = attr.ib(default=None)
    _runner = attr.ib(default=None)
    _measure_map = attr.ib(default=None)
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
        config = deepcopy(self.config)
        self._elements = {}
        self._kernel_dict = KernelDict.empty()

        with cdu.override({"length": "km"}):
            with kdu.override({"length": "km"}):
                # Gather mode information
                wavelength = config["mode"]["wavelength"]

                # Set illumination
                self._elements["illumination"] = SceneElementFactory.create(self.config["illumination"])

                # Set atmosphere
                config_atmosphere = config.get("atmosphere", None)

                if config_atmosphere is not None:
                    self._elements["atmosphere"] = SceneElementFactory.create(config_atmosphere)

                # Set surface
                atmosphere = self._elements.get("atmosphere", None)
                if atmosphere is not None:
                    if "width" in config["surface"].keys():
                        warnings.warn(
                            "overriding 'surface.width' with 'atmosphere.width'",
                            ConfigWarning
                        )
                    config["surface"]["width"] = atmosphere._width.magnitude
                    config["surface"]["width_unit"] = str(atmosphere._width.units)

                self._elements["surface"] = SceneElementFactory.create(config["surface"])

                # Set measure
                for measure in self.config["measure"]:
                    if atmosphere is not None:
                        height, offset = atmosphere._height
                        height = height.to(kdu.get("length")).magnitude
                        offset = offset.to(kdu.get("length")).magnitude
                    else:
                        height = 0.1
                        offset = 0.001
                    measure["origin"] = [0, 0, height + offset]

                    if measure["type"] == "toa_lo_pplane":
                        if "orientation" not in measure:
                            phi_i = self._elements["illumination"].config["azimuth"]
                            measure["orientation"] = [np.cos(phi_i), np.sin(phi_i), 0]
                        element_config = deepcopy(measure)
                        element_config["type"] = "radiance_pplane"
                        element_config["id"] = "toa_lo_pplane"
                    else:
                        element_config = deepcopy(measure)
                        element_config["type"] = "radiancemeter_hsphere"
                        element_config["id"] = "toa_lo_hsphere"

                    element_config["hemisphere"] = "back"
                    measure_obj = SceneElementFactory.create(element_config)

                    if measure_obj.id in self._elements:
                        raise AttributeError(f"Multiple measures with ID {measure_obj.id}"
                                             f" found.  Ensure unique IDs for all measures.")
                    else:
                        self._elements[measure_obj.id] = measure_obj

                # Expand elements to kernel scene dictionary
                self._kernel_dict.add(list(self._elements.values()))

    def compute(self):
        # Ensure that scalar values used as xarray coordinates are arrays
        illumination = self._elements["illumination"]

        theta_i = ensure_array(illumination.zenith, dtype=float)
        phi_i = ensure_array(illumination.azimuth, dtype=float)
        wavelength = ensure_array(self.config["mode"]["wavelength"], dtype=float)

        # Run simulation
        data = self._runner.run()

        for key, data in data.items():

            data = self._elements[key].repack_results(data)

            for dim in [0, 1, 4]:
                data = np.expand_dims(data, dim)

            zenith_res = self._elements[key].zenith_res
            azimuth_res = self._elements[key].azimuth_res

            if key == "toa_lo_hsphere":
                theta_o = np.arange(0., 90., zenith_res)
                phi_o = np.arange(0., 360.001, azimuth_res)
                result_type = "hemisphere"
            # TODO: Fix this once the pplane scene element is merged
            # elif key == "toa_lo_pplane":
            #     theta_o = np.arange(0., 90., self.config["measure"]["zenith_res"])
            #     phi_o = np.array([0, 180])
            else:
                raise ValueError(f"Unsupported measure type {key}")

            self.results[key] = eo_dataarray(data, theta_i, phi_i, theta_o, phi_o,
                                             wavelength, result_type=result_type)
            self.results[f"irradiance"] = (
                ("sza", "saa", "wavelength"),
                np.array(self._kernel_dict["illumination"]["irradiance"]["value"]).reshape(1, 1, 1),
                {
                    "long_name": "illumination spectral irradiance",
                    "units": "W/km^2/nm",
                    "angles_convention": "eo_scene"
                }
            )
            self._postprocess(key)

            self.results.attrs = self.results[key].attrs

    def _postprocess(self, key):
        """Compute the TOA BRDF and TOA BRF from the raw results.
        The BRDF is computed by dividing the raw result by the incident radiance, while
        the BRF is computed by further dividing that result by the BRDF of a
        homogeneously reflecting (lambertian) surface.

        Parameter ``key`` (str)
            Key has to be one of the sensors' `id` attributes and identifies the result
            data to be processed.
        """
        # TODO: make metadata handling more robust
        # TODO: add support of CF convention-style metadata (discuss fields to include with Yvan)
        self.results[key.replace("lo", "brdf")] = self.results[key] / self.results["irradiance"]
        self.results[key.replace("lo", "brdf")].attrs = self.results[key].attrs
        self.results[key.replace("lo", "brf")] = self.results[key.replace("lo", "brdf")] / np.pi
        self.results[key.replace("lo", "brf")].attrs = self.results[key.replace("lo", "brdf")].attrs

    def plot(self, result_name, plot_type=None, ax=None):
        """Generate the requested plot type with the :class:`BRDFView` and store
        the resulting figure in a file under the given path.

        Parameter ``result_name`` (str)
            Chooses the result to plot. Result names depend on the measures configured
            in the scene.


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

        """
        result_type = self.results[result_name].attrs["result_type"]

        if plot_type is None:
            plot_type = result_type
        elif plot_type == "hemisphere" and result_type != "hemisphere":
            raise ValueError("hemispherical plot type requires hemispherical measure type")

        # Select plot based on requested measure type
        if plot_type == "hemisphere":
            if ax is None:
                ax = plt.subplot(111, projection="polar")
            plt.title("Hemispherical view")

            hdata = np.squeeze(self.results[result_name].ert.sel(
                theta_i=self.results["irradiance"]["sza"],
                phi_i=self.results["irradiance"]["saa"],
                wavelength=self.config["mode"]["wavelength"]
            ))

            hdata.ert.plot(kind="polar_pcolormesh", ax=ax)

        elif plot_type == "pplane":
            if ax is None:
                ax = plt.subplot(111)
            plt.title("Principal plane view")

            bhdata = self.results[result_name].ert.sel(wavelength=self.config['mode']['wavelength'])
            plane = view.pplane(bhdata,
                                theta_i=self.results["irradiance"]["sza"],
                                phi_i=self.results['irradiance']['saa'])

            plane.ert.plot(kind="linear", ax=ax)

        else:
            raise ValueError(f"unsupported measure.type {plot_type}")

        return ax
