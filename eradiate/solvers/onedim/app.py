"""One dimensional solver application class and related facilities."""

import os
import warnings
from copy import deepcopy
from pathlib import Path

import attr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import eradiate.kernel
from .runner import OneDimRunner
from ...scenes.core import KernelDict, SceneElementFactory
from ...util import ensure_array, view
from ...util.config_object import ConfigObject
from ...util.exceptions import ConfigWarning
from ...util.units import config_default_units as cdu, ureg
from ...util.units import kernel_default_units as kdu
from ...util.xarray import eo_dataarray


@attr.s
class OneDimSolverApp(ConfigObject):
    r"""Application to run simulations in one-dimensional scenes.

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
            :factorykey:`rpv`,
            :factorykey:`black`

            Default:
            :factorykey:`lambertian`.

        ``atmosphere`` (dict or None):
            Section dedicated to configuring the scene's atmosphere.
            This section must be a factory configuration dictionary which will
            be passed to :meth:`eradiate.scenes.core.Factory.create`.
            If set to ``None``, no atmosphere is added to the scene.

            Allowed scene elements:
            :factorykey:`rayleigh_homogeneous`,
            :factorykey:`heterogeneous`.

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
            Section dedicated to measure definition. The one-dimensional solver
            currently supports two kinds of measurements:

            - Top of atmosphere leaving radiance over hemisphere

            - Top of atmosphere leaving radiance in principal plane

            Both measures record the outgoing radiance at the top of the configured
            atmosphere. Their central difference is that the principal plane measure
            records radiance for only one azimuth value, opposed to the hemispherical
            measure, which covers the entire hemisphere.

            .. admonition:: Note

                BRDF and BRF measures are computed from the outgoing radiance in
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

    .. rubric:: Constructor arguments / instance attributes

    ``config`` (dict):
        Configuration dictionary (see specification below).
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
                        "allowed": ["lambertian", "rpv", "black"],
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
                        "allowed": ["rayleigh_homogeneous", "heterogeneous"],
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
    results = attr.ib(init=False, factory=dict)

    def __attrs_post_init__(self):
        super(OneDimSolverApp, self).__attrs_post_init__()
        self.init()

    def init(self):
        """(Re)initialise hidden internal state.
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
                # Set illumination
                self._elements["illumination"] = SceneElementFactory.create(
                    self.config["illumination"])

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
                    config["surface"]["width"] = atmosphere._width

                self._elements["surface"] = SceneElementFactory.create(config["surface"])

                # Set measure
                for config_measure in self.config["measure"]:
                    if atmosphere is not None:
                        height, offset = atmosphere._height
                        height = height.to(kdu.get("length")).magnitude
                        offset = offset.to(kdu.get("length")).magnitude
                    else:
                        height = 0.1
                        offset = 0.001
                    config_measure["origin"] = [0, 0, height + offset]

                    if config_measure["type"] == "toa_lo_pplane":
                        if "orientation" not in config_measure:
                            # TODO: fix this behaviour (will crash if illumination.type is not directional)
                            #  Suggested change: raise if no orientation and illumination.type is not directional
                            phi_i = self._elements["illumination"].azimuth.to(ureg.rad).magnitude
                            config_measure["orientation"] = [np.cos(phi_i), np.sin(phi_i), 0]

                        config_measure["type"] = "radiancemeter_pplane"
                        config_measure["id"] = "toa_lo_pplane"
                        # TODO: allow custom ID definition and multiple definitions of the same measure
                        #  Use case: run the same measure with different SPP for comparison

                    elif config_measure["type"] == "toa_lo_hsphere":
                        config_measure["type"] = "radiancemeter_hsphere"
                        config_measure["id"] = "toa_lo_hsphere"

                    else:
                        raise ValueError(f"unsupported measure type '{config_measure['type']}'")

                    config_measure["hemisphere"] = "back"
                    # TODO: warn when overriding parameters set by user

                    if config_measure["id"] in self._elements:
                        raise AttributeError(
                            f"found multiple measures with identifier {config_measure['id']}; "
                            f"measure identifiers must be unique"
                        )

                    self._elements[config_measure["id"]] = SceneElementFactory.create(
                        config_measure)

                # Expand elements to kernel scene dictionary
                self._kernel_dict.add(list(self._elements.values()))

    def run(self):
        """Execute the computation and postprocess the results."""

        # Run simulation
        data = self._runner.run()

        # Post-processing
        # TODO: put that in a separate method
        # -- Ensure that scalar values used as xarray coordinates are arrays
        illumination = self._elements["illumination"]

        # -- Collect illumination parameters
        theta_i = ensure_array(illumination.zenith.to(ureg.deg).magnitude, dtype=float)
        phi_i = ensure_array(illumination.azimuth.to(ureg.deg).magnitude, dtype=float)
        wavelength = ensure_array(self.config["mode"]["wavelength"], dtype=float)
        # TODO: This will raise if illumination.type is not directional; handle that

        # -- Post-process TOA radiance arrays and compute BRDF/BRF
        for key, data in data.items():
            results = xr.Dataset()

            data = self._elements[key].repack_results(data)

            for dim in [0, 1, 4]:
                data = np.expand_dims(data, dim)

            zenith_res = self._elements[key].zenith_res

            if key == "toa_lo_hsphere":
                azimuth_res = self._elements[key].azimuth_res
                theta_o = np.arange(0., 90., zenith_res.to(ureg.deg).magnitude)
                phi_o = np.arange(0., 360., azimuth_res.to(ureg.deg).magnitude)
                angular_domain = "hsphere"

            elif key == "toa_lo_pplane":
                theta_o = np.arange(0., 90., zenith_res.to(ureg.deg).magnitude)
                phi_o = np.array([0., 180.])
                angular_domain = "pplane"

            else:
                raise ValueError(f"Unsupported measure type {key}")

            results[key] = eo_dataarray(data, theta_i, phi_i, theta_o, phi_o,
                                        wavelength, angular_domain=angular_domain)
            results[f"irradiance"] = (
                ("sza", "saa", "wavelength"),
                np.array(self._kernel_dict["illumination"]["irradiance"]["value"]).reshape(1, 1, 1),
                {
                    "long_name": "illumination spectral irradiance",
                    "units": "W/km^2/nm",
                    "angles_convention": "observation"
                }
            )

            # TODO: make metadata handling more robust
            # TODO: add support of CF convention-style metadata (discuss fields to include with Yvan)
            results[key.replace("lo", "brdf")] = results[key] / results["irradiance"]
            results[key.replace("lo", "brdf")].attrs = results[key].attrs
            results[key.replace("lo", "brf")] = results[key.replace("lo", "brdf")] / np.pi
            results[key.replace("lo", "brf")].attrs = results[key.replace("lo", "brdf")].attrs

            results.attrs = results[key].attrs
            self.results[key] = results

    def save_results(self, fname_prefix):
        """Save results to netCDF files.

        Parameter ``fname_prefix`` (str):
            Filename prefix for result storage. A netCDF file is created for
            each measure.
        """
        fname_prefix = Path(fname_prefix)
        for key, results in self.results.items():
            fname_results = f"{fname_prefix}_{key}.nc"
            os.makedirs(os.path.dirname(fname_results), exist_ok=True)
            print(f"Saving results to {fname_results}")
            results.to_netcdf(path=fname_results)

    def plot_results(self, fname_prefix):
        """Make default plots for stored results and save them to the hard
        drive.

        Parameter ``fname_prefix`` (str):
            Filename prefix for plot files. A plot file is create for each
            computed quantity of each measure.
        """
        for key, result in self.results.items():
            for quantity, data in result.items():
                if quantity == "irradiance":
                    continue

                if data.attrs["angular_domain"] == "hsphere":
                    ax = plt.subplot(111, projection="polar")
                    data.ert.plot(kind="polar_pcolormesh", title=quantity, ax=ax)

                elif data.attrs["angular_domain"] == "pplane":
                    ax = plt.subplot(111)
                    plane = view.plane(data)
                    plane.ert.plot(kind="linear", title=quantity, ax=ax)

                fname_plots = f"{fname_prefix}_{quantity}.png"
                os.makedirs(os.path.dirname(fname_plots), exist_ok=True)
                plt.savefig(fname_plots, bbox_inches="tight")
                plt.close()
