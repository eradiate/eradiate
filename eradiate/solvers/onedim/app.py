"""One dimensional solver application class and related facilities."""
import datetime
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
from ...scenes.measure import RadianceMeterHsphereMeasure, RadianceMeterPPlaneMeasure
from ...util import ensure_array
from ...util import plot as ertplt
from ...util import xarray as ertxr
from ...util.config_object import ConfigObject
from ...util.exceptions import ConfigWarning
from ...util.units import config_default_units as cdu
from ...util.units import kernel_default_units as kdu
from ...util.units import ureg


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
                       "type": "homogeneous"
                   },
                   "illumination": {
                       "type": "directional"
                   },
                   "measure": {
                       "type": "toa_hsphere",
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
            :factorykey:`homogeneous`,
            :factorykey:`heterogeneous`.

            Default:
            :factorykey:`homogeneous`.

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
            records radiance for only two azimuth values, opposed to the hemispherical
            measure, which covers the entire hemisphere.

            .. admonition:: Note

                This application records top-of-atmosphere radiance values and automatically computes the
                top-of-atmosphere BRDF and BRF from them. For convenience, these measures'
                types can be referred to using multiple aliases (see the type below).

            ``type`` (str):
                Selected measure type.

                Allowed values:

                * ``toa_hsphere``, ``toa_hsphere_brf``, ``toa_hsphere_lo``, ``toa_hsphere_brdf``: TOA leaving radiance, BRF and BRDF in the whole hemisphere;
                * ``toa_pplane``, ``toa_pplane_brf``, ``toa_pplane_lo``, ``toa_pplane_brdf``: TOA leaving radiance, BRF and BRDF in the principal plane.

                Default: ``toa_hsphere``.

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

            ``id`` (str):
                Identifier for the measure. Can be set arbitrarily to help users identify
                results in case of multiple measures in one computation. Will be used in
                the ``results`` attribute.

                Default: This value defaults to the value of the ``type`` attribute.

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
                        "allowed": ["homogeneous", "heterogeneous"],
                        "default": "homogeneous",
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
                        "type": "toa_hsphere",
                        "zenith_res": 10,
                        "azimuth_res": 10,
                        "spp": 32
                    }],
            }
        }

    _measure_aliases = {
        "toa_hsphere": ["toa_hsphere", "toa_hsphere_lo", "toa_hsphere_brdf", "toa_hsphere_brf"],
        "toa_pplane": ["toa_pplane", "toa_pplane_lo", "toa_pplane_brdf", "toa_pplane_brf"]
    }

    # Instance attributes
    _kernel_dict = attr.ib(default=None)
    _elements = attr.ib(default=None)
    _runner = attr.ib(default=None)
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
                    config["surface"]["width"] = atmosphere.kernel_width

                self._elements["surface"] = SceneElementFactory.create(config["surface"])

                # Set measure
                for config_measure in self.config["measure"]:
                    if atmosphere is not None:
                        height = atmosphere.kernel_height.magnitude
                        offset = atmosphere.kernel_offset.magnitude
                    else:
                        height = 0.1
                        offset = 0.001
                    config_measure["origin"] = [0, 0, height + offset]

                    if config_measure["type"] in self._measure_aliases["toa_pplane"]:
                        if "orientation" not in config_measure:
                            # TODO: fix this behaviour (will crash if illumination.type is not directional)
                            #  Suggested change: raise if no orientation and illumination.type is not directional
                            phi_i = self._elements["illumination"].azimuth.to(ureg.rad).magnitude
                            config_measure["orientation"] = [np.cos(phi_i), np.sin(phi_i), 0]

                        if not config_measure.get("id", None):
                            config_measure["id"] = config_measure["type"]
                        config_measure["type"] = "radiancemeter_pplane"

                    elif config_measure["type"] in self._measure_aliases["toa_hsphere"]:
                        if not config_measure.get("id", None):
                            config_measure["id"] = config_measure["type"]
                        config_measure["type"] = "radiancemeter_hsphere"

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
        """Execute the computation and postprocess the results.

        Results will be stored in the `results` attribute as a dictionary mapping each
        declared measure ID to a :class:`~xarray.Dataset` holding one variable per
        physical quantity computed by the measure."""

        # Run simulation
        runner_results = self._runner.run()

        # Post-processing
        # TODO: put that in a separate method
        # -- Ensure that scalar values used as xarray coordinates are arrays
        illumination = self._elements["illumination"]

        # -- Collect illumination parameters
        sza = ensure_array(illumination.zenith.to(ureg.deg).magnitude, dtype=float)
        cos_sza = np.cos(illumination.zenith.to(ureg.rad).magnitude)
        saa = ensure_array(illumination.azimuth.to(ureg.deg).magnitude, dtype=float)
        wavelength = ensure_array(self.config["mode"]["wavelength"], dtype=float)
        # TODO: This will raise if illumination.type is not directional; handle that

        # -- Post-process TOA radiance arrays and compute BRDF/BRF
        for sensor_id, data in runner_results.items():
            results = xr.Dataset()

            data = self._elements[sensor_id].repack_results(data)

            for dim in [0, 1, 4]:
                data = np.expand_dims(data, dim)

            zenith_res = self._elements[sensor_id].zenith_res

            if isinstance(self._elements[sensor_id], RadianceMeterHsphereMeasure):
                azimuth_res = self._elements[sensor_id].azimuth_res
                vza = np.arange(0., 90., zenith_res.to(ureg.deg).magnitude)
                vaa = np.arange(0., 360., azimuth_res.to(ureg.deg).magnitude)
                coord_specs_id = "angular_observation"

            elif isinstance(self._elements[sensor_id], RadianceMeterPPlaneMeasure):
                vza = np.arange(0., 90., zenith_res.to(ureg.deg).magnitude)
                vaa = np.array([0., 180.])
                coord_specs_id = "angular_observation_pplane"

            else:
                raise ValueError(f"Unsupported measure type {sensor_id}")

            results["lo"] = xr.DataArray(
                data,
                coords=(("sza", sza), ("saa", saa), ("vza", vza),
                        ("vaa", vaa), ("wavelength", wavelength))
            )
            results["irradiance"] = (
                ("sza", "saa", "wavelength"),
                np.array(
                    self._kernel_dict["illumination"]["irradiance"]["value"] *
                    cos_sza
                ).reshape((1, 1, 1))
            )

            results["brdf"] = results["lo"] / results["irradiance"]
            results["brf"] = results["brdf"] * np.pi

            if coord_specs_id.endswith("_pplane"):
                results = ertxr.pplane(results)

            # Add missing metadata
            dataset_spec = ertxr.DatasetSpec(
                convention="CF-1.8",
                title="Top-of-atmosphere simulation results",
                history=f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                        f"data creation - {__name__}.OneDimeSolverApp.run",
                source=f"eradiate, version {eradiate.__version__}",
                references="",
                var_specs={
                    "irradiance": ertxr.VarSpec(
                        standard_name="toa_horizontal_solar_irradiance_per_unit_wavelength",
                        units="W/km^2/nm",
                        long_name="top-of-atmosphere horizontal spectral irradiance"
                    ),
                    "lo": ertxr.VarSpec(
                        standard_name="toa_outgoing_radiance_per_unit_wavelength",
                        units="W/km^2/sr/nm",
                        long_name="top-of-atmosphere outgoing spectral radiance"
                    ),
                    "brf": ertxr.VarSpec(
                        standard_name="toa_brf",
                        units="dimensionless",
                        long_name="top-of-atmosphere bi-directional reflectance factor"
                    ),
                    "brdf": ertxr.VarSpec(
                        standard_name="toa_brdf",
                        units="1/sr",
                        long_name="top-of-atmosphere bi-directional reflection distribution function"
                    )
                },
                coord_specs=coord_specs_id
            )
            results.ert.normalize_metadata(dataset_spec)

            self.results[sensor_id] = results

    def save_results(self, fname_prefix):
        """Save results to netCDF files.

        Parameter ``fname_prefix`` (str):
            Filename prefix for result storage. A netCDF file is created for
            each measure.
        """
        fname_prefix = Path(fname_prefix)
        for measure_id, results in self.results.items():
            fname_results = os.path.abspath(f"{fname_prefix}_{measure_id}.nc")
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
        for measure_id, result in self.results.items():
            # Is the data hemispherical or plane?
            dataset_spec = ertxr.DatasetSpec(coord_specs="angular_observation")
            try:
                result.ert.validate_metadata(dataset_spec)
                is_hemispherical = True
            except ValueError:
                is_hemispherical = False

            for quantity, data in result.items():
                if quantity == "irradiance":
                    continue

                if is_hemispherical:
                    data.squeeze().ert.plot_pcolormesh_polar(r="vza", theta="vaa")
                    ax = plt.gca()
                    ertplt.remove_xylabels(ax)

                else:
                    data.squeeze().plot()

                fname_plot = os.path.abspath(f"{fname_prefix}_{measure_id}_{quantity}.png")
                os.makedirs(os.path.dirname(fname_plot), exist_ok=True)
                print(f"Saving plot to {fname_plot}")
                plt.savefig(fname_plot, bbox_inches="tight")
                plt.close()
