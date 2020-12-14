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
from tinydb import Query, TinyDB
from tinydb.storages import MemoryStorage

import eradiate.kernel
from .runner import OneDimRunner
from ...scenes.atmosphere import AtmosphereFactory, HomogeneousAtmosphere
from ...scenes.atmosphere.base import Atmosphere
from ...scenes.core import KernelDict, SceneElement
from ...scenes.illumination import (
    DirectionalIllumination, Illumination, IlluminationFactory
)
from ...scenes.integrators import (
    Integrator, IntegratorFactory, VolPathIntegrator
)
from ...scenes.measure import (
    MeasureFactory, RadianceMeterHsphereMeasure, RadianceMeterPlaneMeasure
)
from ...scenes.surface import LambertianSurface, Surface, SurfaceFactory
from ...util import plot as ertplt
from ...util import xarray as ertxr
from ...util.config_object import ConfigObject
from ...util.exceptions import ConfigWarning
from ...util.misc import always_iterable, ensure_array
from ...util.units import config_default_units as cdu
from ...util.units import kernel_default_units as kdu
from ...util.units import ureg


@MeasureFactory.register(
    "toa_hsphere", "toa_hsphere_lo", "toa_hsphere_brdf", "toa_hsphere_brf"
)
@attr.s
class TOAHsphereMeasure(RadianceMeterHsphereMeasure):
    """Top-of-atmosphere radiancemeter (hemisphere coverage). This class is a
    lightweight specialisation of its :class:`RadianceMeterHsphereMeasure`
    parent class and should only be used with :class:`OneDimScene`.
    It is registered to :class:`.MeasureFactory` with the following keys:
    ``"toa_hsphere"``, ``"toa_hsphere_lo"``, ``"toa_hsphere_brdf"``,
    ``"toa_hsphere_brf"``.
    """
    id = attr.ib(default="toa_hsphere")

    # Only the back hemisphere is supported
    hemisphere = attr.ib(
        default="back",
        validator=attr.validators.in_(("back",)),
    )


@MeasureFactory.register(
    "toa_pplane", "toa_pplane_lo", "toa_pplane_brdf", "toa_pplane_brf"
)
@attr.s
class TOAPPlaneMeasure(RadianceMeterPlaneMeasure):
    """Top-of-atmosphere radiancemeter (principal plane coverage).
    This class is a lightweight specialisation of its
    :class:`RadianceMeterPlaneMeasure` parent class and should only be used
    with :class:`OneDimScene`.
    It is registered to :class:`.MeasureFactory` with the following keys:
    ``"toa_pplane"``, ``"toa_pplane_lo"``, ``"toa_pplane_brdf"``,
    ``"toa_pplane_brf"``.
    """
    id = attr.ib(default="toa_pplane")

    # Only the back hemisphere is supported
    hemisphere = attr.ib(
        default="back",
        validator=attr.validators.in_(("back",)),
    )


@attr.s
class OneDimScene(SceneElement):
    """Scene abstraction suitable for radiative transfer simulation on
    one-dimensional scenes.

    .. rubric:: Constructor arguments / instance attributes

    ``atmosphere`` (:class:`.Atmosphere` or dict):
        Atmosphere specification.
        This parameter can be specified as a dictionary which will be
        interpreted by
        :meth:`AtmosphereFactory.convert() <.AtmosphereFactory.convert>`.
        Default: :class:`.HomogeneousAtmosphere`.

    ``surface`` (:class:`.Surface` or dict):
        Surface specification.
        This parameter can be specified as a dictionary which will be
        interpreted by
        :meth:`SurfaceFactory.convert() <.SurfaceFactory.convert>`.
        Default: :class:`.LambertianSurface`.

    ``illumination`` (:class:`.Illumination` or dict):
        Illumination specification.
        This parameter can be specified as a dictionary which will be
        interpreted by
        :meth:`IlluminationFactory.convert() <.IlluminationFactory.convert>`.
        Default: :class:`.DirectionalIllumination`.

    ``measures`` (list[:class:`.Measure`] or list[dict] or :class:`.Measure` or dict):
        List of measure specifications. The passed list may contain dictionary,
        which will be interpreted by
        :meth:`MeasureFactory.convert() <.MeasureFactory.convert>`.
        Optionally, a single :class:`.Measure` or dictionary specification
        may be passed and will automatically be wrapped into a list.
        Allowed values: :class:`TOAHsphereMeasure`, :class:`TOAPPlaneMeasure`.
        Default: :class:`TOAHsphereMeasure`.

    ``integrator`` (:class:`.Integrator` or dict):
        Monte Carlo integration algorithm specification.
        This parameter can be specified as a dictionary which will be
        interpreted by
        :meth:`IntegratorFactory.convert() <.IntegratorFactory.convert>`.
        Default: :class:`.VolPathIntegrator`.
    """
    atmosphere = attr.ib(
        factory=HomogeneousAtmosphere,
        converter=AtmosphereFactory.convert,
        validator=attr.validators.instance_of(Atmosphere),
    )

    surface = attr.ib(
        factory=LambertianSurface,
        converter=SurfaceFactory.convert,
        validator=attr.validators.instance_of(Surface)
    )

    illumination = attr.ib(
        factory=DirectionalIllumination,
        converter=IlluminationFactory.convert,
        validator=attr.validators.instance_of(Illumination)
    )

    measures = attr.ib(
        factory=lambda: [TOAHsphereMeasure()],
        converter=lambda value:
        [MeasureFactory.convert(x) for x in always_iterable(value)]
        if not isinstance(value, dict)
        else [MeasureFactory.convert(value)]
    )

    @measures.validator
    def _measures_validator(self, attribute, value):
        for element in value:
            # Check measure type
            if not isinstance(element, (TOAPPlaneMeasure, TOAHsphereMeasure)):
                raise TypeError(
                    f"while validating {attribute.name}: must be a list of "
                    f"objects of one of the following types: "
                    f"(TOAPPlaneMeasure, TOAHSphereMeasure)"
                )

            # Principal plane measures only work with directional illumination
            if isinstance(element, TOAPPlaneMeasure) and \
                    not isinstance(self.illumination, DirectionalIllumination):
                raise ValueError(
                    f"while validating {attribute.name}: found measure of type "
                    f"{element.__class__.__name__}, incompatible with "
                    f"illumination of type "
                    f"{self.illumination.__class__.__name__}"
                )

    integrator = attr.ib(
        factory=VolPathIntegrator,
        converter=IntegratorFactory.convert,
        validator=attr.validators.instance_of(Integrator)
    )

    _measure_registry = attr.ib(
        init=False,
        repr=False,
        factory=lambda: TinyDB(storage=MemoryStorage)
    )

    def __attrs_post_init__(self):
        # Parts of the init sequence we could take care of using converters

        # Override surface width with atmosphere width
        self.surface.width = self.atmosphere.kernel_width

        # Process measures
        for measure in self.measures:
            if isinstance(measure, (TOAHsphereMeasure, TOAPPlaneMeasure)):
                # Override ray origin
                if self.atmosphere is not None:
                    sensor_height = self.atmosphere.kernel_height.to(cdu.get("length")).magnitude
                    measure.origin = [0., 0., sensor_height]

            if isinstance(measure, TOAPPlaneMeasure):
                # Set principal plane measure orientation if any
                phi_i = self.illumination.azimuth.to(ureg.rad).magnitude
                measure.orientation = [np.cos(phi_i), np.sin(phi_i), 0.]

        # Populate measure registry
        for measure in self.measures:
            for sensor_id, sensor_spp in measure.sensor_info():
                self._measure_registry.insert(
                    {
                        "measure_id": measure.id,
                        "sensor_id": sensor_id,
                        "sensor_spp": sensor_spp
                    }
                )

    def kernel_dict(self, ref=True):
        result = KernelDict.empty()
        result.add([
            self.atmosphere,
            self.surface,
            self.illumination,
            *self.measures,
            self.integrator
        ])
        return result


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
                        "allowed": ["mono", "mono_double"],
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
    _measure_registry = attr.ib(init=False,
                                factory=lambda: TinyDB(storage=MemoryStorage))
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

        with cdu.override({"length": "km"}), kdu.override({"length": "km"}):
            # Set illumination
            self._elements["illumination"] = \
                IlluminationFactory.create(self.config["illumination"])

            # Set atmosphere
            config_atmosphere = config.get("atmosphere", None)

            if config_atmosphere is not None:
                self._elements["atmosphere"] = \
                    AtmosphereFactory.create(config_atmosphere)

            # Set surface
            atmosphere = self._elements.get("atmosphere", None)
            if atmosphere is not None:
                if "width" in config["surface"].keys():
                    warnings.warn(
                        "overriding 'surface.width' with 'atmosphere.width'",
                        ConfigWarning
                    )
                config["surface"]["width"] = atmosphere.kernel_width

            self._elements["surface"] = SurfaceFactory.create(config["surface"])

            # Set measure
            for config_measure in self.config["measure"]:
                if atmosphere is not None:
                    offset = atmosphere.kernel_offset.to(cdu.get("length")).magnitude
                    height = atmosphere.kernel_height.to(cdu.get("length")).magnitude - offset

                else:
                    height = 0.1
                    offset = 1e-3
                config_measure["origin"] = [0, 0, height + offset]

                if config_measure["type"] in self._measure_aliases["toa_pplane"]:
                    if "orientation" not in config_measure:
                        # TODO: fix this behaviour (will crash if illumination.type is not directional)
                        #  Suggested change: raise if no orientation and illumination.type is not directional
                        phi_i = self._elements["illumination"].azimuth.to(ureg.rad).magnitude
                        config_measure["orientation"] = [np.cos(phi_i), np.sin(phi_i), 0]

                    if not config_measure.get("id", None):
                        config_measure["id"] = config_measure["type"]
                    config_measure["type"] = "radiancemeter_plane"

                elif config_measure["type"] in self._measure_aliases["toa_hsphere"]:
                    if not config_measure.get("id", None):
                        config_measure["id"] = config_measure["type"]
                    config_measure["type"] = "radiancemeter_hsphere"

                else:
                    raise ValueError(f"unsupported measure type '{config_measure['type']}'")

                config_measure["hemisphere"] = "back"
                # TODO: warn when overriding parameters set by user

                self._elements[config_measure["id"]] = \
                    MeasureFactory.create(config_measure)
                sensor_info = self._elements[config_measure["id"]].sensor_info()
                for sensor_id, sensor_spp in sensor_info:
                    self._measure_registry.insert(
                        {
                            "measure_id": config_measure["id"],
                            "sensor_id": sensor_id,
                            "sensor_spp": sensor_spp
                        }
                    )

            # Expand elements to kernel scene dictionary
            self._kernel_dict.add(list(self._elements.values()))

            # Add integrator
            self._kernel_dict.add({"integrator": {"type": "volpath"}})

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

        sensor_query = Query()
        measure_ids = set(measure["measure_id"] for measure in self._measure_registry)
        for measure_id in measure_ids:
            related_entries = self._measure_registry.search(sensor_query.measure_id == measure_id)
            sensor_ids = [db_entry["sensor_id"] for db_entry in related_entries]
            sensor_spp = [db_entry["sensor_spp"] for db_entry in related_entries]
            data = self._elements[measure_id].postprocess_results(sensor_ids,
                                                                  sensor_spp,
                                                                  runner_results)
            results = xr.Dataset()
            for dim in [0, 1, 4]:
                data = np.expand_dims(data, dim)

            zenith_res = self._elements[measure_id].zenith_res

            if isinstance(self._elements[measure_id], RadianceMeterHsphereMeasure):
                azimuth_res = self._elements[measure_id].azimuth_res
                vza = np.arange(0., 90., zenith_res.to(ureg.deg).magnitude)
                vaa = np.arange(0., 360., azimuth_res.to(ureg.deg).magnitude)
                coord_specs_id = "angular_observation"

            elif isinstance(self._elements[measure_id], RadianceMeterPlaneMeasure):
                vza = np.arange(0., 90., zenith_res.to(ureg.deg).magnitude)
                vaa = np.array([0., 180.])
                coord_specs_id = "angular_observation_pplane"
            else:
                raise ValueError(f"Unsupported measure type {type(self._elements[measure_id])}")

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

            self.results[measure_id] = results

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
