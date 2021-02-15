"""One dimensional solver application class and related facilities."""
import datetime
import os
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
from ...util.attrs import documented, parse_docs
from ...util.exceptions import ModeError
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


@parse_docs
@attr.s
class OneDimScene(SceneElement):
    """
    Scene abstraction suitable for radiative transfer simulation on
    one-dimensional scenes.
    """
    atmosphere = documented(
        attr.ib(
            factory=HomogeneousAtmosphere,
            converter=AtmosphereFactory.convert,
            validator=attr.validators.instance_of(Atmosphere),
        ),
        doc="Atmosphere specification. "
            "This parameter can be specified as a dictionary which will be "
            "interpreted by "
            ":meth:`AtmosphereFactory.convert() <.AtmosphereFactory.convert>`.",
        type=":class:`.Atmosphere` or dict",
        default=":class:`HomogeneousAtmosphere() <.HomogeneousAtmosphere>`",
    )

    surface = documented(
        attr.ib(
            factory=LambertianSurface,
            converter=SurfaceFactory.convert,
            validator=attr.validators.instance_of(Surface)
        ),
        doc="Surface specification. "
            "This parameter can be specified as a dictionary which will be "
            "interpreted by "
            ":meth:`SurfaceFactory.convert() <.SurfaceFactory.convert>`.",
        type=":class:`.Surface` or dict",
        default=":class:`LambertianSurface() <.LambertianSurface>`"
    )

    illumination = documented(
        attr.ib(
            factory=DirectionalIllumination,
            converter=IlluminationFactory.convert,
            validator=attr.validators.instance_of(Illumination)
        ),
        doc="Illumination specification. "
            "This parameter can be specified as a dictionary which will be "
            "interpreted by "
            ":meth:`IlluminationFactory.convert() <.IlluminationFactory.convert>`.",
        type=":class:`.Illumination` or dict",
        default=":class:`DirectionalIllumination() <.DirectionalIllumination>`"
    )

    measures = documented(
        attr.ib(
            factory=lambda: [TOAHsphereMeasure()],
            converter=lambda value:
            [MeasureFactory.convert(x) for x in always_iterable(value)]
            if not isinstance(value, dict)
            else [MeasureFactory.convert(value)]
        ),
        doc="List of measure specifications. The passed list may contain dictionary, "
            "which will be interpreted by "
            ":meth:`MeasureFactory.convert() <.MeasureFactory.convert>`. "
            "Optionally, a single :class:`.Measure` or dictionary specification "
            "may be passed and will automatically be wrapped into a list. "
            "Allowed value types: :class:`TOAHsphereMeasure`, "
            ":class:`TOAPPlaneMeasure`.",
        type="list[:class:`.Measure`] or list[dict] or :class:`.Measure` or dict",
        default=":class:`TOAHsphereMeasure() <TOAHsphereMeasure>`",
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

    integrator = documented(
        attr.ib(
            factory=VolPathIntegrator,
            converter=IntegratorFactory.convert,
            validator=attr.validators.instance_of(Integrator)
        ),
        doc="Monte Carlo integration algorithm specification. "
            "This parameter can be specified as a dictionary which will be "
            "interpreted by "
            ":meth:`IntegratorFactory.convert() <.IntegratorFactory.convert>`.",
        type=":class:`.Integrator` or dict",
        default=":class:`VolPathIntegrator() <.VolPathIntegrator>`"
    )

    measure_registry = attr.ib(
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
                    sensor_altitude = self.atmosphere.kernel_height.to(cdu.get("length")).magnitude
                    measure.origin = [0., 0., sensor_altitude]

            if isinstance(measure, TOAPPlaneMeasure):
                # Set principal plane measure orientation if any
                phi_i = self.illumination.azimuth.to(ureg.rad).magnitude
                measure.orientation = [np.cos(phi_i), np.sin(phi_i), 0.]

        # Populate measure registry
        for measure in self.measures:
            for sensor_id, sensor_spp in measure.sensor_info():
                self.measure_registry.insert(
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


@parse_docs
@attr.s
class OneDimSolverApp:
    """
    Solver application dedicated to the simulation of radiative transfer on
    one-dimensional scenes.
    """
    # Class attributes
    SUPPORTED_MODES = frozenset({"mono", "mono_double"})

    # Instance attributes
    scene = documented(
        attr.ib(
            factory=OneDimScene,
            converter=lambda x: OneDimScene.from_dict(x)
            if isinstance(x, dict) else x,
            validator=attr.validators.instance_of(OneDimScene)
        ),
        doc="One-dimensional scene to simulate radiative transfer on. "
            "This parameter can be specified as a dictionary which will be "
            "interpreted by "
            ":meth:`OneDimScene.from_dict() <.OneDimScene.from_dict>`.",
        type=":class:`OneDimScene`",
        default=":class:`OneDimScene() <OneDimScene>`",
    )

    results = documented(
        attr.ib(factory=dict, init=False),
        doc="Post-processed simulation results. Each entry uses a measure ID as its "
            "key and holds a value consisting of a :class:`~xarray.Dataset` holding "
            "one variable per physical quantity computed by the measure.\n"
            "\n"
            ".. note:: This field is read-only and is not accessible as a "
            "constructor argument.",
        type="dict",
    )

    _kernel_dict = attr.ib(default=None, init=False, repr=False)  # Cached kernel dictionary

    _runner = attr.ib(default=None, init=False, repr=False)  # Runner

    def __attrs_post_init__(self):
        # Initialise runner
        self._kernel_dict = self.scene.kernel_dict()
        self._kernel_dict.check()
        self._runner = OneDimRunner(self._kernel_dict)

    @classmethod
    def new(cls, *args, **kwargs):
        """Create a :class:`.OneDimSolverApp` instance after preparatory checks.
        All arguments are forwarded to the :class:`.OneDimSolverApp`
        constructor.
        """
        if isinstance(eradiate.mode, eradiate.ModeNone):
            raise ModeError(f"no mode selected, use eradiate.set_mode()")

        return cls(*args, **kwargs)

    @classmethod
    def from_dict(cls, d):
        """Instantiate from a dictionary."""
        # Collect mode configuration
        solver_config = deepcopy(d)

        try:
            mode_config = solver_config.pop("mode")
        except KeyError:
            raise ValueError("section 'mode' missing from configuration "
                             "dictionary")

        try:
            mode_id = mode_config.pop("type")
        except KeyError or AttributeError:
            raise ValueError("parameter 'mode.type' missing from configuration "
                             "dictionary")

        # Select appropriate operational mode
        if mode_id not in cls.SUPPORTED_MODES:
            raise ValueError(f"unsupported mode type '{mode_id}' (must be one "
                             f"of {tuple(cls.SUPPORTED_MODES)})")

        eradiate.set_mode(mode_id, **mode_config)

        # Create scene
        scene = OneDimScene(**solver_config)

        # Instantiate class
        return cls.new(scene=scene)

    def run(self):
        """Perform radiative transfer simulation and post-process results.
        """

        # Run simulation
        runner_results = self._runner.run()

        # Post-processing
        # TODO: put that in a separate method
        scene = self.scene
        # -- Ensure that scalar values used as xarray coordinates are arrays
        illumination = scene.illumination

        # -- Collect illumination parameters
        sza = ensure_array(illumination.zenith.to(ureg.deg).magnitude, dtype=float)
        cos_sza = np.cos(illumination.zenith.to(ureg.rad).magnitude)
        saa = ensure_array(illumination.azimuth.to(ureg.deg).magnitude, dtype=float)
        wavelength = ensure_array(eradiate.mode.wavelength.magnitude, dtype=float)
        # TODO: This will raise if illumination is not directional; handle that

        sensor_query = Query()
        for measure in scene.measures:
            measure_id = measure.id
            entries = scene.measure_registry.search(sensor_query.measure_id == measure_id)
            sensor_ids = [db_entry["sensor_id"] for db_entry in entries]
            sensor_spps = [db_entry["sensor_spp"] for db_entry in entries]
            data = measure.postprocess_results(sensor_ids, sensor_spps, runner_results)
            results = xr.Dataset()
            for dim in [0, 1, 4]:
                data = np.expand_dims(data, dim)

            zenith_res = measure.zenith_res

            if isinstance(measure, RadianceMeterHsphereMeasure):
                azimuth_res = measure.azimuth_res
                vza = np.arange(0., 90., zenith_res.to(ureg.deg).magnitude)
                vaa = np.arange(0., 360., azimuth_res.to(ureg.deg).magnitude)
                coord_specs_id = "angular_observation"

            elif isinstance(measure, RadianceMeterPlaneMeasure):
                vza = np.arange(0., 90., zenith_res.to(ureg.deg).magnitude)
                vaa = np.array([0., 180.])
                coord_specs_id = "angular_observation_pplane"

            else:
                raise ValueError(f"Unsupported measure type {measure.__class__}")

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
                        f"data creation - {__name__}.OneDimSolverApp.run",
                source=f"eradiate, version {eradiate.__version__}",
                references="",
                var_specs={
                    "irradiance": ertxr.VarSpec(
                        standard_name="toa_horizontal_solar_irradiance_per_unit_wavelength",
                        units=str(kdu.get("irradiance")),
                        long_name="top-of-atmosphere horizontal spectral irradiance"
                    ),
                    "lo": ertxr.VarSpec(
                        standard_name="toa_outgoing_radiance_per_unit_wavelength",
                        units=str(kdu.get("radiance")),
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
