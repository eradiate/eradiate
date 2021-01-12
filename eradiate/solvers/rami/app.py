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
from ..onedim.runner import OneDimRunner
from ...scenes.biosphere import (BiosphereFactory, Canopy)
from ...scenes.core import KernelDict, SceneElement
from ...scenes.illumination import (
    DirectionalIllumination, Illumination, IlluminationFactory
)
from ...scenes.integrators import (
    Integrator, IntegratorFactory, PathIntegrator
)
from ...scenes.measure import (DistantMeasure, MeasureFactory, PerspectiveCameraMeasure)
from ...scenes.surface import LambertianSurface, Surface, SurfaceFactory
from ...util import plot as ertplt
from ...util import xarray as ertxr
from ...util.exceptions import ModeError
from ...util.frame import direction_to_angles, square_to_uniform_hemisphere
from ...util.misc import always_iterable, ensure_array
from ...util.units import kernel_default_units as kdu
from ...util.units import ureg


@attr.s
class RamiScene(SceneElement):
    """Scene abstraction suitable for radiative transfer simulation on
    RAMI benchmark scenes.

    .. rubric:: Constructor arguments / instance attributes

    ``atmosphere`` (:class:`.Atmosphere` or dict):
        Atmosphere specification.
        This parameter can be specified as a dictionary which will be
        interpreted by
        :meth:`AtmosphereFactory.convert() <.AtmosphereFactory.convert>`.
        Default: :class:`HomogeneousAtmosphere() <.HomogeneousAtmosphere>`.

    ``surface`` (:class:`.Surface` or dict):
        Surface specification.
        This parameter can be specified as a dictionary which will be
        interpreted by
        :meth:`SurfaceFactory.convert() <.SurfaceFactory.convert>`.
        Default: :class:`LambertianSurface() <.LambertianSurface>`.

    ``biosphere`` (:class:`.Biosphere` or dict):
        Biosphere specification.
        This parameter can be specified as a dictionary which will be
        interpreted by
        :meth:`BiosphereFactory.convert() <.BiosphereFactory.convert>`
        Default: :class:`HomogeneousDiscreteCanopy() <.HomogeneousDiscreteCanopy>`.

    ``illumination`` (:class:`.Illumination` or dict):
        Illumination specification.
        This parameter can be specified as a dictionary which will be
        interpreted by
        :meth:`IlluminationFactory.convert() <.IlluminationFactory.convert>`.
        Default: :class:`DirectionalIllumination() <.DirectionalIllumination>`.

    ``measures`` (list[:class:`.Measure`] or list[dict] or :class:`.Measure` or dict):
        List of measure specifications. The passed list may contain dictionary,
        which will be interpreted by
        :meth:`MeasureFactory.convert() <.MeasureFactory.convert>`.
        Optionally, a single :class:`.Measure` or dictionary specification
        may be passed and will automatically be wrapped into a list.
        Allowed value types: :class:`TOAHsphereMeasure`,
        :class:`TOAPPlaneMeasure`.
        Default: :class:`TOAHsphereMeasure() <TOAHsphereMeasure>`.

    ``integrator`` (:class:`.Integrator` or dict):
        Monte Carlo integration algorithm specification.
        This parameter can be specified as a dictionary which will be
        interpreted by
        :meth:`IntegratorFactory.convert() <.IntegratorFactory.convert>`.
        Default: :class:`VolPathIntegrator() <.VolPathIntegrator>`.
    """

    surface = attr.ib(
        factory=LambertianSurface,
        converter=SurfaceFactory.convert,
        validator=attr.validators.instance_of(Surface)
    )

    canopy = attr.ib(
        default=None,
        converter=attr.converters.optional(BiosphereFactory.convert),
        validator=attr.validators.optional(attr.validators.instance_of(Canopy))
    )

    illumination = attr.ib(
        factory=DirectionalIllumination,
        converter=IlluminationFactory.convert,
        validator=attr.validators.instance_of(Illumination)
    )

    measures = attr.ib(
        factory=lambda: [DistantMeasure()],
        converter=lambda value:
        [MeasureFactory.convert(x) for x in always_iterable(value)]
        if not isinstance(value, dict)
        else [MeasureFactory.convert(value)]
    )

    @measures.validator
    def _measures_validator(self, attribute, value):
        for element in value:
            # Check measure type
            if not isinstance(element,
                              (DistantMeasure, PerspectiveCameraMeasure)):
                raise TypeError(
                    f"while validating {attribute.name}: must be a list of "
                    f"objects of one of the following types: "
                    f"(DistantMeasure)"
                )

    integrator = attr.ib(
        factory=PathIntegrator,
        converter=IntegratorFactory.convert,
        validator=attr.validators.instance_of(Integrator)
    )

    measure_registry = attr.ib(
        init=False,
        repr=False,
        factory=lambda: TinyDB(storage=MemoryStorage)
    )

    def __attrs_post_init__(self):
        # Parts of the init sequence we could take care of using converters

        # Override surface width with canopy width
        if self.canopy is not None:
            self.surface.width = max(self.canopy.size[:2])

        # Process measures
        for measure in self.measures:
            # Override ray target location
            if isinstance(measure, DistantMeasure):
                if self.canopy is not None:
                    measure.target = dict(
                        type="rectangle",
                        xmin=-0.5 * self.canopy.size[0],
                        xmax=0.5 * self.canopy.size[0],
                        ymin=-0.5 * self.canopy.size[1],
                        ymax=0.5 * self.canopy.size[1],
                    )
                else:
                    measure.target = dict(
                        type="rectangle",
                        xmin=-0.5 * self.surface.width,
                        xmax=0.5 * self.surface.width,
                        ymin=-0.5 * self.surface.width,
                        ymax=0.5 * self.surface.width,
                    )

        # Populate measure registry
        for measure in self.measures:
            for sensor_id, sensor_spp in measure.sensor_info():
                self.measure_registry.insert({
                    "measure_id": measure.id,
                    "sensor_id": sensor_id,
                    "sensor_spp": sensor_spp
                })

    def kernel_dict(self, ref=True):
        result = KernelDict.empty()

        if self.canopy is not None:
            result.add(self.canopy)

        result.add([
            self.surface,
            self.illumination,
            *self.measures,
            self.integrator
        ])

        return result


@attr.s
class RamiSolverApp:
    """Solver application dedicated to the simulation of radiative transfer on
    RAMI benchmark scenes.

    .. rubric:: Constructor arguments / instance attributes

    ``scene`` (:class:`RamiScene`):
        RAMI benchmark scene to simulate radiative transfer on.
        This parameter can be specified as a dictionary which will be
        interpreted by
        :meth:`RamiScene.from_dict() <.RamiScene.from_dict>`.
        Default: :class:`RamiScene() <RamiScene>`.

    ``results`` (dict) [read-only]:
        Post-processed simulation results. Each entry uses a measure ID as its
        key and holds a value consisting of a :class:`~xarray.Dataset` holding
        one variable per physical quantity computed by the measure.
    """
    # Class attributes
    SUPPORTED_MODES = frozenset({"mono", "mono_double"})

    # Instance attributes
    scene = attr.ib(
        factory=RamiScene,
        converter=lambda x: RamiScene.from_dict(x)
        if isinstance(x, dict) else x,
        validator=attr.validators.instance_of(RamiScene)
    )

    results = attr.ib(factory=dict, init=False)

    _kernel_dict = attr.ib(default=None, init=False, repr=False)  # Cached kernel dictionary

    _runner = attr.ib(default=None, init=False, repr=False)  # Runner

    def __attrs_post_init__(self):
        # Initialise runner
        self._kernel_dict = self.scene.kernel_dict()
        self._kernel_dict.check()
        self._runner = OneDimRunner(self._kernel_dict)

    @classmethod
    def new(cls, *args, **kwargs):
        """Create a :class:`.RamiSolverApp` instance after preparatory checks.
        All arguments are forwarded to the :class:`.RamiSolverApp`
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
        scene = RamiScene(**solver_config)

        # Instantiate class
        return cls.new(scene=scene)

    def run(self):
        """Perform radiative transfer simulation and post-process results."""

        def _add_missing_dims(x, dims):
            """Local function; add missing dimensions to numpy array."""
            for dim in dims:
                x = np.expand_dims(x, dim)
            return x

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

        # -- TODO: Format results
        sensor_query = Query()
        for measure in scene.measures:
            # Collect results from sensors associated to processed measure
            measure_id = measure.id
            entries = scene.measure_registry.search(sensor_query.measure_id == measure_id)
            sensor_ids = [db_entry["sensor_id"] for db_entry in entries]
            sensor_spps = [db_entry["sensor_spp"] for db_entry in entries]
            data = measure.postprocess_results(sensor_ids, sensor_spps, runner_results)
            data = _add_missing_dims(data, [0, 1, 4])

            # Create an empty dataset to store results
            ds = xr.Dataset()

            if isinstance(measure, DistantMeasure):
                # Assign leaving radiance variable and corresponding coords
                ds["lo"] = xr.DataArray(
                    data,
                    coords=(
                        ("sza", sza),
                        ("saa", saa),
                        ("y", range(data.shape[2])),
                        ("x", range(data.shape[3])),
                        ("wavelength", wavelength)
                    ),
                )

                # Did the sensor sample ray directions in the hemisphere or on a
                # plane?
                plane = (data.shape[2] == 1)

                # Compute viewing angles at pixel centers
                xs = ds.x
                ys = ds.y
                theta = np.full((len(ys), len(xs)), np.nan)
                phi = np.full_like(theta, np.nan)

                if plane:
                    for x in xs:
                        for y in ys:
                            sample = float(x + 0.5) / len(xs)
                            theta[y, x] = 90. - 180. * sample
                            phi[y, x] = measure.orientation.to("deg").m
                else:
                    for x in xs:
                        for y in ys:
                            xy = [float((x + 0.5) / len(xs)),
                                  float((y + 0.5) / len(ys))]
                            d = square_to_uniform_hemisphere(xy)
                            theta[y, x], phi[y, x] = \
                                direction_to_angles(d).to("deg").m

                # Assign angles as non-dimension coords
                ds["vza"] = (("y", "x"), theta, {"units": "deg"})
                ds["vaa"] = (("y", "x"), phi, {"units": "deg"})
                ds = ds.set_coords(("vza", "vaa"))

            else:
                raise ValueError(f"Unsupported measure type {measure.__class__}")

            # Add other variables
            ds["irradiance"] = (
                ("sza", "saa", "wavelength"),
                np.array(
                    self._kernel_dict["illumination"]["irradiance"]["value"] *
                    cos_sza
                ).reshape((1, 1, 1))
            )
            ds["brdf"] = ds["lo"] / ds["irradiance"]
            ds["brf"] = ds["brdf"] * np.pi

            # Add missing metadata
            dataset_spec = ertxr.DatasetSpec(
                convention="CF-1.8",
                title="Top-of-atmosphere simulation results",
                history=f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                        f"data creation - {__name__}.{self.__class__}.run",
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
                coord_specs="angular_observation"
            )
            ds.ert.normalize_metadata(dataset_spec)

            self.results[measure_id] = ds

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
