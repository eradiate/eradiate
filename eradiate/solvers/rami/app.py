import os
from copy import deepcopy
from pathlib import Path

import attr
import matplotlib.pyplot as plt
import numpy as np
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
from ...util.misc import always_iterable, ensure_array
from ...util.units import ureg


@MeasureFactory.register(
    "toa_pplane_rami", "toa_pplane_lo_rami", "toa_pplane_brdf_rami", "toa_pplane_brf_rami"
)
@attr.s
class TOAPPlaneMeasure(DistantMeasure):
    """Top-of-atmosphere radiancemeter (principal plane coverage).
    This class is a lightweight specialisation of its
    :class:`radiancemeter` parent class and should only be used
    with :class:`RamiScene`.
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

        # Run simulation
        runner_results = self._runner.run()
        # print(runner_results)

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
            measure_id = measure.id
            entries = scene.measure_registry.search(sensor_query.measure_id == measure_id)
            sensor_ids = [db_entry["sensor_id"] for db_entry in entries]
            sensor_spps = [db_entry["sensor_spp"] for db_entry in entries]
            data = measure.postprocess_results(sensor_ids, sensor_spps, runner_results)

            results = data.squeeze()
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
