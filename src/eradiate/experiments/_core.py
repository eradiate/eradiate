import logging
import typing as t
from abc import ABC, abstractmethod
from datetime import datetime

import attrs
import mitsuba as mi
import pinttr
import xarray as xr
from tqdm.auto import tqdm

import eradiate

from .. import pipelines
from .._config import ProgressLevel, config
from ..attrs import documented
from ..contexts import KernelDictContext
from ..kernel._kernel_dict import ParameterMap
from ..pipelines import Pipeline
from ..rng import SeedState, root_seed_state
from ..scenes.core import Scene, SceneElement, get_factory, traverse
from ..scenes.illumination import (
    ConstantIllumination,
    DirectionalIllumination,
    illumination_factory,
)
from ..scenes.integrators import Integrator, PathIntegrator, integrator_factory
from ..scenes.measure import (
    DistantFluxMeasure,
    DistantMeasure,
    HemisphericalDistantMeasure,
    Measure,
    MultiDistantMeasure,
    measure_factory,
)
from ..util.misc import onedict_value

logger = logging.getLogger(__name__)


def mi_render(
    mi_scene: "mitsuba.Scene",
    params: ParameterMap,
    ctxs: t.List[KernelDictContext],
    mi_params: t.Optional["mitsuba.SceneParameters"] = None,
    sensors: t.Union[None, int, t.List[int]] = 0,
    spp: int = 0,
    seed_state: t.Optional[SeedState] = None,
) -> t.Dict[t.Any, "mitsuba.Bitmap"]:
    """
    Render a Mitsuba scene multiple times given specified contexts and sensor
    indices.

    Parameters
    ----------
    mi_scene : :class:`mitsuba.Scene`
        Mitsuba scene to render.

    params : ParameterMap
        Parameter map used to generate the Mitsuba parameter table at each
        iteration.

    ctxs : list of :class:`.KernelDictContext`
        List of contexts used to generate the parameter update table at each
        iteration.

    mi_params : :class:`mitsuba.SceneParameters`, optional
        Mitsuba parameter table for the rendered scene. If unset, a new
        parameter table is created.

    sensors : int or list of int, optional
        Sensor indices to render. If ``None``, all sensors are rendered.

    spp : int, optional, default: 0
        Number of samples per pixel. If set to 0, the value set in the original
        scene definition takes precedence.

    seed_state : :class:`.SeedState, optional
        Seed state used to generate seeds to initialise Mitsuba's RNG at
        each run. If unset, Eradiate's root seed state is used.

    Returns
    -------
    dict
        A nested dictionary mapping context and sensor indices to rendered
        bitmaps.
    """
    if mi_params is None:
        mi_params = mi.traverse(mi_scene)

    if seed_state is None:
        seed_state = root_seed_state

    results = {}

    # Loop on contexts
    with tqdm(
        initial=0,
        total=len(ctxs),
        unit_scale=1.0,
        leave=True,
        bar_format="{desc}{n:g}/{total:g}|{bar}| {elapsed}, ETA={remaining}",
        disable=(config.progress < ProgressLevel.SPECTRAL_LOOP) or len(ctxs) <= 1,
    ) as pbar:
        for ctx in ctxs:
            pbar.set_description(
                f"Eradiate [{ctx.index_formatted}]",
                refresh=True,
            )

            mi_params.update(params.render(ctx))

            if sensors is None:
                mi_sensors = [
                    (i, sensor) for i, sensor in enumerate(mi_scene.sensors())
                ]

            else:
                if isinstance(sensors, int):
                    sensors = [sensors]
                mi_sensors = [(i, mi_scene.sensors()[i]) for i in sensors]

            # Loop on sensors
            for i_sensor, mi_sensor in mi_sensors:
                # Render sensor
                mi.render(
                    mi_scene,
                    sensor=i_sensor,
                    seed=int(seed_state.next()),
                    spp=spp,
                )

                # Store result in a new Bitmap object
                if ctx.spectral_ctx.spectral_index not in results:
                    results[ctx.spectral_ctx.spectral_index] = {}

                results[ctx.spectral_ctx.spectral_index][mi_sensor.id()] = mi.Bitmap(
                    mi_sensor.film().bitmap()
                )

            pbar.update()

    return results


@attrs.define
class Experiment(ABC):
    mi_scene: t.Optional["mitsuba.Scene"] = attrs.field(
        default=None,
        repr=False,
    )

    mi_params: t.Optional["mitsuba.SceneParameters"] = attrs.field(
        default=None,
        repr=False,
    )

    params: t.Optional[ParameterMap] = attrs.field(
        default=None,
        repr=False,
    )

    measures: t.List[Measure] = documented(
        attrs.field(
            factory=lambda: [MultiDistantMeasure()],
            converter=lambda value: [
                measure_factory.convert(x) for x in pinttr.util.always_iterable(value)
            ]
            if not isinstance(value, dict)
            else [measure_factory.convert(value)],
            validator=attrs.validators.deep_iterable(
                member_validator=attrs.validators.instance_of(Measure)
            ),
        ),
        doc="List of measure specifications. The passed list may contain "
        "dictionaries, which will be interpreted by "
        ":data:`.measure_factory`. "
        "Optionally, a single :class:`.Measure` or dictionary specification "
        "may be passed and will automatically be wrapped into a list.",
        type="list of :class:`.Measure`",
        init_type="list of :class:`.Measure` or list of dict or "
        ":class:`.Measure` or dict",
        default=":class:`MultiDistantMeasure() <.MultiDistantMeasure>`",
    )

    _integrator: Integrator = documented(
        attrs.field(
            factory=PathIntegrator,
            converter=integrator_factory.convert,
            validator=attrs.validators.instance_of(Integrator),
        ),
        doc="Monte Carlo integration algorithm specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.integrator_factory`.",
        type=":class:`.Integrator`",
        init_type=":class:`.Integrator` or dict",
        default=":class:`PathIntegrator() <.PathIntegrator>`",
    )

    @property
    def integrator(self) -> Integrator:
        """
        :class:`.Integrator`: Integrator used to solve the radiative transfer equation.
        """
        return self._integrator

    _results: t.Dict[str, xr.Dataset] = attrs.field(factory=dict, repr=False)

    @property
    def results(self) -> t.Dict[str, xr.Dataset]:
        """
        Post-processed simulation results.

        Returns
        -------
        dict[str, Dataset]
            Dictionary mapping measure IDs to xarray datasets.
        """
        return self._results

    def clear(self) -> None:
        """
        Clear previous experiment results and reset internal state.
        """
        self.mi_params = None
        self.params = None
        self.results.clear()

        for measure in self.measures:
            measure.mi_results.clear()

    @abstractmethod
    def init(self) -> None:
        """
        Generate kernel dictionary and initialise Mitsuba scene.
        """
        pass

    @abstractmethod
    def process(
        self,
        spp: int = 0,
        seed_state: t.Optional[SeedState] = None,
    ) -> None:
        """
        Run simulation and collect raw results.

        Parameters
        ----------
        spp : int, optional
            Sample count. If set to 0, the value set in the original scene
            definition takes precedence.

        seed_state : :class:`.SeedState`, optional
            Seed state used to generate seeds to initialise Mitsuba's RNG at
            every iteration of the parametric loop. If unset, Eradiate's
            :attr:`root seed state <.root_seed_state>` is used.
        """
        pass

    @abstractmethod
    def postprocess(self) -> None:
        """
        Post-process raw results and store them in :attr:`results`.
        """
        pass

    @abstractmethod
    def pipeline(self, measure: Measure) -> Pipeline:
        """
        Return the post-processing pipeline for a given measure.

        Parameters
        ----------
        measure : .Measure
            Measure for which the pipeline is to be generated.

        Returns
        -------
        .Pipeline
        """
        pass

    @property
    @abstractmethod
    def context_init(self) -> KernelDictContext:
        """
        Return a single context used for scene initialisation.
        """
        pass

    @property
    @abstractmethod
    def contexts(self) -> t.List[KernelDictContext]:
        """
        Return a list of contexts used for processing.
        """
        pass


def _extra_objects_converter(value):
    result = {}

    for key, element_spec in value.items():
        if isinstance(element_spec, dict):
            element_spec = element_spec.copy()
            element_type = element_spec.pop("factory")
            factory = get_factory(element_type)
            result[key] = factory.convert(element_spec)

        else:
            result[key] = element_spec

    return result


@attrs.define
class EarthObservationExperiment(Experiment, ABC):
    extra_objects: t.Dict[str, SceneElement] = documented(
        attrs.field(
            factory=dict,
            converter=_extra_objects_converter,
            validator=attrs.validators.deep_mapping(
                key_validator=attrs.validators.instance_of(str),
                value_validator=attrs.validators.instance_of(SceneElement),
            ),
        ),
        doc="Dictionary of extra objects to be added to the scene. "
        "The keys of this dictionary are used to identify the objects "
        "in the kernel dictionary.",
        type="dict",
        default="{}",
    )

    illumination: t.Union[DirectionalIllumination, ConstantIllumination] = documented(
        attrs.field(
            factory=DirectionalIllumination,
            converter=illumination_factory.convert,
            validator=attrs.validators.instance_of(
                (DirectionalIllumination, ConstantIllumination)
            ),
        ),
        doc="Illumination specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.illumination_factory`.",
        type=":class:`.DirectionalIllumination`",
        init_type=":class:`.DirectionalIllumination` or dict",
        default=":class:`DirectionalIllumination() <.DirectionalIllumination>`",
    )

    def _dataset_metadata(self, measure: Measure) -> t.Dict[str, str]:
        """
        Generate additional metadata applied to dataset after post-processing.

        Parameters
        ----------
        measure : :class:`.Measure`
            Measure for which the metadata is created.

        Returns
        -------
        dict[str, str]
            Metadata to be attached to the produced dataset.
        """

        return {
            "convention": "CF-1.8",
            "source": f"eradiate, version {eradiate.__version__}",
            "history": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
            f"data creation - {self.__class__.__name__}.postprocess()",
            "references": "",
        }

    @property
    @abstractmethod
    def scene_objects(self) -> t.Dict[str, SceneElement]:
        pass

    @property
    def scene(self) -> Scene:
        """
        Return a scene object used for kernel dictionary template and parameter
        table generation.
        """
        return Scene(objects={**self.scene_objects, **self.extra_objects})

    def init(self):
        # Inherit docstring

        logger.info("Initializing kernel scene")

        template, params = traverse(self.scene)
        kernel_dict = template.render(ctx=self.context_init, drop=True)

        try:
            self.mi_scene = mi.load_dict(kernel_dict)
        except RuntimeError as e:
            raise RuntimeError(f"(while loading kernel scene dictionary){e}") from e

        self.params = params

    def process(
        self,
        spp: int = 0,
        seed_state: t.Optional[SeedState] = None,
    ) -> None:
        # Inherit docstring

        # Set up Mitsuba scene
        if self.mi_scene is None:
            self.init()

        # Run Mitsuba for each context
        logger.info("Running simulation")

        mi_results = mi_render(
            self.mi_scene,
            self.params,
            self.contexts,
            mi_params=self.mi_params,
            seed_state=seed_state,
            spp=spp,
        )

        # Assign collected results to the appropriate measure
        sensor_to_measure: t.Dict[str, Measure] = {
            measure.sensor_id: measure for measure in self.measures
        }

        for ctx_index, spectral_group_dict in mi_results.items():
            for sensor_id, mi_bitmap in spectral_group_dict.items():
                measure = sensor_to_measure[sensor_id]
                measure.mi_results[ctx_index] = {
                    "bitmap": mi_bitmap,
                    "spp": spp if spp > 0 else measure.spp,
                }

    def postprocess(self, pipeline_kwargs: t.Optional[t.Dict] = None) -> None:
        # Inherit docstring
        logger.info("Post-processing results")
        measures = self.measures

        if pipeline_kwargs is None:
            pipeline_kwargs = {}

        # Apply pipelines
        for measure in measures:
            pipeline = self.pipeline(measure)

            # Collect measure results
            self._results[measure.id] = pipeline.transform(
                measure.mi_results, **pipeline_kwargs
            )

            # Apply additional metadata
            self._results[measure.id].attrs.update(self._dataset_metadata(measure))

    def pipeline(self, measure: Measure) -> Pipeline:
        pipeline = pipelines.Pipeline()

        # Gather
        pipeline.add(
            "gather",
            pipelines.Gather(var=measure.var),
        )

        # Aggregate
        pipeline.add(
            "aggregate_ckd_quad",
            pipelines.AggregateCKDQuad(measure=measure, var=measure.var[0]),
        )

        if isinstance(measure, (DistantFluxMeasure,)):
            pipeline.add(
                "aggregate_radiosity",
                pipelines.AggregateRadiosity(
                    sector_radiosity_var=measure.var[0],
                    radiosity_var="radiosity",
                ),
            )

        # Assemble
        pipeline.add(
            "add_illumination",
            pipelines.AddIllumination(
                illumination=self.illumination,
                measure=measure,
                irradiance_var="irradiance",
            ),
        )

        if isinstance(measure, DistantMeasure):
            pipeline.add(
                "add_viewing_angles", pipelines.AddViewingAngles(measure=measure)
            )

        pipeline.add("add_srf", pipelines.AddSpectralResponseFunction(measure=measure))

        # Compute
        if isinstance(measure, (MultiDistantMeasure, HemisphericalDistantMeasure)):
            pipeline.add(
                "compute_reflectance",
                pipelines.ComputeReflectance(
                    radiance_var="radiance",
                    irradiance_var="irradiance",
                    brdf_var="brdf",
                    brf_var="brf",
                ),
            )

            if eradiate.mode().is_ckd:
                pipeline.add(
                    "apply_srf",
                    pipelines.ApplySpectralResponseFunction(
                        measure=measure,
                        vars=["radiance", "irradiance"],
                    ),
                )

                pipeline.add(
                    "compute_reflectance_srf",
                    pipelines.ComputeReflectance(
                        radiance_var="radiance_srf",
                        irradiance_var="irradiance_srf",
                        brdf_var="brdf_srf",
                        brf_var="brf_srf",
                    ),
                )

        elif isinstance(measure, (DistantFluxMeasure,)):
            pipeline.add(
                "compute_albedo",
                pipelines.ComputeAlbedo(
                    radiosity_var="radiosity",
                    irradiance_var="irradiance",
                    albedo_var="albedo",
                ),
            )

            if eradiate.mode().is_ckd:
                pipeline.add(
                    "apply_srf",
                    pipelines.ApplySpectralResponseFunction(
                        measure=measure,
                        vars=["radiosity", "irradiance"],
                    ),
                )

                pipeline.add(
                    "compute_albedo_srf",
                    pipelines.ComputeAlbedo(
                        radiosity_var="radiosity_srf",
                        irradiance_var="irradiance_srf",
                        albedo_var="albedo_srf",
                    ),
                )

        return pipeline


# ------------------------------------------------------------------------------
#                              Experiment runner
# ------------------------------------------------------------------------------


def run(
    exp: Experiment,
    spp: int = 0,
    seed_state: t.Optional[SeedState] = None,
) -> t.Tuple[xr.Dataset]:
    exp.process(spp=spp, seed_state=seed_state)
    exp.postprocess()
    return exp.results if len(exp.results) > 1 else onedict_value(exp.results)
