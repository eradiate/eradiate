from __future__ import annotations

import datetime
import logging
import typing as t
from abc import ABC, abstractmethod

import attr
import mitsuba as mi
import pinttr
import xarray as xr
from tqdm.auto import tqdm

import eradiate

from .. import config, pipelines
from .._mode import ModeFlags, supported_mode
from ..attrs import documented, parse_docs
from ..contexts import KernelDictContext
from ..exceptions import KernelVariantError
from ..pipelines import Pipeline
from ..rng import SeedState, root_seed_state
from ..scenes.core import KernelDict
from ..scenes.illumination import (
    ConstantIllumination,
    DirectionalIllumination,
    illumination_factory,
)
from ..scenes.integrators import Integrator, PathIntegrator, integrator_factory
from ..scenes.measure import (
    DistantFluxMeasure,
    HemisphericalDistantMeasure,
    Measure,
    MultiDistantMeasure,
    measure_factory,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#                               Mitsuba runner
# ------------------------------------------------------------------------------

_SUPPORTED_VARIANTS = {"scalar_mono", "scalar_mono_double"}


def _check_variant():
    variant = mi.variant()
    if variant not in _SUPPORTED_VARIANTS:
        raise KernelVariantError(f"unsupported kernel variant '{variant}'")


def mitsuba_run(
    kernel_dict: KernelDict,
    sensor_ids: t.Optional[t.List[str]] = None,
    seed_state: t.Optional[SeedState] = None,
) -> t.Dict:
    """
    Run Mitsuba on a kernel dictionary.

    In practice, this function instantiates a kernel scene based on a
    dictionary and runs the integrator with all sensors. It returns results
    structured as nested dictionaries.

    Parameters
    ----------
    kernel_dict : :class:`.KernelDict`
        Dictionary describing the kernel scene.

    sensor_ids : list of str, optional
        Identifiers of the sensors for which the integrator is run. If set to
        ``None``, all sensors are selected.

    seed_state : .SeedState, optional
        A RNG seed state used generate the seeds used by Mitsuba's RNG
        generator. By default, Eradiate's :data:`.root_seed_state` is used.

    Returns
    -------
    dict
        Results stored as nested dictionaries.

    Notes
    -----
    The results are stored as dictionaries with the following structure:

    .. code:: python

       {
           "values": {
               "sensor_0": data_0, # mitsuba.core.Bitmap object
               "sensor_1": data_1, # mitsuba.core.Bitmap object
               ...
           },
           "spp": {
               "sensor_0": sample_count_0,
               "sensor_1": sample_count_1,
               ...
           },
       }

    The sample count is stored in a dedicated sub-dictionary in order to allow
    for sample-count-based aggregation.
    """
    _check_variant()
    if seed_state is None:
        seed_state = root_seed_state

    # Result storage
    results = {}

    # Run computation
    kernel_scene = kernel_dict.load()

    # Define the list of processed sensors
    if sensor_ids is None:
        sensors = kernel_scene.sensors()
    else:
        sensors = [
            sensor
            for sensor in kernel_scene.sensors()
            if str(sensor.id()) in sensor_ids
        ]

    # Run kernel for selected sensors
    for i_sensor, sensor in enumerate(sensors):
        # Run Mitsuba
        seed = seed_state.next()
        kernel_scene.integrator().render(kernel_scene, sensor, seed=seed)

        # Collect results (store a copy of the sensor's bitmap)
        film = sensor.film()
        result = mi.Bitmap(film.bitmap())

        sensor_id = str(sensor.id())
        # Raise if sensor doesn't have an ID (shouldn't happen since Mitsuba
        # should assign defaults based on scene dict keys)
        assert sensor_id

        if "values" not in results:
            results["values"] = {}
        results["values"][sensor_id] = result

        # Add sensor SPPs
        if "spp" not in results:
            results["spp"] = {}
        results["spp"][sensor_id] = sensor.sampler().sample_count()

    return results


# ------------------------------------------------------------------------------
#                                 Base classes
# ------------------------------------------------------------------------------


@parse_docs
@attr.s
class Experiment(ABC):
    """
    Base class for experiment simulations.
    """

    # --------------------------------------------------------------------------
    #                           Fields and properties
    # --------------------------------------------------------------------------

    measures: t.List[Measure] = documented(
        attr.ib(
            factory=lambda: [MultiDistantMeasure()],
            converter=lambda value: [
                measure_factory.convert(x) for x in pinttr.util.always_iterable(value)
            ]
            if not isinstance(value, dict)
            else [measure_factory.convert(value)],
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(Measure)
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
        attr.ib(
            factory=PathIntegrator,
            converter=integrator_factory.convert,
            validator=attr.validators.instance_of(Integrator),
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

    _results: t.Dict[str, xr.Dataset] = documented(
        attr.ib(factory=dict, init=False, repr=False),
        doc="Post-processed simulation results. Each entry uses a measure ID as "
        "its key and holds a value consisting of a :class:`~xarray.Dataset` "
        "holding one variable per physical quantity computed by the measure.",
        type="dict[str, dataset]",
    )

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

    # --------------------------------------------------------------------------
    #                          Additional constructors
    # --------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: t.Mapping) -> Experiment:
        """
        Instantiate from a dictionary. The default implementation raises an
        exception.

        Parameters
        ---------
        d : dict
            Dictionary to be converted to an :class:`.Experiment`.

        Returns
        -------
        :class:`.Experiment`
        """
        raise NotImplementedError

    # --------------------------------------------------------------------------
    #                              Processing
    # --------------------------------------------------------------------------

    def run(
        self,
        *measures: t.Union[Measure, int],
        seed_state: t.Optional[SeedState] = None,
    ) -> None:
        """
        Perform radiative transfer simulation and post-process results.
        Essentially chains :meth:`.process` and :meth:`.postprocess`.

        Parameters
        ----------
        *measures : :class:`.Measure` or int
            One or several measures for which to compute radiative transfer.
            Alternatively, indexes in the measure array can be passed.
            If no value is passed, all measures are processed.

        seed_state : .SeedState, optional
            A RNG seed state used generate the seeds used by Mitsuba's RNG
            generator. By default, Eradiate's :data:`.root_seed_state` is used.

        See Also
        --------
        :meth:`.process`, :meth:`.postprocess`
        """
        self.process(*measures, seed_state=seed_state)
        self.postprocess(*measures)

    def process(
        self,
        *measures: t.Union[Measure, int],
        seed_state: t.Optional[SeedState] = None,
    ) -> None:
        """
        Run simulation on the configured scene. Raw results yielded by the
        runner function are stored in ``measure.results``.

        Parameters
        ----------
        *measures : :class:`.Measure` or int
            One or several measures for which to compute radiative transfer.
            Alternatively, indexes in the measure array can be passed.
            If no value is passed, all measures are processed.

        seed_state : .SeedState, optional
            A RNG seed state used generate the seeds used by Mitsuba's RNG
            generator. By default, Eradiate's :data:`.root_seed_state` is used.

        See Also
        --------
        :meth:`.postprocess`, :meth:`.run`
        """

        # Mode safeguard
        supported_mode(ModeFlags.ANY_MONO | ModeFlags.ANY_CKD)

        if not measures:
            measures = self.measures

        for measure in measures:
            if isinstance(measure, int):
                measure = self.measures[measure]

            logger.info(f"Processing measure '{measure.id}'")

            # Reset measure results
            measure.results = {}

            # Collect sensor IDs
            sensor_ids = measure._sensor_ids()

            # Spectral loop
            spectral_ctxs = measure.spectral_cfg.spectral_ctxs()

            with tqdm(
                initial=0,
                total=len(spectral_ctxs),
                unit_scale=1.0,
                leave=True,
                bar_format="{desc}{n:g}/{total:g}|{bar}| {elapsed}, ETA={remaining}",
                disable=config.progress < 1 or len(spectral_ctxs) <= 1,
            ) as pbar:
                for kernel_dict, ctx in self.kernel_dicts(measure):
                    spectral_ctx = ctx.spectral_ctx

                    pbar.set_description(
                        f"Spectral loop [{spectral_ctx.spectral_index_formatted}]",
                        refresh=True,
                    )

                    # Run simulation
                    run_results = mitsuba_run(kernel_dict, sensor_ids, seed_state)

                    # Store results
                    measure.results[spectral_ctx.spectral_index] = run_results

                    # Update progress display
                    pbar.update()

    @abstractmethod
    def pipeline(
        self, *measures: t.Union[Measure, int]
    ) -> t.Union[Pipeline, t.Tuple[Pipeline, ...]]:
        """
        Request post-processing pipeline for a given measure.

        Parameters
        ----------
        *measures : :class:`.Measure` or int
            One or several measures for which to get a post-processing pipeline.
            If integer values are passed, they are used to query the measure
            list.

        Returns
        -------
        pipelines : :class:`.Pipeline` or tuple of :class:`.Pipeline`
            If a single measure is passed, a single :class:`.Pipeline` instance
            is returned; if multiple measures are passed, a tuple of pipelines
            is returned.
        """
        pass

    def postprocess(
        self,
        *measures: t.Union[Measure, int],
        pipeline_kwargs: t.Optional[t.Dict] = None,
    ) -> None:
        """
        Post-process raw results stored in a measure's ``results`` field. This
        requires a successful execution of :meth:`.process`. Post-processed
        results are stored in ``self.results``.

        Parameters
        ----------
        *measures : :class:`.Measure` or int
            One or several measures for which to perform post-processing.
            Alternatively, indexes in the measure array can be passed.
            If no value is passed, all measures are processed.

        pipeline_kwargs : dict, optional
            A dictionary of pipeline keyword arguments forwarded to
            :meth:`.Pipeline.transform`.

        Raises
        ------
        ValueError
            If ``measure.results`` is ``None``, *i.e.* if :meth:`.process`
            has not been successfully run.

        See Also
        --------
        :meth:`.process`, :meth:`.run`
        """
        if not measures:
            measures = self.measures

        if pipeline_kwargs is None:
            pipeline_kwargs = {}

        # Convert integer values to measure entries
        measures = [
            self.measures[measure] if isinstance(measure, int) else measure
            for measure in measures
        ]

        # Apply pipelines
        for measure in measures:
            pipeline = self.pipeline(measure)

            # Collect measure results
            self._results[measure.id] = pipeline.transform(
                measure.results, **pipeline_kwargs
            )

            # Apply additional metadata
            self._results[measure.id].attrs.update(self._dataset_metadata(measure))

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
            "history": f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
            f"data creation - {self.__class__.__name__}.postprocess()",
            "references": "",
        }

    @abstractmethod
    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        """
        Return a dictionary suitable for kernel scene configuration.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        :class:`.KernelDict`
            Kernel dictionary which can be loaded as a Mitsuba object.
        """
        pass

    def kernel_dicts(
        self,
        measure: t.Union[Measure, int],
    ) -> t.Generator[t.Tuple[KernelDict, KernelDictContext], None, None]:
        """
        A generator which returns kernel dictionaries (and the associated
        context) relevant to a given measure.

        Parameters
        ----------
        measure : .Measure or int
            Measure for which kernel dictionaries are to be generated.
            Alternatively, the index in the ``self.measure`` list can be passed.

        Yields
        ------
        kernel_dict : .KernelDict
            Generated kernel dictionary.

        ctx : .KernelDictContext
            Context used to generate ``kernel_dict``.
        """
        if isinstance(measure, int):
            measure = self.measures[measure]

        spectral_ctxs = measure.spectral_cfg.spectral_ctxs()

        for spectral_ctx in spectral_ctxs:
            ctx = KernelDictContext(spectral_ctx=spectral_ctx, ref=True)
            yield self.kernel_dict(ctx=ctx), ctx


@parse_docs
@attr.s
class EarthObservationExperiment(Experiment, ABC):
    """
    A base class used for Earth observation simulations. These experiments
    all feature illumination from a distant source such as the Sun.
    """

    illumination: t.Union[DirectionalIllumination, ConstantIllumination] = documented(
        attr.ib(
            factory=DirectionalIllumination,
            converter=illumination_factory.convert,
            validator=attr.validators.instance_of(
                (DirectionalIllumination, ConstantIllumination)
            ),
        ),
        doc="Illumination specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.illumination_factory`.",
        type=":class:`.DirectionalIllumination` or :class:`.ConstantIllumination`",
        init_type=":class:`.DirectionalIllumination` or "
        ":class:`.ConstantIllumination` or dict",
        default=":class:`DirectionalIllumination() <.DirectionalIllumination>`",
    )

    def pipeline(
        self, *measures: t.Union[Measure, int]
    ) -> t.Union[Pipeline, t.Tuple[Pipeline, ...]]:
        result = []

        # Convert integer values to measure entries
        measures = [
            self.measures[measure] if isinstance(measure, int) else measure
            for measure in measures
        ]

        for measure in measures:
            pipeline = pipelines.Pipeline()

            # Gather
            pipeline.add(
                "gather",
                pipelines.Gather(sensor_dims=measure.sensor_dims, var=measure.var),
            )

            # Aggregate
            pipeline.add("aggregate_sample_count", pipelines.AggregateSampleCount())
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

            if measure.is_distant():
                pipeline.add(
                    "add_viewing_angles", pipelines.AddViewingAngles(measure=measure)
                )

            pipeline.add(
                "add_srf", pipelines.AddSpectralResponseFunction(measure=measure)
            )

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

                if eradiate.mode().has_flags(ModeFlags.ANY_CKD):
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

                if eradiate.mode().has_flags(ModeFlags.ANY_CKD):
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

            result.append(pipeline)

        return result[0] if len(result) == 1 else tuple(result)
