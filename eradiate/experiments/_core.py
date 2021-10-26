from __future__ import annotations

import datetime
import logging
import typing as t
from abc import ABC, abstractmethod

import attr
import mitsuba
import numpy as np
import pinttr
import xarray as xr
from tqdm import tqdm

import eradiate

from .. import config
from .._mode import ModeFlags, supported_mode
from ..attrs import documented, parse_docs
from ..contexts import KernelDictContext
from ..exceptions import KernelVariantError, UnsupportedModeError
from ..kernel import bitmap_to_dataset
from ..scenes.core import KernelDict
from ..scenes.illumination import (
    ConstantIllumination,
    DirectionalIllumination,
    illumination_factory,
)
from ..scenes.integrators import Integrator, PathIntegrator, integrator_factory
from ..scenes.measure import (
    DistantAlbedoMeasure,
    DistantMeasure,
    DistantRadianceMeasure,
    DistantReflectanceMeasure,
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
    variant = mitsuba.variant()
    if variant not in _SUPPORTED_VARIANTS:
        raise KernelVariantError(f"unsupported kernel variant '{variant}'")


def mitsuba_run(
    kernel_dict: KernelDict, sensor_ids: t.Optional[t.List[str]] = None
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
               "sensor_0": data_0, # xarray.Dataset with 'img' variable of shape
               "sensor_1": data_1, # (width, height, n_channels)
               ...                 # (n_channels == 1 for mono variants, 3 for RGB)
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

    # Result storage
    results = dict()

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
        kernel_scene.integrator().render(kernel_scene, sensor)

        # Collect results
        film = sensor.film()
        result = bitmap_to_dataset(film.bitmap(), dtype=float)

        sensor_id = str(sensor.id())
        if not sensor_id:  # Assign default ID if sensor doesn't have one
            sensor_id = f"__sensor_{i_sensor}"

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

    measures: t.List[Measure] = documented(
        attr.ib(
            factory=lambda: [DistantRadianceMeasure()],
            converter=lambda value: [
                measure_factory.convert(x) for x in pinttr.util.always_iterable(value)
            ]
            if not isinstance(value, dict)
            else [measure_factory.convert(value)],
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(
                    (DistantMeasure, MultiDistantMeasure)
                )
            ),
        ),
        doc="List of measure specifications. The passed list may contain "
        "dictionaries, which will be interpreted by "
        ":data:`.measure_factory`. "
        "Optionally, a single :class:`.Measure` or dictionary specification "
        "may be passed and will automatically be wrapped into a list.",
        type="list of :class:`.Measure`",
        init_type="list of :class:`.Measure` or list of dict or :class:`.Measure` or "
        "dict",
        default=":class:`DistantRadianceMeasure() <.DistantRadianceMeasure>`",
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

    _results: t.Dict[str, xr.Dataset] = documented(
        attr.ib(factory=dict, init=False, repr=False),
        doc="Post-processed simulation results. Each entry uses a measure ID as "
        "its key and holds a value consisting of a :class:`~xarray.Dataset` "
        "holding one variable per physical quantity computed by the measure.",
        type="dict[str, dataset]",
    )

    @property
    def integrator(self) -> Integrator:
        """
        :class:`.Integrator`: Integrator used to solve the radiative transfer equation.
        """
        return self._integrator

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

    def run(self, *measures: t.Union[Measure, int]) -> None:
        """
        Perform radiative transfer simulation and post-process results.
        Essentially chains :meth:`.preprocess`, :meth:`.process` and
        :meth:`.postprocess`.

        Parameters
        ----------
        *measures : :class:`.Measure` or int
            One or several measures for which to compute radiative transfer.
            Alternatively, indexes in the measure array can be passed.
            If no value is passed, all measures are processed.

        See Also
        --------
        :meth:`.preprocess`, :meth:`.process`, :meth:`.postprocess`

        """
        self.preprocess(*measures)
        self.process(*measures)
        self.postprocess(*measures)

    def preprocess(self, *measures: t.Union[Measure, int]) -> None:
        """
        Pre-process internal state if relevant.

        Parameters
        ----------
        *measures : :class:`.Measure` or int
            One or several measures for which to compute radiative transfer.
            Alternatively, indexes in the measure array can be passed.
            If no value is passed, all measures are processed.
        """
        pass

    def process(self, *measures: t.Union[Measure, int]) -> None:
        """
        Run simulation on the configured scene. Raw results yielded by the
        runner function are stored in ``measure.results``.

        Parameters
        ----------
        *measures : :class:`.Measure` or int
            One or several measures for which to compute radiative transfer.
            Alternatively, indexes in the measure array can be passed.
            If no value is passed, all measures are processed.

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

            # Spectral loop
            spectral_ctxs = measure.spectral_cfg.spectral_ctxs()

            with tqdm(
                initial=0,
                total=len(spectral_ctxs),
                unit_scale=1.0,
                leave=True,
                bar_format="{desc}{n:g}/{total:g}|{bar}| {elapsed}, ETA={remaining}",
                disable=config.progress < 1,
            ) as pbar:
                for spectral_ctx in spectral_ctxs:
                    pbar.set_description(
                        f"Spectral loop [{spectral_ctx.spectral_index_formatted}]",
                        refresh=True,
                    )

                    # Initialise context
                    ctx = KernelDictContext(spectral_ctx=spectral_ctx, ref=True)

                    # Collect sensor IDs
                    sensor_ids = [
                        sensor_info.id for sensor_info in measure.sensor_infos()
                    ]

                    # Run simulation
                    kernel_dict = self.kernel_dict(ctx=ctx)
                    run_results = mitsuba_run(kernel_dict, sensor_ids)

                    # Store results
                    measure.results[spectral_ctx.spectral_index] = run_results

                    # Update progress display
                    pbar.update()

    def postprocess(self, *measures: t.Union[Measure, int]) -> None:
        """
        Post-process raw results stored in a measure's ``results`` field. This
        requires a successful execution of :meth:`.process`. Post-processed results
        are stored in ``self.results``.

        Parameters
        ----------
        *measures : :class:`.Measure` or int
            One or several measures for which to perform post-processing.
            Alternatively, indexes in the measure array can be passed.
            If no value is passed, all measures are processed.

        Raises
        ------
        ValueError
            If ``measure.raw_results`` is ``None``, *i.e.* if :meth:`.process`
            has not been successfully run.

        See Also
        --------
        :meth:`.process`, :meth:`.run`
        """
        if not eradiate.mode().has_flags(ModeFlags.ANY_MONO | ModeFlags.ANY_CKD):
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

        if not measures:
            measures = self.measures

        for measure in measures:
            if isinstance(measure, int):
                measure = self.measures[measure]

            # Prepare measure postprocessing arguments
            measure_kwargs = {}
            if isinstance(measure, (DistantReflectanceMeasure, DistantAlbedoMeasure)):
                measure_kwargs["illumination"] = self.illumination

            # Collect measure results
            self._results[measure.id] = measure.postprocess(**measure_kwargs)

            # Apply additional metadata
            self._results[measure.id].attrs.update(self._dataset_metadata(measure))

    def _dataset_metadata(self, measure: Measure) -> t.Dict[str, str]:
        """
        Generate additional metadata applied to dataset after post-processing.
        Default implementation returns an empty dictionary.

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
