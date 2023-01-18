import logging
import typing as t
from abc import ABC, abstractmethod

import attrs
import mitsuba as mi
import xarray as xr
from tqdm.auto import tqdm

from .._config import ProgressLevel, config
from ..attrs import documented
from ..contexts import KernelDictContext
from ..pipelines import Pipeline
from ..rng import SeedState, root_seed_state
from ..scenes.core import ParameterMap
from ..scenes.illumination import DirectionalIllumination, illumination_factory
from ..util.misc import onedict_value

logger = logging.getLogger(__name__)


def mi_render(
    mi_scene: "mitsuba.Scene",
    params: ParameterMap,
    ctxs: t.List[KernelDictContext],
    mi_params: t.Optional["mitsuba.SceneParameters"] = None,
    sensors: t.Union[None, int, t.List[int]] = 0,
    spp: int = 4,
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

    spp : int, optional, default: 4
        Number of samples per pixel.

    seed_state : :class:`.SeedState, optional
        Seed state used to generate seeds to initialise Mitsuba's RNG at
        each run. If unset, Eradiate's root seed state is used.
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
                    spp=spp,
                    seed=int(seed_state.next()),
                )

                # Store result in a new Bitmap object
                results[(ctx.spectral_ctx.spectral_index, mi_sensor.id())] = mi.Bitmap(
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

    results: t.Dict[str, xr.Dataset] = attrs.field(
        factory=dict,
        repr=False,
    )

    def clear(self):
        """
        Clear previous experiment results and reset internal state.
        """
        self.mi_params = None
        self.params = None
        self.mi_results.clear()
        self.results.clear()

    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def process(self, spp: int = 4, seed_state: t.Optional[SeedState] = None) -> None:
        pass

    @abstractmethod
    def postprocess(self) -> None:
        pass

    @abstractmethod
    def pipeline(self) -> t.Tuple[Pipeline, ...]:
        pass


@attrs.define
class EarthObservationExperiment(Experiment, ABC):
    illumination: t.Union[DirectionalIllumination] = documented(
        attrs.field(
            factory=DirectionalIllumination,
            converter=illumination_factory.convert,
            validator=attrs.validators.instance_of(DirectionalIllumination),
        ),
        doc="Illumination specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.illumination_factory`.",
        type=":class:`.DirectionalIllumination`",
        init_type=":class:`.DirectionalIllumination` or dict",
        default=":class:`DirectionalIllumination() <.DirectionalIllumination>`",
    )

    def process(self, spp: int = 4, seed_state: t.Optional[SeedState] = None) -> None:
        # Generate context sequence
        ctxs = ...
        raise NotImplementedError

        # Map measures to sensors
        measures, sensors = ...

        # Run Mitsuba for each context
        mi_results = mi_render(
            self.mi_scene,
            self.params,
            ctxs,
            sensors=sensors,
            mi_params=self.mi_params,
            seed_state=seed_state,
        )

        # Assign collected results to the appropriate measure

    def postprocess(self, pipeline_kwargs: t.Optional[t.Dict] = None) -> None:
        measures = self.measures

        if pipeline_kwargs is None:
            pipeline_kwargs = {}

        # Apply pipelines
        for measure in measures:
            pipeline = self.pipeline(measure)

            # Collect measure results
            self._results[measure.id] = pipeline.transform(
                measure.results, **pipeline_kwargs
            )

            # Apply additional metadata
            self._results[measure.id].attrs.update(self._dataset_metadata(measure))


# ------------------------------------------------------------------------------
#                              Experiment runner
# ------------------------------------------------------------------------------


def run(
    exp: Experiment,
    spp: int = 4,
    seed_state: t.Optional[SeedState] = None,
) -> t.Tuple[xr.Dataset]:
    exp.process(spp=spp, seed_state=seed_state)
    exp.postprocess()
    return exp.results if len(exp.results) > 1 else onedict_value(exp.results)
