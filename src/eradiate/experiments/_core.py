from __future__ import annotations

import datetime
import logging
import typing as t
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence

import attrs
import mitsuba as mi
import numpy as np
import pinttr
import xarray as xr
from hamilton.driver import Driver

import eradiate

from .. import converters, validators
from .. import pipelines as pl
from ..attrs import AUTO, define, documented, frozen
from ..contexts import KernelContext
from ..exceptions import UnsupportedModeError
from ..kernel import (
    KernelDict,
    KernelSceneParameterMap,
    MitsubaObjectWrapper,
    mi_render,
    mi_traverse,
)
from ..quad import Quad
from ..rng import SeedState
from ..scenes.core import Scene, SceneElement, get_factory, traverse
from ..scenes.illumination import (
    AbstractDirectionalIllumination,
    ConstantIllumination,
    DirectionalIllumination,
    illumination_factory,
)
from ..scenes.integrators import Integrator, integrator_factory
from ..scenes.measure import (
    Measure,
    MultiDistantMeasure,
    measure_factory,
)
from ..spectral.ckd_quad import CKDQuadConfig
from ..spectral.grid import CKDSpectralGrid, MonoSpectralGrid, SpectralGrid
from ..spectral.index import CKDSpectralIndex, MonoSpectralIndex, SpectralIndex
from ..units import unit_registry as ureg
from ..util.misc import deduplicate_sorted

logger = logging.getLogger(__name__)


@frozen(init=False)
class MeasureRegistry(Sequence):
    """
    A simple list wrapper holding measures, with additional lookup methods and
    metadata.

    The constructor converts dictionaries automatically and checks for duplicate
    IDs (raises a :class:`ValueError` if it finds any).
    """

    _measures: list[Measure] = attrs.field(
        converter=list,
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.instance_of(Measure)
        ),
    )
    _id_to_idx: dict[str, int] = attrs.field(factory=dict, repr=False)
    _idx_to_id: dict[int, str] = attrs.field(factory=dict, repr=False)

    def __init__(self, measures: t.Sequence):
        # Convert all values to a measure
        measures = [measure_factory.convert(x) for x in measures]

        # Check for duplicate IDs
        values = dict(zip(*np.unique([m.id for m in measures], return_counts=True)))
        duplicate_ids = [k for k, v in values.items() if v > 1]
        if duplicate_ids:
            raise ValueError(f"duplicate measure ids: {duplicate_ids}")

        # Update index and ID registries
        id_to_idx = {m.id: i for i, m in enumerate(measures)}
        idx_to_id = {i: m.id for i, m in enumerate(measures)}

        # Finalize initialization
        self.__attrs_init__(measures=measures, id_to_idx=id_to_idx, idx_to_id=idx_to_id)

    def __getitem__(self, index):
        return self._measures[index]

    def __len__(self):
        return len(self._measures)

    def get_index(self, value: str | int) -> int:
        """
        Get the index corresponding to a given measure ID. Integers are passed
        through.
        """
        if isinstance(value, str):
            return self._id_to_idx[value]
        elif isinstance(value, int):
            return value
        else:
            raise TypeError(f"unhandled type {type(value)}")

    def get_id(self, value: str | int) -> str:
        """
        Get the ID corresponding to a given index. Strings are passed through.
        """
        if isinstance(value, str):
            return value
        elif isinstance(value, int):
            return self._idx_to_id[value]
        else:
            raise TypeError(f"unhandled type {type(value)}")

    def resolve(self, value: str | int) -> Measure:
        """
        Resolve a measure based on its ID or index.
        """
        return self[self.get_index(value)]


@define
class Experiment(ABC):
    """
    Abstract base class for all Eradiate experiments. An experiment consists of
    a high-level scene specification parametrized by natural user input, a
    processing and post-processing pipeline, and a result storage data
    structure.
    """

    # Internal Mitsuba scene. This member is not set by the end-user, but rather
    # by the Experiment itself during initialization.
    mi_scene: MitsubaObjectWrapper | None = attrs.field(
        default=None,
        repr=False,
        init=False,
    )

    measures: MeasureRegistry = documented(
        attrs.field(
            factory=lambda: MeasureRegistry([MultiDistantMeasure()]),
            converter=lambda value: MeasureRegistry(
                pinttr.util.always_iterable(value)
                if not isinstance(value, dict)
                else [measure_factory.convert(value)]
            ),
        ),
        doc="List of measure specifications. The passed list may contain "
        "dictionaries, which will be interpreted by "
        ":data:`.measure_factory`. "
        "Optionally, a single :class:`.Measure` or dictionary specification "
        "may be passed and will automatically be wrapped into a list.",
        type=".MeasureRegistry",
        init_type="list of :class:`.Measure` or list of dict or "
        ":class:`.Measure` or dict",
        default=":class:`MultiDistantMeasure() <.MultiDistantMeasure>`",
    )

    integrator: Integrator = documented(
        attrs.field(
            default=AUTO,
            converter=converters.auto_or(integrator_factory.convert),
            validator=validators.auto_or(
                attrs.validators.instance_of(Integrator),
            ),
        ),
        doc="Monte Carlo integration algorithm specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.integrator_factory`."
        "The integrator defaults to :data:`AUTO`, which will choose the appropriate "
        "integrator depending on the experiment's configuration. ",
        type=":class:`.Integrator` or AUTO",
        init_type=":class:`.Integrator` or dict or AUTO",
        default="AUTO",
    )

    # Storage for results, for each computed measure
    _results: dict[str, xr.Dataset] = attrs.field(factory=dict, repr=False)

    @property
    def results(self) -> dict[str, xr.Dataset]:
        """
        Post-processed simulation results.

        Returns
        -------
        dict[str, Dataset]
            Dictionary mapping measure IDs to xarray datasets.
        """
        return self._results

    _background_spectral_grid: SpectralGrid = documented(
        attrs.field(
            default=AUTO,
            validator=validators.auto_or(attrs.validators.instance_of(SpectralGrid)),
            repr=False,
        ),
        doc="Background spectral grid. "
        "If the value is :data:`.AUTO`, the background spectral grid is "
        "automatically generated depending on the active mode and internal "
        "experiment constraints. Otherwise, the value must be convertible to "
        "a :class:`.SpectralGrid` instance.",
        type=".SpectralGrid or AUTO",
        init_type=".SpectralGrid or AUTO",
        default="AUTO",
    )

    @property
    def background_spectral_grid(self) -> SpectralGrid:
        return self._background_spectral_grid

    # Grid used to walk the spectral dimension for each measure.
    # Set upon initialization by the '_normalize_spectral()' method.
    _spectral_grids: dict[int, SpectralGrid] = attrs.field(
        factory=dict, init=False, repr=False
    )

    @property
    def spectral_grids(self) -> dict[int, SpectralGrid]:
        """
        A dictionary mapping measure index to the associated spectral grid.
        """
        return self._spectral_grids

    ckd_quad_config: CKDQuadConfig = documented(
        attrs.field(
            factory=lambda: CKDQuadConfig(ng_max=16),
            converter=CKDQuadConfig.convert,
            validator=attrs.validators.instance_of(CKDQuadConfig),
        ),
        doc="CKD quadrature rule generation configuration.",
        type=".CKDQuadConfig",
        init_type=".CKDQuadConfig or dict",
    )

    # CKD quadrature configuration for each bin.
    # Set upon initialization by the '_normalize_spectral()' method.
    _ckd_quads: dict[int, list[Quad]] = attrs.field(
        factory=dict, init=False, repr=False
    )

    @property
    def ckd_quads(self) -> dict[int, list[Quad]]:
        """
        A dictionary mapping measure index to the associated CKD quadrature rule
        (if relevant).
        """
        return self._ckd_quads

    def __attrs_post_init__(self):
        self._normalize_spectral()

    def _normalize_spectral(self) -> None:
        """
        Assemble a spectral grid based on the various elements in the scene.
        """
        # Collect atmosphere-based grid if relevant
        atmosphere = getattr(self, "atmosphere", None)
        abs_db = None
        if atmosphere is not None:
            abs_db = getattr(atmosphere, "absorption_data", None)
        if abs_db is not None:
            if self._background_spectral_grid is not AUTO:
                warnings.warn(
                    "User-specified a background spectral grid is overridden by "
                    "atmosphere spectral grid."
                )
            self._background_spectral_grid = SpectralGrid.from_absorption_database(
                atmosphere.absorption_data
            )

        # If needed, set the background grid
        if self._background_spectral_grid is AUTO:
            self._background_spectral_grid = SpectralGrid.default()

        # Select subparts of the grid that are covered by the SRF
        self._spectral_grids = {
            i: self.background_spectral_grid.select(measure.srf)
            for i, measure in enumerate(self.measures)
        }

        # Get quadrature rules for all bins
        ckd_quads = {}
        for i, measure in enumerate(self.measures):
            if eradiate.mode().is_ckd:
                spectral_grid: CKDSpectralGrid = self._spectral_grids[i]
                ckd_quads[i] = [
                    x[1] for x in spectral_grid.walk_quads(self.ckd_quad_config, abs_db)
                ]
            else:
                ckd_quads[i] = []
        self._ckd_quads = ckd_quads

    def clear(self) -> None:
        """
        Clear previous experiment results and reset internal state.
        """
        self.results.clear()

        for measure in self.measures:
            measure.mi_results.clear()

    @abstractmethod
    def init(self, drop_parameters: bool = True) -> None:
        """
        Generate kernel dictionary and initialize Mitsuba scene.

        Parameters
        ----------
        drop_parameters : bool
            If ``True``, drop Mitsuba scene parameters that are not used (*i.e.*
            that do not have an updater associated).
        """
        pass

    @abstractmethod
    def process(
        self,
        measures: None | int | list[int] = None,
        spp: int = 0,
        seed_state: SeedState | None = None,
    ) -> None:
        """
        Run simulation and collect raw results.

        Parameters
        ----------
        measures : int or list of int, optional
            Indices of the measures that will be processed. By default, all
            measures are processed.

        spp : int, optional
            Sample count. If set to 0, the value set in the original scene
            definition takes precedence.

        seed_state : :class:`.SeedState`, optional
            Seed state used to generate seeds to initialize Mitsuba's RNG at
            every iteration of the parametric loop. If unset, Eradiate's
            :attr:`root seed state <.root_seed_state>` is used.
        """
        pass

    @abstractmethod
    def postprocess(self, measures: None | int | list[int] = None) -> None:
        """
        Post-process raw results and store them in :attr:`results`.

        Parameters
        ----------
        measures : int or list of int, optional
            Indices of the measures that will be processed. By default, all
            measures are processed.
        """
        pass

    @abstractmethod
    def pipeline(self, measure: Measure | int) -> Driver:
        """
        Return the post-processing pipeline for a given measure.

        Parameters
        ----------
        measure : .Measure or int
            Measure for which the pipeline is generated.

        Returns
        -------
        hamilton.driver.Driver
        """
        pass

    @abstractmethod
    def context_init(self) -> KernelContext:
        """
        Return a single context used for scene initialization.
        """
        pass

    @abstractmethod
    def contexts(self, measures: None | int | list[int] = None) -> list[KernelContext]:
        """
        Return a list of contexts used for processing.

        Parameters
        ----------
        measures : int or list of int, optional
            A list of the indexes of the measures to account for when emitting
            kernel contexts. If unset, all measures are accounted for.

        Returns
        -------
        list of .KernelContext
        """
        pass


def _extra_objects_converter(value: dict | None) -> dict:
    if not value:
        return {}

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


@define
class EarthObservationExperiment(Experiment, ABC):
    """
    Abstract based class for experiments illuminated by a distant directional
    emitter.
    """

    extra_objects: dict[str, SceneElement] = documented(
        attrs.field(
            default=None,
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
        init_type="dict or None",
        default="None",
    )

    illumination: AbstractDirectionalIllumination | ConstantIllumination = documented(
        attrs.field(
            factory=DirectionalIllumination,
            converter=illumination_factory.convert,
            validator=attrs.validators.instance_of(
                (AbstractDirectionalIllumination, ConstantIllumination)
            ),
        ),
        doc="Illumination specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.illumination_factory`.",
        type=":class:`.AbstractDirectionalIllumination` or "
        ":class:`.ConstantIllumination`",
        init_type=":class:`.DirectionalIllumination` or "
        ":class:`.ConstantIllumination` or dict",
        default=":class:`DirectionalIllumination() <.DirectionalIllumination>`",
    )

    kdict: KernelDict = documented(
        attrs.field(factory=KernelDict, converter=KernelDict),
        doc="Additional kernel dictionary template appended to the "
        "experiment-controlled template.",
        type=".KernelDict",
        init_type="mapping",
        default="{}",
    )

    kpmap: KernelSceneParameterMap = documented(
        attrs.field(factory=KernelSceneParameterMap, converter=KernelSceneParameterMap),
        doc="Additional scene parameter update map template appended to the "
        "experiment-controlled template.",
        type=".KernelSceneParameterMap",
        init_type="mapping",
        default="{}",
    )

    def kdict_base(self) -> KernelDict:
        # This is inefficient and exists at the moment only for debugging purposes
        return traverse(self.scene)[0]

    def kdict_full(self) -> KernelDict:
        # Return the user-defined kdict template merged with additional scene
        # element contributions
        kdict = self.kdict_base()
        kdict.update(self.kdict)
        return kdict

    def kpmap_base(self) -> KernelSceneParameterMap:
        # This is inefficient and exists at the moment only for debugging purposes
        return traverse(self.scene)[1]

    def kpmap_full(self) -> KernelSceneParameterMap:
        # Return the user-defined kpmap template merged with additional scene
        # element contributions
        kpmap = self.kpmap_base()
        kpmap.update(self.kpmap)
        return kpmap

    def _dataset_metadata(self, measure: Measure) -> dict[str, str]:
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
            "convention": "CF-1.10",
            "source": f"eradiate, version {eradiate.__version__}",
            "history": f"{datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()}"
            f" - data creation - {self.__class__.__name__}.postprocess()",
            "references": "",
        }

    def spectral_indices(self, measure_index: int) -> t.Generator[SpectralIndex]:
        """
        Generate spectral indices for a given measure.

        Parameters
        ----------
        measure_index : int
            Measure index for which spectral indices are generated.

        Yields
        ------
        :class:`.SpectralIndex`
            Spectral index.
        """

        if eradiate.mode().is_mono:
            spectral_grid: MonoSpectralGrid = self.spectral_grids[measure_index]

            def generator():
                yield from spectral_grid.walk_indices()

        elif eradiate.mode().is_ckd:
            spectral_grid: CKDSpectralGrid = self.spectral_grids[measure_index]
            quad_config = self.ckd_quad_config
            try:
                abs_db = self.atmosphere.abs_db
            except (
                AttributeError
            ):  # There is either no atmosphere or no absorption database
                abs_db = None

            def generator():
                yield from spectral_grid.walk_indices(quad_config, abs_db)
        else:
            raise UnsupportedModeError

        yield from generator()

    def context_init(self):
        # Inherit docstring

        return KernelContext(
            si=self.spectral_indices(0).__next__(), kwargs=self._context_kwargs()
        )

    @abstractmethod
    def _context_kwargs(self) -> dict[str, t.Any]:
        pass

    def contexts(self, measures: None | int | list[int] = None) -> list[KernelContext]:
        # Inherit docstring

        # Collect contexts from all measures
        sis = []

        if measures is None:
            measures = list(range(len(self.measures)))

        if isinstance(measures, int):
            measures = [measures]

        for measure_index in measures:
            _si = list(self.spectral_indices(measure_index))
            sis.extend(_si)

        # Sort and remove duplicates
        key = {
            MonoSpectralIndex: lambda si: si.w.m,
            CKDSpectralIndex: lambda si: (si.w.m, si.g),
        }[SpectralIndex.subtypes.resolve()]

        sis = deduplicate_sorted(
            sorted(sis, key=key), cmp=lambda x, y: key(x) == key(y)
        )

        kwargs = self._context_kwargs()
        return [KernelContext(si, kwargs=kwargs) for si in sis]

    @property
    @abstractmethod
    def scene_objects(self) -> dict[str, SceneElement]:
        """
        Return a dictionary of string identifiers to the :class:`.SceneElement`
        instances generated by this experiment.

        Notes
        -----
        These elements make up the core scene contents for this experiment, and
        are merged together with the contents of the ``extra_objects``
        dictionary to initialize the underlying kernel scene. If you are writing
        your own :class:`.EarthObservationExperiment` subclass, this property is
        critical.
        """
        pass

    @property
    def scene(self) -> Scene:
        """
        Return a scene object used for kernel dictionary template and parameter
        table generation.
        """
        return Scene(objects={**self.scene_objects, **self.extra_objects})

    def init(self, drop_parameters: bool = True):
        # Inherit docstring

        logger.info("Initializing kernel scene")

        kdict_template, umap_template = traverse(self.scene)
        kdict_template.update(self.kdict)
        umap_template.update(self.kpmap)
        try:
            ctx = self.context_init()
            self.mi_scene = mi_traverse(
                mi.load_dict(kdict_template.render(ctx=ctx)),
                umap_template=umap_template,
            )
        except RuntimeError as e:
            raise RuntimeError(f"(while loading kernel scene dictionary){e}") from e

        # Remove unused elements from Mitsuba scene parameter table
        if drop_parameters:
            self.mi_scene.drop_parameters()

    def process(
        self,
        measures: None | int | str | list[int | str] = None,
        spp: int = 0,
        seed_state: SeedState | None = None,
    ) -> None:
        # Inherit docstring

        # Set up Mitsuba scene
        if self.mi_scene is None:
            self.init()

        # Normalize list of processed measures
        if measures is None:
            measures = self.measures
        else:
            if isinstance(measures, (int, str)):
                measures = [measures]
            measures = [self.measures.resolve(i) for i in measures]

        # Generate kernel contexts
        measure_idxs = [self.measures.get_index(measure.id) for measure in measures]
        ctxs = self.contexts(measure_idxs)

        # Collect active sensor IDs
        active_sensors = [measure.sensor_id for measure in measures]
        mi_sensors = self.mi_scene.obj.sensors()
        active_sensors = [
            i for i, sensor in enumerate(mi_sensors) if sensor.id() in active_sensors
        ]

        # Run Mitsuba for each context
        logger.info("Launching simulation")

        mi_results = mi_render(
            self.mi_scene,
            ctxs=ctxs,
            sensors=active_sensors,
            seed_state=seed_state,
            spp=spp,
        )

        # Assign collected results to the appropriate measure
        sensor_to_measure: dict[str, Measure] = {
            measure.sensor_id: measure for measure in measures
        }

        def convert_to_y_format(img):
            img_np = np.array(img, copy=False)[:, :, [0]]
            return mi.Bitmap(img_np, mi.Bitmap.PixelFormat.Y)

        # Map bitmap names to result names
        mapping = {}
        if self.integrator.stokes:
            stokes = ["S0", "S1", "S2", "S3"]
            iquv = ["I", "Q", "U", "V"]

            if self.integrator.moment:
                stokes = ["nested." + s for s in stokes]
                stokes += ["m2_" + s for s in stokes]
                iquv += ["m2_" + s for s in iquv]

            for s, i in zip(stokes, iquv):
                mapping[s] = i

        else:
            mapping = {"<root>": "bitmap"}
            if self.integrator.moment:
                mapping["m2_nested"] = "m2"

        # gather results and info from measures
        for ctx_index, spectral_group_dict in mi_results.items():
            for sensor_id, mi_bitmap in spectral_group_dict.items():
                measure = sensor_to_measure[sensor_id]
                result_imgs = {"spp": spp if spp > 0 else measure.spp}

                splits = mi_bitmap.split()
                for split in splits:
                    if split[0] in mapping:
                        img = split[1]
                        # convert any result that has more than one channel
                        if img.pixel_format() != mi.Bitmap.PixelFormat.Y:
                            img = convert_to_y_format(img)
                        result_imgs[mapping[split[0]]] = img

                measure.mi_results[ctx_index] = result_imgs

    def postprocess(self, measures: None | int | list[int] = None) -> None:
        # Inherit docstring
        logger.info("Post-processing results")

        if measures is None:
            measures = list(range(len(self.measures)))
        else:
            if isinstance(measures, (int, str)):
                measures = [self.measures.get_index(measures)]

        # Run pipelines
        for i in measures:
            measure = self.measures[i]
            drv: Driver = self.pipeline(measure)
            inputs = self._pipeline_inputs(i)
            outputs = pl.outputs(drv)
            result = drv.execute(final_vars=outputs, inputs=inputs)
            self.results[measure.id] = xr.Dataset({var: result[var] for var in outputs})

    def pipeline(self, measure: Measure | int | str) -> Driver:
        # Inherit docstring
        if isinstance(measure, (int, str)):
            measure = self.measures.resolve(measure)
        config = pl.config(measure, integrator=self.integrator)
        return eradiate.pipelines.driver(config, "eradiate.pipelines.definitions.core")

    def _pipeline_inputs(self, i_measure: int):
        # This convenience function collects pipeline inputs for a specific measure

        measure = self.measures[i_measure]
        result = {
            "bitmaps": measure.mi_results,
            "spectral_grid": self.spectral_grids[i_measure],
            "ckd_quads": self.ckd_quads[i_measure],
            "illumination": self.illumination,
            "srf": measure.srf,
        }

        config = pl.config(measure)
        if config.get("add_viewing_angles", False):
            result["angles"] = measure.viewing_angles.m_as(ureg.deg)
        else:
            result["viewing_angles"] = None

        return result


# ------------------------------------------------------------------------------
#                              Experiment runner
# ------------------------------------------------------------------------------


def run(
    exp: Experiment,
    measures: None | int | str | list[int | str] = None,
    spp: int = 0,
    seed_state: SeedState | None = None,
) -> xr.Dataset | dict[str, xr.Dataset]:
    """
    Run an Eradiate experiment. This function performs kernel scene assembly,
    runs the computation and post-processes the raw results. The output consists
    of one or several xarray datasets.

    Parameters
    ----------
    exp : Experiment
        Reference to the experiment object which will be processed.

    measures : int or str or list of int or str, optional
        Indices of the measures that will be processed. By default, all measures
        are processed.

    spp : int, optional, default: 0
        Optional parameter to override the number of samples per pixel for all
        computed measures. If set to 0, the configured value for each measure
        takes precedence.

    seed_state : :class:`.SeedState`, optional
            Seed state used to generate seeds to initialize Mitsuba's RNG at
            every iteration of the parametric loop. If unset, Eradiate's
            :attr:`root seed state <.root_seed_state>` is used.

    Returns
    -------
    Dataset or dict[str, Dataset]
        If a single measure is processed, a single xarray dataset is returned.
        If several measures are processed, a dictionary mapping measure IDs to
        the corresponding result dataset is returned.

    Notes
    -----
    * Successive calls to this function with different measures will not reset
      the :attr:`Experiment.results` dictionary.
    * Successive calls with already processed measures will overwrite prior
      results.
    """
    if measures is None:
        measures = list(range(len(exp.measures)))
    if isinstance(measures, (int, str)):
        measures = [measures]

    exp.process(spp=spp, measures=measures, seed_state=seed_state)
    exp.postprocess(measures=measures)

    measure_ids = [exp.measures.get_id(m) for m in measures]
    return (
        {x: exp.results[x] for x in measure_ids}
        if len(measure_ids) > 1
        else exp.results[measure_ids[0]]
    )
