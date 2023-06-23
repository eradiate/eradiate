from __future__ import annotations

import logging
import typing as t
from abc import ABC, abstractmethod
from datetime import datetime

import attrs
import mitsuba as mi
import pinttr
import xarray as xr

import eradiate

from ._core import Experiment, _extra_objects_converter
from .. import pipelines, validators
from ..attrs import AUTO, documented, parse_docs
from ..contexts import KernelContext
from ..exceptions import UnsupportedModeError
from ..kernel import MitsubaObjectWrapper, mi_render, mi_traverse
from ..pipelines import Pipeline
from ..rng import SeedState
from ..scenes.core import Scene, SceneElement, get_factory, traverse
from ..scenes.illumination import (
    ConstantIllumination,
    DirectionalIllumination,
    SpotIllumination,
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
from ..scenes.spectra import InterpolatedSpectrum
from ..spectral.ckd import BinSet
from ..spectral.index import CKDSpectralIndex, MonoSpectralIndex, SpectralIndex
from ..spectral.mono import WavelengthSet
from ..util.misc import deduplicate_sorted, onedict_value

logger = logging.getLogger(__name__)


@parse_docs
@attrs.define
class GoniometerExperiment(Experiment):
    """A class for Experiments simulating gonioreflectometer observations"""

    target_objects: dict[str, SceneElement] = documented(
        attrs.field(
            factory=dict,
            converter=_extra_objects_converter,
            validator=attrs.validators.deep_mapping(
                key_validator=attrs.validators.instance_of(str),
                value_validator=attrs.validators.instance_of(SceneElement),
            ),
        ),
        doc="Dictionary of target objects to be added to the scene. "
        "The keys of this dictionary are used to identify the objects "
        "in the kernel dictionary.",
        type="dict",
        default="{}",
    )

    illumination: DirectionalIllumination | ConstantIllumination | SpotIllumination = (
        documented(
            attrs.field(
                factory=DirectionalIllumination,
                converter=illumination_factory.convert,
                validator=attrs.validators.instance_of(
                    (DirectionalIllumination, ConstantIllumination, SpotIllumination)
                ),
            ),
            doc="Illumination specification. "
            "This parameter can be specified as a dictionary which will be "
            "interpreted by :data:`.illumination_factory`.",
            type=":class:`.DirectionalIllumination`",
            init_type=":class:`.DirectionalIllumination` or dict",
            default=":class:`DirectionalIllumination() <.DirectionalIllumination>`",
        )
    )

    def __attrs_post_init__(self):
        if eradiate.mode().is_ckd:
            raise RuntimeError(f"unsupported mode '{eradiate.mode().id}'")
        self._normalize_spectral()

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
            "history": f"{datetime.utcnow().replace(microsecond=0).isoformat()}"
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
            generator = self.spectral_indices_mono
        else:
            raise RuntimeError(f"unsupported mode '{eradiate.mode().id}'")

        yield from generator(measure_index)

    def spectral_indices_mono(
        self, measure_index: int
    ) -> t.Generator[MonoSpectralIndex]:
        yield from self.spectral_set[measure_index].spectral_indices()

    @property
    def context_init(self) -> KernelContext:
        return KernelContext(
            si=SpectralIndex.new(),
            kwargs=self._context_kwargs,
        )

    @property
    def contexts(self) -> list[KernelContext]:
        # Inherit docstring

        # Collect contexts from all measures
        sis = []

        for measure_index, measure in enumerate(self.measures):
            _si = list(self.spectral_indices(measure_index))
            sis.extend(_si)

        # Sort and remove duplicates
        key = {
            MonoSpectralIndex: lambda si: si.w.m,
            CKDSpectralIndex: lambda si: (si.w.m, si.g),
        }[type(sis[0])]

        sis = deduplicate_sorted(
            sorted(sis, key=key), cmp=lambda x, y: key(x) == key(y)
        )
        kwargs = self._context_kwargs

        return [KernelContext(si, kwargs=kwargs) for si in sis]

    @property
    def _context_kwargs(self) -> dict[str, t.Any]:
        return {}

    @property
    def scene_objects(self) -> dict[str, SceneElement]:
        # Inherit docstring

        objects = {}
        objects.update(
            {
                "illumination": self.illumination,
                **{measure.id: measure for measure in self.measures},
                "integrator": self.integrator,
            }
        )

        return objects

    @property
    def scene(self) -> Scene:
        """
        Return a scene object used for kernel dictionary template and parameter
        table generation.
        """
        return Scene(objects={**self.scene_objects, **self.target_objects})

    def init(self):
        # Inherit docstring

        logger.info("Initializing kernel scene")

        kdict_template, umap_template = traverse(self.scene)

        try:
            self.mi_scene = mi_traverse(
                mi.load_dict(kdict_template.render(ctx=self.context_init)),
                umap_template=umap_template,
            )
        except RuntimeError as e:
            raise RuntimeError(f"(while loading kernel scene dictionary){e}") from e

        # Remove unused elements from Mitsuba scene parameter table
        self.mi_scene.drop_parameters()

    def process(
        self,
        spp: int = 0,
        seed_state: SeedState | None = None,
    ) -> None:
        # Inherit docstring

        # Set up Mitsuba scene
        if self.mi_scene is None:
            self.init()

        # Run Mitsuba for each context
        logger.info("Launching simulation")

        mi_results = mi_render(
            self.mi_scene,
            self.contexts,
            seed_state=seed_state,
            spp=spp,
        )

        # Assign collected results to the appropriate measure
        sensor_to_measure: dict[str, Measure] = {
            measure.sensor_id: measure for measure in self.measures
        }

        for ctx_index, spectral_group_dict in mi_results.items():
            for sensor_id, mi_bitmap in spectral_group_dict.items():
                measure = sensor_to_measure[sensor_id]
                measure.mi_results[ctx_index] = {
                    "bitmap": mi_bitmap,
                    "spp": spp if spp > 0 else measure.spp,
                }

    def postprocess(self, pipeline_kwargs: dict | None = None) -> None:
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
        measure_index = self.measures.index(measure)

        pipeline = pipelines.Pipeline()

        # Gather
        pipeline.add(
            "gather",
            pipelines.Gather(var=measure.var),
        )

        # Aggregate
        if eradiate.mode().is_ckd:
            pipeline.add(
                "aggregate_ckd_quad",
                pipelines.AggregateCKDQuad(
                    measure=measure,
                    binset=self.spectral_set[measure_index],
                    var=measure.var[0],
                ),
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
        if isinstance(
            self.illumination, (DirectionalIllumination, ConstantIllumination)
        ):
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

        if isinstance(measure.srf, InterpolatedSpectrum):
            pipeline.add(
                "add_srf",
                pipelines.AddSpectralResponseFunction(measure=measure),
            )

        return pipeline
