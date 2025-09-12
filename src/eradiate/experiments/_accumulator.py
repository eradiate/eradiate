from __future__ import annotations

import logging
import typing as t
import warnings

import attrs
import mitsuba as mi
import pint
import pinttr
import xarray as xr
from hamilton.driver import Driver

import eradiate

from . import EarthObservationExperiment, Experiment, MeasureRegistry
from ._helpers import (
    check_geometry_atmosphere,
    surface_converter,
)
from .. import pipelines as pl
from ..attrs import define, documented, get_doc
from ..frame import AzimuthConvention
from ..scenes.atmosphere import (
    Atmosphere,
    HeterogeneousAtmosphere,
    HomogeneousAtmosphere,
    MolecularAtmosphere,
    atmosphere_factory,
)
from ..scenes.biosphere import Canopy, biosphere_factory
from ..scenes.bsdfs import LambertianBSDF
from ..scenes.core import BoundingBox, SceneElement
from ..scenes.filters import FilterFlags, FilterType
from ..scenes.geometry import (
    PlaneParallelGeometry,
    SceneGeometry,
)
from ..scenes.illumination._directional_periodic import (
    DirectionalPeriodicIllumination,
)
from ..scenes.illumination._isotropic_periodic import (
    IsotropicPeriodicIllumination,
)
from ..scenes.integrators import PAccumulatorIntegrator, integrator_factory
from ..scenes.measure import (
    AbsorbedFluxMeasure,
    CountMeasure,
    Measure,
    VoxelFluxMeasure,
    measure_factory,
)
from ..scenes.shapes import RectangleShape
from ..scenes.surface import BasicSurface, CentralPatchSurface
from ..spectral.response import BandSRF
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg

logger = logging.getLogger(__name__)


@define
class AccumulatorExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a scene with an explicit canopy and atmosphere.
    Accumulates quantities defined by the measures used. See `.measures`
    for the list of supported measures.
    This experiment assumes that the surface is plane and accounts for ground
    unit cell padding.
    TODO : is padding still relevant? since we now have periodic bounds we
    might not need padding anymore.

    Warnings
    --------
    * Canopy padding is controlled using the `padding` parameter: do *not* pad
      the canopy itself manually.

    Notes
    -----
    * Currently this experiment is limited to the plane-parallel geometry.
    """

    measures: MeasureRegistry = documented(
        attrs.field(
            factory=lambda: MeasureRegistry([CountMeasure()]),
            converter=lambda value: MeasureRegistry(
                pinttr.util.always_iterable(value)
                if not isinstance(value, dict)
                else [measure_factory.convert(value)]
            ),
            validator=attrs.validators.deep_iterable(
                member_validator=attrs.validators.instance_of(
                    (CountMeasure, AbsorbedFluxMeasure, VoxelFluxMeasure)
                ),
            ),
        ),
        doc=get_doc(Experiment, "measures", "doc")
        + "\n Warning: Currently the Accumulator experiment only accepts "
        ":class:`.CountMeasure`, :class:`.AbsorbedFluxMeasure`, and "
        ":class:`.VoxelFluxMeasure`",
        type=".MeasureRegistry",
        init_type="list of :class:`.Measure` or list of dict or "
        ":class:`.Measure` or dict",
        default=":class:`CountMeasure() <.CountMeasure>`",
    )

    integrator: PAccumulatorIntegrator = documented(
        attrs.field(
            factory=PAccumulatorIntegrator,
            converter=integrator_factory.convert,
            validator=attrs.validators.instance_of(PAccumulatorIntegrator),
        ),
        doc="Monte Carlo integration algorithm specification. "
        "The :class:`.PAccumualtorIntegrator` is the only compatible integrator "
        "with the accumulator experiment. It is a particle (forward) tracing "
        "algorithm that delegates accumulation of various measures to the measure. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.integrator_factory`.",
        type=":class:`.PAccumulatorIntegrator`",
        init_type=":class:`.PAccumulatorIntegrator` or dict",
        default=":class:`.PAccumulatorIntegrator`",
    )

    # Currently, only the plane parallel geometry is supported
    geometry: PlaneParallelGeometry = documented(
        attrs.field(
            default="plane_parallel",
            converter=SceneGeometry.convert,
            validator=attrs.validators.instance_of(PlaneParallelGeometry),
        ),
        doc="Problem geometry.",
        type=".PlaneParallelGeometry",
        init_type="str or dict or .PlaneParallelGeometry",
        default='"plane_parallel"',
    )

    atmosphere: Atmosphere | None = documented(
        attrs.field(
            factory=HomogeneousAtmosphere,
            converter=attrs.converters.optional(atmosphere_factory.convert),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(Atmosphere)
            ),
        ),
        doc="Atmosphere specification. If set to ``None``, no atmosphere will "
        "be added. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.atmosphere_factory`.",
        type=".Atmosphere or None",
        init_type=".Atmosphere or dict or None, optional",
        default=":class:`HomogeneousAtmosphere() <.HomogeneousAtmosphere>`",
    )

    canopy: Canopy | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(biosphere_factory.convert),
            validator=attrs.validators.optional(attrs.validators.instance_of(Canopy)),
        ),
        doc="Canopy specification. "
        "This parameter can be specified as a dictionary which will be "
        "interpreted by :data:`.biosphere_factory`.",
        type=".Canopy or None",
        init_type=".Canopy or dict or None, optional",
        default="None",
    )

    surface: BasicSurface | CentralPatchSurface | None = documented(
        attrs.field(
            factory=lambda: BasicSurface(bsdf=LambertianBSDF()),
            converter=attrs.converters.optional(surface_converter),
            validator=attrs.validators.optional(
                attrs.validators.instance_of((BasicSurface, CentralPatchSurface))
            ),
        ),
        doc="Surface specification. A :class:`.Surface` object may be passed: "
        "its shape specifications will be bypassed and the surface size will "
        "be computed automatically upon kernel dictionary generation. "
        "A :class:`.BSDF` may also be passed: it will be wrapped automatically "
        "in a :class:`.BasicSurface` instance. If a dictionary is passed, it "
        "will be first interpreted as a :class:`.BSDF`; if this fails, it will "
        "then be interpreted as a :class:`.Surface`. Finally, this field can "
        "be set to ``None``: in that case, no surface will be added.",
        type=".Surface or None",
        init_type=".Surface or .BSDF or dict or None, optional",
        default=":class:`BasicSurface(bsdf=LambertianBSDF()) <.BasicSurface>`",
    )

    periodic_box: BoundingBox | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(BoundingBox.convert),
        ),
        doc="Bounding box of the periodic boundary. Rays exiting one face "
        "of the boundary will enter back from the opposing face. Note that "
        "rays must originate from inside the periodic box when specified. "
        "See the family of periodic emitters e.g. :class:`.DirectionalPeriodicIllumination`",
        type=":class:`.BoundingBox` or None",
        init_type=":class:`.BoundingBox`, dict, tuple, or array-like, optional",
        default=None,
    )

    def __attrs_post_init__(self):
        self._normalize_spectral()
        self._normalize_atmosphere()
        self._normalize_integrator()
        self._normalize_illumination()

    def _normalize_atmosphere(self) -> None:
        """
        Enforce the experiment geometry on the atmosphere component(s).
        """
        if self.atmosphere is not None:
            # Since 'MolecularAtmosphere' cannot evaluate outside of its
            # vertical extent, we verify here that the experiment's geometry
            # comply with the atmosphere's vertical extent.
            if isinstance(self.atmosphere, MolecularAtmosphere):
                check_geometry_atmosphere(self.geometry, self.atmosphere)
            if isinstance(self.atmosphere, HeterogeneousAtmosphere):
                if self.atmosphere.molecular_atmosphere is not None:
                    check_geometry_atmosphere(
                        self.geometry, self.atmosphere.molecular_atmosphere
                    )

            # Override atmosphere geometry with experiment geometry
            self.atmosphere.geometry = self.geometry

            # The below call to update is required in the case of a
            # HeterogeneousAtmosphere, as it will propagate the geometry
            # override to its components.
            self.atmosphere.update()

    def _normalize_integrator(self) -> None:
        """
        Ensures that the integrator is compatible with the atmosphere and geometry.
        """
        if isinstance(self.integrator, PAccumulatorIntegrator):
            if self.periodic_box is not None:
                self.integrator = attrs.evolve(
                    self.integrator,
                    periodic_box=self.periodic_box,
                )

    def _normalize_illumination(self) -> None:
        """
        Ensures that the illumination source is compatible with the atmosphere and geometry.
        """
        if isinstance(
            self.illumination,
            (DirectionalPeriodicIllumination, IsotropicPeriodicIllumination),
        ):
            self.illumination = attrs.evolve(
                self.illumination,
                periodic_box=self.periodic_box,
            )

    def _dataset_metadata(self, measure: Measure) -> dict[str, str]:
        result = super()._dataset_metadata(measure)
        return result

    def get_filter_parameter_paths(self, filter_flags: FilterFlags) -> list[str]:
        """
        Get the path to the filter parameters in the kernel dictionnary. Note
        that this method will initialise the kernel dictionary if not already done.

        Parameters
        ----------
        filter_flags: FilterFlags
            Indicates which filter parameters should be retrieved.

        Returns
        -------
        list[str]
            List of filter of parameter paths.
        """
        if self.mi_scene is None:
            self.init()

        # Mapping from flags to Mitsuba types, to be used when traversing the scene.
        filter_flag_to_mi = {
            FilterFlags.BSDF: mi.BSDF,
            FilterFlags.SHAPE: mi.Shape,
            FilterFlags.PHASE: mi.PhaseFunction,
        }

        # Determine the parent types to look for based on the provided flags
        parent_type = []
        for flag in FilterFlags:
            if flag in filter_flags:
                parent_type.append(filter_flag_to_mi[flag])
        parent_type = tuple(parent_type)

        obj_wrapper = eradiate.kernel.mi_traverse(self.mi_scene.obj)
        filter_paths = []
        for k, v in obj_wrapper.parameters.properties.items():
            value, value_type, node, flags = v

            keep_param = True
            keep_param &= k.endswith(".filter")
            keep_param &= isinstance(node, parent_type)

            if mi.BSDF == parent_type or mi.BSDF in parent_type:
                keep_param &= node.class_().name() != "Null"

            if keep_param:
                filter_paths.append(k)

        return filter_paths

    def update_filters(self, filter_dict: dict[str, FilterType]) -> None:
        """
        Update the filter parameter in the kernel dictionary. The parameter path
        to the filter can be listed using `get_filter_parameter_paths`. Note
        that this method will initialise the kernel dictionary if not already done.

        Parameters
        ----------
        filter_dict : dict[str, FilterType]
            Mapping from filter parameter path to filter type.
        """
        if self.mi_scene is None:
            self.init()

        obj_wrapper = eradiate.kernel.mi_traverse(self.mi_scene.obj)
        for param_path, filter_value in filter_dict.items():
            obj_wrapper.parameters[param_path] = filter_value
        obj_wrapper.parameters.update()

    def update_illumination(
        self,
        zenith: pint.Quantity | None = None,
        azimuth: pint.Quantity | None = None,
        azimuth_convention: AzimuthConvention | None = None,
    ) -> None:
        """
        Update the illumination direction in the kernel dictionary. This is only
        available for directional illuminations. Note that this method will
        initialise the kernel dictionary if not already done.

        Parameters
        ----------
        zenith : pint.Quantity | None
            Illumination zenith angle.

        azimuth : pint.Quantity | None
            Illumination azimuth angle.

        azimuth_convention: AzimuthConvention | None
            Illumination azimuth convention.
        """

        if not isinstance(
            self.illumination,
            eradiate.scenes.illumination.AbstractDirectionalIllumination,
        ):
            warnings.warn(
                "Illumination not a directional illumination. Cannot update its direction."
            )
            return

        if self.mi_scene is None:
            self.init()

        if zenith is not None:
            self.illumination.zenith = zenith
        if azimuth is not None:
            self.illumination.azimuth = azimuth
        if azimuth_convention is not None:
            self.illumination.azimuth_convention = azimuth_convention

        self._normalize_illumination()

        param_path = None
        obj_wrapper = eradiate.kernel.mi_traverse(self.mi_scene.obj)
        for k, v in obj_wrapper.parameters.properties.items():
            value, value_type, node, flags = v
            keep_param = True

            keep_param &= k.endswith(".to_world")
            keep_param &= isinstance(node, mi.Emitter)

            if keep_param:
                param_path = k
                break

        if param_path is None:
            warnings.warn(
                "Could not find the to_world parameter of the emitter to update its direction."
            )
            return

        obj_wrapper.parameters[param_path] = self.illumination._to_world
        obj_wrapper.parameters.update()

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
        config = self._pipeline_config(measure)
        return eradiate.pipelines.driver(
            config, "eradiate.pipelines.definitions.accumulator"
        )

    def _pipeline_inputs(self, i_measure: int):
        # This convenience function collects pipeline inputs for a specific measure
        measure = self.measures[i_measure]
        result = {
            "tensors": measure.mi_results,
            "spectral_grid": self.spectral_grids[i_measure],
            "ckd_quads": self.ckd_quads[i_measure],
            "illumination": self.illumination,
            "srf": measure.srf,
        }
        return result

    def _pipeline_config(self, measure):
        result = {}

        # Which mode is selected?
        mode = eradiate.mode()
        result["mode_id"] = mode.id

        # Shall we apply spectral response function weighting (a.k.a convolution)?
        result["apply_spectral_response"] = isinstance(measure.srf, BandSRF)

        # Which physical variable are we processing?
        result["var_name"], result["var_metadata"] = measure.var

        # How to convert the raw tensor to a dataarray. Can either be a
        # `dict` or a `Callable[[mi.TensorXf], xr.DataArray]`.
        if hasattr(measure, "tensor_to_dataarray"):
            result["tensor_to_dataarray"] = measure.tensor_to_dataarray

        return result

    @property
    def _context_kwargs(self) -> dict[str, t.Any]:
        kwargs = {}

        return kwargs

    @property
    def _default_surface_width(self):
        return 10.0 * ucc.get("length")

    @property
    def scene_objects(self) -> dict[str, SceneElement]:
        # Inherit docstring

        objects = {}

        # Note: Object size computation logic
        # - The atmosphere, if set, must be the largest object in the
        #   scene. If the geometry setup defines the atmosphere width, it is
        #   used. Otherwise, a size is computed automatically.
        # - The canopy size must be lower than the atmosphere size if it is
        #   defined.
        # - The surface must be larger than the largest object in the scene.
        #   If the atmosphere is set, the surface matches its size.
        #   Otherwise, if the canopy is set, the surface matches its size.
        #   Otherwise, the surface defaults to a size possibly specified by the
        #   geometry setup.

        # Pre-process atmosphere
        if self.atmosphere is not None:
            atmosphere = attrs.evolve(self.atmosphere, geometry=self.geometry)
            atmosphere_width = self.geometry.width
        else:
            atmosphere = None
            atmosphere_width = 0.0 * ureg.m

        # Pre-process canopy
        if self.canopy is not None:
            canopy_width = max(self.canopy.size[:2])

            # if self.padding > 0:  # We must add extra instances if padding is requested
            #     canopy_width *= 2.0 * self.padding + 1.0
            #     canopy = self.canopy.padded_copy(self.padding)
            # else:
            canopy = self.canopy
        else:
            canopy = None
            canopy_width = 0.0 * ureg.m

        # Check sizes, compute surface size
        if atmosphere is not None:
            assert atmosphere_width > canopy_width
        surface_width = self._default_surface_width
        if canopy_width > surface_width:
            surface_width = canopy_width
        if atmosphere_width > surface_width:
            surface_width = atmosphere_width

        # Pre-process surface
        if self.surface is not None:
            altitude = (
                atmosphere.bottom_altitude if atmosphere is not None else 0.0 * ureg.km
            )
            surface = attrs.evolve(
                self.surface,
                shape=RectangleShape.surface(altitude=altitude, width=surface_width),
            )
        else:
            surface = None

        # Add all configured elements to the scene
        if atmosphere is not None:
            objects["atmosphere"] = atmosphere

        if canopy is not None:
            objects["canopy"] = canopy

        if surface is not None:
            objects["surface"] = surface

        objects.update(
            {
                "illumination": self.illumination,
                **{measure.id: measure for measure in self.measures},
                "integrator": self.integrator,
            }
        )

        return objects
