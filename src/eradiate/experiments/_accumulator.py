from __future__ import annotations

import logging
import typing as t

import attrs

from eradiate.scenes.biosphere._tree import MeshTreeElement
from eradiate.scenes.filters import FilterType
from eradiate.scenes.illumination._directional_periodic import (
    DirectionalPeriodicIllumination,
)
from eradiate.scenes.illumination._isotropic_periodic import (
    IsotropicPeriodicIllumination,
)
from eradiate.scenes.integrators._paccumulator import PAccumulatorIntegrator

from ._core import EarthObservationExperiment
from ._helpers import (
    check_geometry_atmosphere,
    surface_converter,
)
from ..attrs import define, documented
from ..scenes.atmosphere import (
    Atmosphere,
    HeterogeneousAtmosphere,
    HomogeneousAtmosphere,
    MolecularAtmosphere,
    atmosphere_factory,
)
from ..scenes.biosphere import Canopy, CanopyElement, biosphere_factory
from ..scenes.bsdfs import LambertianBSDF
from ..scenes.core import BoundingBox, SceneElement
from ..scenes.geometry import (
    PlaneParallelGeometry,
    SceneGeometry,
)
from ..scenes.measure import AbstractDistantMeasure, Measure
from ..scenes.shapes import RectangleShape
from ..scenes.surface import BasicSurface, CentralPatchSurface
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg

logger = logging.getLogger(__name__)


@define
class AccumulatorExperiment(EarthObservationExperiment):
    """
    Simulate radiation in a scene with an explicit canopy and atmosphere.
    This experiment assumes that the surface is plane and accounts for ground
    unit cell padding.

    Warnings
    --------
    * Canopy padding is controlled using the `padding` parameter: do *not* pad
      the canopy itself manually.

    Notes
    -----
    * A post-initialization step will constrain the measure setup if a
      distant measure is used and no target is defined:

      * if a canopy is defined, the target will be set to the top of the canopy
        unit cell (*i.e.* without its padding);
      * if no canopy is defined, the target will be set according to the
        atmosphere (*i.e.* to [0, 0, `toa`] where `toa` is the top-of-atmosphere
        altitude);
      * if neither atmosphere nor canopy are defined, the target is set to
        [0, 0, 0].

    * This experiment supports arbitrary measure positioning, except for
      :class:`.MultiRadiancemeterMeasure`, for which subsensor origins are
      required to be either all inside or all outside of the atmosphere. If an
      unsuitable configuration is detected, a :class:`ValueError` will be raised
      during initialization.

    * Currently this experiment is limited to the plane-parallel geometry.
    """

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

    bsdf_filters: dict[str, FilterType] | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(
                lambda x: {k: FilterType(v) for k, v in x.items()} if x else None
            ),
        ),
        doc="Dictionary mapping BSDF IDs to filter types. Allows fine-grained "
        "control over which specific BSDFs to include/exclude from measurements. "
        "BSDF IDs follow the pattern 'bsdf_{{element_id}}' (e.g., 'bsdf_tree_1', 'bsdf_leaf_cloud_1').",
        type="dict[str, .FilterType] or None",
        init_type="dict[str, .FilterType or int] or None, optional",
        default="None",
    )

    default_bsdf_filter: FilterType = documented(
        attrs.field(
            default=FilterType.INCLUDE,
            converter=FilterType,
            validator=attrs.validators.instance_of(FilterType),
        ),
        doc="Default filter type for BSDFs not explicitly listed in bsdf_filters. "
        "Use FilterType.INCLUDE to measure interactions, FilterType.IGNORE to exclude them.",
        type=".FilterType",
        init_type=".FilterType or int",
        default="FilterType.INCLUDE",
    )

    context_kwargs_ext: dict[str, t.Any] | None = documented(
        attrs.field(
            default=None,
        ),
        doc="Attempt at updating parameters without reloading... ",
        type="dict[str, t.Any] or None",
        init_type="None",
    )

    def __attrs_post_init__(self):
        self._normalize_spectral()
        self._normalize_atmosphere()
        self._normalize_measures()
        self._normalize_integrator()
        self._normalize_illumination()
        self._apply_bsdf_filters()

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

    def _normalize_measures(self) -> None:
        """
        Ensure that distant measure targets are set to appropriate values.
        Processed measures will have their ray target and origin parameters
        overridden if relevant.
        """
        for measure in self.measures:
            # Override ray target location if relevant
            if isinstance(measure, AbstractDistantMeasure):
                if measure.target is None:
                    if self.canopy is None:  # No canopy: target origin point
                        measure.target = {"type": "point", "xyz": [0, 0, 0]}

                    else:  # Canopy: target top of canopy
                        measure.target = {
                            "type": "rectangle",
                            "xmin": -0.5 * self.canopy.size[0],
                            "xmax": 0.5 * self.canopy.size[0],
                            "ymin": -0.5 * self.canopy.size[1],
                            "ymax": 0.5 * self.canopy.size[1],
                            "z": self.canopy.size[2],
                        }

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

    def get_bsdf_ids(self) -> list[str]:
        """
        Return list of all BSDF IDs that will be generated in the scene.

        This method helps users discover which BSDF IDs are available for filtering.
        The scene is temporarily built to extract BSDF IDs from templates.

        Returns
        -------
        list[str]
            List of BSDF IDs that can be used in bsdf_filters.
        """
        bsdf_ids = []

        def extract_bsdf_ids_from_element(element):
            """Recursively extract BSDF IDs from scene elements."""
            if hasattr(element, "_template_bsdfs"):
                try:
                    template = element._template_bsdfs
                    if isinstance(template, property):
                        template = template.fget(element)
                    elif callable(template):
                        template = template()

                    # Extract BSDF IDs from template keys
                    for key in template.keys():
                        if key.endswith(".type"):
                            bsdf_id = key.split(".")[0]
                            if bsdf_id not in bsdf_ids:
                                bsdf_ids.append(bsdf_id)
                except Exception:
                    # Skip if template generation fails
                    pass

            # Handle composite canopy elements recursively
            if hasattr(element, "instanced_canopy_elements"):
                for sub_element in element.instanced_canopy_elements:
                    if hasattr(sub_element, "canopy_element"):
                        extract_bsdf_ids_from_element(sub_element.canopy_element)

            # Handle discrete canopy elements list
            if hasattr(element, "elements"):
                for sub_element in element.elements:
                    if hasattr(sub_element, "canopy_element"):
                        extract_bsdf_ids_from_element(sub_element.canopy_element)
                    else:
                        extract_bsdf_ids_from_element(sub_element)

        # Extract BSDF IDs from canopy
        if self.canopy is not None:
            extract_bsdf_ids_from_element(self.canopy)

        # Extract BSDF IDs from surface (surface BSDFs typically don't follow the same pattern,
        # but we can try to get them if they have _template_bsdfs)
        if self.surface is not None:
            extract_bsdf_ids_from_element(self.surface)

        return sorted(bsdf_ids)

    def _apply_bsdf_filters(self):
        """Apply BSDF filters by updating canopy element filter fields."""
        if self.bsdf_filters is None and self.default_bsdf_filter == FilterType.INCLUDE:
            return

        def apply_filters_to_element(element):
            """Recursively apply filter settings to canopy elements."""

            # If this is a canopy element, update its filter field
            if isinstance(element, (CanopyElement, MeshTreeElement)):
                # Get the BSDF ID that will be generated for this element
                bsdf_id = f"bsdf_{element.id}"

                # Determine filter value: use specific filter if defined, otherwise use default
                if self.bsdf_filters and bsdf_id in self.bsdf_filters:
                    filter_value = self.bsdf_filters[bsdf_id]
                else:
                    filter_value = self.default_bsdf_filter

                # Update the element with the determined filter value
                element = attrs.evolve(element, bsdf_filter=filter_value)
                print(
                    f"Pippin filter: {filter_value}, id: {bsdf_id}, {element.bsdf_filter}"
                )
                # print(f"element: {element}")

            # Handle MeshTree's mesh_tree_elements
            if hasattr(element, "mesh_tree_elements"):
                updated_elements = []
                for mesh_element in element.mesh_tree_elements:
                    updated_mesh_element = apply_filters_to_element(mesh_element)
                    updated_elements.append(updated_mesh_element)
                element = attrs.evolve(element, mesh_tree_elements=updated_elements)

            # Handle composite canopy elements recursively
            if hasattr(element, "instanced_canopy_elements"):
                updated_elements = []
                for sub_element in element.instanced_canopy_elements:
                    if hasattr(sub_element, "canopy_element"):
                        updated_canopy_element = apply_filters_to_element(
                            sub_element.canopy_element
                        )
                        # print(f"{updated_canopy_element =}")
                        updated_sub_element = attrs.evolve(
                            sub_element, canopy_element=updated_canopy_element
                        )
                        # print(f"updated filter: {updated_sub_element.canopy_element.bsdf_filter}")
                        updated_elements.append(updated_sub_element)
                    else:
                        updated_elements.append(sub_element)
                        # print("recurse frodo")
                element = attrs.evolve(
                    element, instanced_canopy_elements=updated_elements
                )
                # print(f"{element =}")

            # Handle discrete canopy elements list
            if hasattr(element, "elements"):
                updated_elements = []
                for sub_element in element.elements:
                    if hasattr(sub_element, "canopy_element"):
                        updated_canopy_element = apply_filters_to_element(
                            sub_element.canopy_element
                        )
                        updated_sub_element = attrs.evolve(
                            sub_element, canopy_element=updated_canopy_element
                        )
                        updated_elements.append(updated_sub_element)
                    else:
                        updated_sub_element = apply_filters_to_element(sub_element)
                        updated_elements.append(updated_sub_element)
                element = attrs.evolve(element, elements=updated_elements)

            return element

        if self.canopy is not None:
            self.canopy = apply_filters_to_element(self.canopy)
        # print(f"{self.canopy.instanced_canopy_elements[0].canopy_element = }")

        if self.surface is not None:
            # If the surface has a BSDF, apply filters to it as well
            if hasattr(self.surface, "bsdf") and self.surface.bsdf is not None:
                bsdf_id = self.surface._bsdf_id
                if self.bsdf_filters and bsdf_id in self.bsdf_filters:
                    filter_value = self.bsdf_filters[bsdf_id]
                else:
                    filter_value = self.default_bsdf_filter

                # Update the surface BSDF filter
                self.surface = attrs.evolve(self.surface.bsdf, filter_type=filter_value)
                print(
                    f"Surface BSDF filter: {filter_value}, id: {bsdf_id}, {self.surface.bsdf.filter_type}"
                )

    def _dataset_metadata(self, measure: Measure) -> dict[str, str]:
        result = super()._dataset_metadata(measure)

        if measure.is_distant():
            result["title"] = "Top-of-atmosphere simulation results"

        return result

    @property
    def _context_kwargs(self) -> dict[str, t.Any]:
        kwargs = {} if self.context_kwargs_ext is None else {**self.context_kwargs_ext}

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
