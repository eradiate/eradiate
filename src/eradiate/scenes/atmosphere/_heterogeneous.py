"""
Heterogeneous atmospheres.
"""

from __future__ import annotations

import warnings
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

import attrs
import mitsuba as mi
import pint

from ._core import (
    AbstractHeterogeneousAtmosphere,
    AtmosphereComponent,
    atmosphere_factory,
    geometry_on_grid,
)
from ._merge import (
    MergeAll,
    MergedComponent,
    MergePolicy,
    merge_policy_converter,
)
from ._molecular import MolecularComponent
from ._particle_component import ParticleComponent
from .._gridvolume import generate_gridvolume
from ..core import traverse
from ..geometry import (
    FilterType,
    PlaneParallelGeometry,
    SphericalShellGeometry,
)
from ..phase import PhaseFunction
from ...attrs import define, documented
from ...grid import SpatialGrid
from ...kernel import KernelSceneParameterFlags, SceneParameter, SearchSceneParameter
from ...spectral.index import SpectralIndex
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...util.deprecation import DeprecatedWarning, renamed_kwargs
from ...util.misc import flatten

if TYPE_CHECKING:
    from axsdb import AbsorptionDatabase


def _molecular_converter(value):
    if isinstance(value, MutableMapping) and ("type" not in value):
        value["type"] = "molecular"
    return atmosphere_factory.convert(value, allowed_cls=MolecularComponent)


def _particles_converter(value):
    if not value:
        return []
    if not isinstance(value, (list, tuple)):
        return _particles_converter([value])
    return [ParticleComponent.convert(element) for element in value]


def _components_converter(value):
    if not value:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [
        atmosphere_factory.convert(element, allowed_cls=AtmosphereComponent)
        for element in value
    ]


@renamed_kwargs(
    {"particle_layers": "particle_components"},
    deprecated_in="1.3.0",
    removed_in="2.0.0",
)
@define(eq=False, slots=False)
class HeterogeneousAtmosphere(AbstractHeterogeneousAtmosphere):
    """
    Heterogeneous atmosphere scene element [``heterogeneous``].

    Supports both 1D (layer-based) and 3D (fully gridded)
    configurations.
    """

    _components: list[AtmosphereComponent] = documented(
        attrs.field(
            factory=list,
            converter=_components_converter,
            validator=attrs.validators.deep_iterable(
                attrs.validators.instance_of(AtmosphereComponent)
            ),
        ),
        doc="List of atmosphere components, *i.e.* a :class:`.MolecularComponent` "
        ":class:`.ParticleComponent`. Elements may be specified as dictionaries "
        "interpreted by :data:`.atmosphere_factory`; in that case, the "
        "``type`` parameter is required. At most one "
        ":class:`.MolecularComponent` is allowed. :class:`.MergedComponent` "
        "elements are kept as they are; ``merge_policy`` applies to the other "
        "components.",
        type="list of .AtmosphereComponent",
        init_type="list of .AtmosphereComponent or dict, optional",
        default="[]",
    )

    @_components.validator
    def _components_validator(self, attribute, value):
        members = self.components

        if sum(isinstance(c, MolecularComponent) for c in members) > 1:
            raise ValueError(
                f"while validating {attribute.name}: at most one "
                "MolecularComponent is allowed"
            )

        if len({id(c) for c in members}) != len(members):
            raise ValueError(
                f"while validating {attribute.name}: a component may appear only once"
            )

        if not all(component.scale is None for component in [*value, *members]):
            raise ValueError(
                f"while validating {attribute.name}: components cannot be "
                "scaled individually"
            )

    merge_policy: MergePolicy = documented(
        attrs.field(
            factory=MergeAll,
            converter=merge_policy_converter,
            validator=attrs.validators.instance_of(MergePolicy),
            kw_only=True,
        ),
        doc="Policy merging the components which are not already part of a "
        ":class:`.MergedComponent`. Each merged component is rendered as one "
        "medium; several merged components are rendered as overlapping media. "
        "May be specified as a dictionary interpreted by "
        ":data:`.merge_policy_factory` or as a registered type ID, *e.g.* "
        '``"merge_all"``, ``"overlap_all"``, ``"merge_by_grid"``.',
        type=".MergePolicy",
        init_type=".MergePolicy or dict or str, optional",
        default=":class:`MergeAll() <.MergeAll>`",
    )

    use_phase_mis: bool = documented(
        attrs.field(default=True, converter=bool, kw_only=True),
        doc="If ``True``, multiple importance sampling is enabled for the "
        "3D phase function mixture.",
        type="bool",
        init_type="bool",
        default="True",
    )

    components_merged: list[MergedComponent] = documented(
        attrs.field(factory=list, init=False, repr=False),
        doc="Merged components generated from ``components`` by "
        ":meth:`merge_components`.",
        type="list of .MergedComponent",
    )

    def __init__(
        self, *args, molecular_atmosphere=None, particle_components=None, **kwargs
    ):
        if molecular_atmosphere is not None or particle_components is not None:
            if "components" in kwargs:
                raise TypeError(
                    "cannot pass 'components' together with "
                    "'molecular_atmosphere' or 'particle_components'"
                )

            for name, value in [
                ("molecular_atmosphere", molecular_atmosphere),
                ("particle_components", particle_components),
            ]:
                if value is not None:
                    warnings.warn(
                        DeprecatedWarning(
                            component=f"'HeterogeneousAtmosphere.{name}'",
                            deprecated_in="1.3.1",
                            removed_in="2.0.0",
                            details="Use 'components' instead.",
                        ),
                        stacklevel=3,
                    )

            # Molecular first in order to match legacy input
            components = []
            if molecular_atmosphere is not None:
                components.append(_molecular_converter(molecular_atmosphere))
            components += _particles_converter(particle_components)

            kwargs["components"] = components

        self.__attrs_init__(*args, **kwargs)

    @property
    def components(self) -> list[AtmosphereComponent]:
        """
        Atmosphere components, with the members of user-defined
        :class:`.MergedComponent` elements in place of the merged components.
        """
        return [
            member
            for component in self._components
            for member in (
                component.components
                if isinstance(component, MergedComponent)
                else [component]
            )
        ]

    @property
    def molecular_atmosphere(self) -> MolecularComponent | None:
        """
        Molecular component of this atmosphere, if any.
        """
        return next(
            (c for c in self.components if isinstance(c, MolecularComponent)),
            None,
        )

    @property
    def particle_components(self) -> list[AtmosphereComponent]:
        """
        Particle components of this atmosphere, if any.
        """
        return [c for c in self.components if not isinstance(c, MolecularComponent)]

    @property
    def is_overlapping(self) -> bool:
        """
        ``True`` if this atmosphere is rendered as overlapping media, *i.e.*
        if it has more than one merged component.
        """
        return len(self.components_merged) > 1

    @property
    def is_piecewise(self) -> bool:
        """
        ``True`` if this atmosphere is rendered with a single ``piecewise``
        medium, which requires a single merged component whose geometry is a
        :class:`.PlaneParallelGeometry` with a 1D grid and the ``nearest``
        filter type, and ``force_majorant`` unset. Piecewise media get
        optimized during traversal when used with integrators that support it
        (see :attr:`.EOVolPathIntegrator.uses_eo_medium_api`).
        """
        if self.is_overlapping:
            return False

        geometry = self.components_merged[0].geometry
        return (
            isinstance(geometry, PlaneParallelGeometry)
            and not self.force_majorant
            and geometry.grid.is_1d
            and geometry.filter_type is FilterType.NEAREST
        )

    @property
    def particle_layers(self) -> list[ParticleComponent]:
        """
        Particle components of this atmosphere.

        .. deprecated:: 1.3.0
           Use :attr:`particle_components` instead.
        """
        warnings.warn(
            DeprecatedWarning(
                component="'HeterogeneousAtmosphere.particle_layers'",
                deprecated_in="1.3.0",
                removed_in="2.0.0",
                details="Use 'particle_components' instead.",
            ),
            stacklevel=2,
        )
        return self.particle_components

    def update(self):
        # Inherit docstring
        if not self.components:
            raise ValueError("HeterogeneousAtmosphere must have at least one component")

        for i, component in enumerate(self.components):
            component.update()
            component.id = f"{self.id}_component_{i}"

        self._merged_all = MergedComponent(
            id=self.id,
            components=self.components,
            use_phase_mis=self.use_phase_mis,
        )
        self._merged_all.geometry = self.geometry
        self.merge_components()

    def merge_components(self) -> None:
        """
        Populate :attr:`components_merged`: components which are not part of a
        :class:`.MergedComponent` are merged according to
        :attr:`merge_policy`, then merged components passed by the user are
        appended. Each merged component gets its render geometry: the
        geometry of this atmosphere, with its grid replaced by the grid of the
        merged component if set.

        This method is called by :meth:`update`.
        """
        loose = [c for c in self._components if not isinstance(c, MergedComponent)]
        groups = self.merge_policy.merge(loose, self.geometry) if loose else []
        groups += [c for c in self._components if isinstance(c, MergedComponent)]

        self.components_merged = []

        for i, group in enumerate(groups):
            group.geometry = geometry_on_grid(
                self.geometry, group.grid, group.crop_to_grid
            )
            # Fail early on member grids which do not match the frame
            for member in group.components:
                geometry_on_grid(self.geometry, member.grid)

            group.id = f"{self.id}_group_{i}"
            group.use_phase_mis = self.use_phase_mis
            self.components_merged.append(group)

    def check_geometry(self) -> None:
        """
        Check that each component can be evaluated on the geometries it is
        rendered on: the render geometry of its merged component (radiative
        properties) and, if it has its own grid, that grid (phase function).

        Not called by :meth:`update`, since the geometry is final only once
        set by the experiment.

        Raises
        ------
        ValueError
            If a component cannot be evaluated on one of these geometries.
        """
        for group in self.components_merged:
            for member in group.components:
                member.check_geometry(group.geometry)
                if member.grid is not None:
                    member.check_geometry(
                        geometry_on_grid(
                            group.geometry, member.grid, member.crop_to_grid
                        )
                    )

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    @property
    def absorption_data(self) -> AbsorptionDatabase | None:
        # Inherit docstring
        if self.molecular_atmosphere is not None:
            return self.molecular_atmosphere.absorption_data
        else:
            return None

    def eval_albedo(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self._merged_all.eval_albedo(si, grid)

    def eval_sigma_t(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self._merged_all.eval_sigma_t(si, grid)

    def eval_sigma_a(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self._merged_all.eval_sigma_a(si, grid)

    def eval_sigma_s(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self._merged_all.eval_sigma_s(si, grid)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def phase(self) -> PhaseFunction:
        # Inherit docstring
        return self._merged_all.phase

    @property
    def template(self) -> dict:
        # Inherit docstring
        if not self.is_overlapping:
            return super().template

        phases = {}
        for group in self.components_merged:
            phase = group.get_phase(group.geometry)
            phases[group.phase_id] = traverse(phase)[0].data

        result = flatten(
            {
                **phases,
                self.medium_id: self._template_medium,
                self.shape_id: self._template_shape,
            }
        )
        result.update(
            {
                f"{self.shape_id}.bsdf.type": "null",
                f"{self.shape_id}.interior.type": "ref",
                f"{self.shape_id}.interior.id": self.medium_id,
            }
        )

        return result

    @property
    def _template_phase(self) -> dict:
        # Inherit docstring
        group = self.components_merged[0]
        return traverse(group.get_phase(group.geometry))[0].data

    @property
    def _template_medium(self) -> dict:
        # Inherit docstring
        if not self.is_overlapping:
            return self._build_medium_template(self.components_merged[0])

        result = {"type": "overlap", "ddis_threshold": self.ddis_threshold}

        for i, group in enumerate(self.components_merged):
            result[f"component{i}"] = {
                **self._build_medium_template(group),
                "id": group.medium_id,
                "phase": {"type": "ref", "id": group.phase_id},
            }

        return result

    @property
    def _params_phase(self) -> dict:
        # Inherit docstring
        result = {}

        for group in self.components_merged:
            if self.is_overlapping:
                phase_id, prefix = group.phase_id, f"{group.phase_id}."
            else:
                phase_id, prefix = self.phase_id, ""

            phase = group.get_phase(group.geometry)
            for uparam_key, uparam in traverse(phase)[1].data.items():
                result[f"{prefix}{uparam_key}"] = attrs.evolve(
                    uparam,
                    search=SearchSceneParameter(
                        node_type=mi.PhaseFunction,
                        node_id=phase_id,
                        parameter_relpath=uparam_key,
                    ),
                )

        return result

    @property
    def _params_medium(self) -> dict:
        # Inherit docstring
        if not self.is_overlapping:
            return self._build_medium_params(self.components_merged[0], self.medium_id)

        result = {}

        for group in self.components_merged:
            params = self._build_medium_params(group, group.medium_id)
            result.update({f"{group.medium_id}.{k}": v for k, v in params.items()})

        return result

    def _build_medium_template(self, group: MergedComponent) -> dict:
        """
        Generate a medium template from the radiative properties of ``group``,
        evaluated on its render geometry. Medium options are taken from
        this atmosphere, except ``extremum_resolution``, which ``group`` may
        override. The medium is ``piecewise`` if :attr:`is_piecewise` is
        ``True``, ``eoheterogeneous`` otherwise.

        Parameters
        ----------
        group : .MergedComponent
            Merged component rendered as the medium.

        Returns
        -------
        dict
            Medium template.
        """
        collision_coefficient_units = uck.get("collision_coefficient")
        length_units = uck.get("length")

        extremum = None
        geometry = group.geometry
        extremum_resolution = group.extremum_resolution or self.extremum_resolution

        volumes = {
            "albedo": generate_gridvolume(
                geometry,
                group.eval_albedo,
                units=ureg.dimensionless,
                extra_kwargs={"grid": geometry.grid},
            ),
            "sigma_t": generate_gridvolume(
                geometry,
                group.eval_sigma_t,
                units=collision_coefficient_units,
                extra_kwargs={"grid": geometry.grid},
            ),
        }

        medium = "piecewise" if self.is_piecewise else "eoheterogeneous"

        aabb_conf = {}
        if isinstance(geometry, SphericalShellGeometry):
            if any(r < 1 for r in extremum_resolution):
                raise ValueError(
                    "in spherical shell geometry, 'extremum_resolution' must "
                    "be >= 1 in all dimensions. got "
                    f"{extremum_resolution}."
                )
            extr_conf = {"type": "sphericalcoordsextremum"}
        elif isinstance(geometry, PlaneParallelGeometry):
            extr_conf = {"type": "gridextremum"}

            aabb_min = geometry.bbox.min.m_as(length_units)
            # The geometry bbox inserts a sub-surface padding. We want the data to be
            # evaluated on the entire geometry extent in x and y directions, but we
            # should avoid the wrap_mode affecting the z direction.
            aabb_min[2] = geometry.grid.levels[0].m_as(length_units)
            aabb_conf = {
                "aabb_min": aabb_min,
                "aabb_max": geometry.bbox.max.m_as(length_units),
            }

        if extremum_resolution != (1, 1, 1):
            extremum = {
                **extr_conf,
                "resolution": extremum_resolution,
            }

        # Create medium dictionary
        result = {
            "type": medium,
            "has_spectral_extinction": False,
            "rr_threshold": self.rr_threshold,
            "use_rrt": self.use_rrt,
            "ddis_threshold": self.ddis_threshold,
            **aabb_conf,
            **volumes,
            # Note: "phase" is deliberately unset, this is left to the
            # Atmosphere.template property
        }

        if self.scale is not None:
            result["scale"] = self.scale

        if extremum is not None:
            result["extremum"] = extremum

        return result

    def _build_medium_params(
        self, group: MergedComponent, medium_id: str
    ) -> dict[str, SceneParameter]:
        """
        Generate the parameter update map of a medium created with
        :meth:`_build_medium_template`.

        Parameters
        ----------
        group : .MergedComponent
            Merged component rendered as the medium.

        medium_id : str
            ID of the target medium in the Mitsuba scene tree.

        Returns
        -------
        dict
            Parameter update map.
        """
        collision_coefficient_units = uck.get("collision_coefficient")
        geometry = group.geometry

        if isinstance(geometry, PlaneParallelGeometry):
            albedo_key = "albedo.data"
            sigma_t_key = "sigma_t.data"
        elif isinstance(geometry, SphericalShellGeometry):
            albedo_key = "albedo.volume.data"
            sigma_t_key = "sigma_t.volume.data"
        else:
            raise TypeError(
                f"unhandled scene geometry type '{type(geometry).__name__}'"
            )

        albedo_search = SearchSceneParameter(
            node_type=mi.Medium, node_id=medium_id, parameter_relpath=albedo_key
        )
        sigma_t_search = SearchSceneParameter(
            node_type=mi.Medium, node_id=medium_id, parameter_relpath=sigma_t_key
        )
        albedo_grid = generate_gridvolume(
            geometry,
            group.eval_albedo,
            units=ureg.dimensionless,
            search=albedo_search,
            flag=KernelSceneParameterFlags.SPECTRAL,
            extra_kwargs={"grid": geometry.grid},
        )
        sigma_t_grid = generate_gridvolume(
            geometry,
            group.eval_sigma_t,
            units=collision_coefficient_units,
            search=sigma_t_search,
            flag=KernelSceneParameterFlags.SPECTRAL,
            extra_kwargs={"grid": geometry.grid},
        )

        return {
            albedo_key: albedo_grid,
            sigma_t_key: sigma_t_grid,
        }
