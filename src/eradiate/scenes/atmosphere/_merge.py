"""
Merged atmosphere components and merge policies.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import attrs
import numpy as np
import pint

from ._core import AtmosphereComponent, atmosphere_factory
from ..geometry import SceneGeometry
from ..phase import MultiPhaseFunction, PhaseFunction
from ..._factory import Factory
from ...attrs import define, documented
from ...grid import SpatialGrid
from ...spectral.index import SpectralIndex
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...util.misc import cache_by_id

if TYPE_CHECKING:
    from axsdb import AbsorptionDatabase


def _components_converter(value):
    return [
        atmosphere_factory.convert(element, allowed_cls=AtmosphereComponent)
        for element in value
    ]


@define(eq=False, slots=False)
class MergedComponent(AtmosphereComponent):
    """
    Merged atmosphere component [``merged``].

    Sums the radiative properties of its members on its own geometry and
    blends their phase functions. A :class:`.HeterogeneousAtmosphere` renders
    each merged component as one medium; several merged components are
    rendered as overlapping media.
    """

    components: list[AtmosphereComponent] = documented(
        attrs.field(
            converter=_components_converter,
            validator=attrs.validators.deep_iterable(
                attrs.validators.instance_of(AtmosphereComponent)
            ),
            kw_only=True,
        ),
        doc="Merged atmosphere components. Elements may be specified as "
        "dictionaries interpreted by :data:`.atmosphere_factory`. Merged "
        "components cannot be nested. Radiative properties of members are "
        "evaluated on the grid of the merged component; members without a "
        "geometry of their own get the geometry of the merged component.",
        type="list of .AtmosphereComponent",
        init_type="list of .AtmosphereComponent or dict",
    )

    @components.validator
    def _components_validator(self, attribute, value):
        if not value:
            raise ValueError(
                f"while validating {attribute.name}: at least one component is required"
            )

        if any(isinstance(component, MergedComponent) for component in value):
            raise ValueError(
                f"while validating {attribute.name}: merged components cannot be nested"
            )

    use_phase_mis: bool = documented(
        attrs.field(default=True, converter=bool, kw_only=True),
        doc="If ``True``, multiple importance sampling is enabled for the "
        "3D phase function mixture. Overridden by the parent "
        ":class:`.HeterogeneousAtmosphere`.",
        type="bool",
        init_type="bool",
        default="True",
    )

    @property
    def medium_id(self) -> str:
        """
        Returns
        -------
        str
            ID of the medium generated for this merged component.
        """
        return f"medium_{self.id}"

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    @property
    def absorption_data(self) -> AbsorptionDatabase | None:
        # Inherit docstring
        return next(
            (
                component.absorption_data
                for component in self.components
                if component.absorption_data is not None
            ),
            None,
        )

    @cache_by_id
    def _eval_sigma_t_impl(self, si: SpectralIndex, grid: SpatialGrid) -> pint.Quantity:
        sigma_units = ucc.get("collision_coefficient")
        result = [
            np.broadcast_to(
                component.eval_sigma_t(si, grid).m_as(sigma_units), grid.shape
            )
            for component in self.components
        ]
        return np.stack(result) * sigma_units

    @cache_by_id
    def _eval_sigma_s_impl(self, si: SpectralIndex, grid: SpatialGrid) -> pint.Quantity:
        sigma_units = ucc.get("collision_coefficient")
        result = [
            np.broadcast_to(
                component.eval_sigma_s(si, grid).m_as(sigma_units), grid.shape
            )
            for component in self.components
        ]
        return np.stack(result) * sigma_units

    def eval_albedo(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        units = ucc.get("collision_coefficient")
        sigma_s = self.eval_sigma_s(si, grid).m_as(units)
        sigma_t = self.eval_sigma_t(si, grid).m_as(units)
        albedo = np.zeros_like(sigma_s)
        np.divide(sigma_s, sigma_t, where=sigma_t != 0.0, out=albedo)

        return albedo * ureg.dimensionless

    def eval_sigma_t(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        grid = self.geometry.grid if grid is None else grid
        return self._eval_sigma_t_impl(si, grid).sum(axis=0)

    def eval_sigma_a(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self.eval_sigma_t(si, grid) - self.eval_sigma_s(si, grid)

    def eval_sigma_s(
        self, si: SpectralIndex, grid: SpatialGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        grid = self.geometry.grid if grid is None else grid
        return self._eval_sigma_s_impl(si, grid).sum(axis=0)

    # --------------------------------------------------------------------------
    #                       Phase function
    # --------------------------------------------------------------------------

    @property
    def phase(self) -> PhaseFunction:
        # Inherit docstring
        if len(self.components) == 1:
            return self.components[0].phase

        components, weights = [], []
        sigma_units = ucc.get("collision_coefficient")

        for i, component in enumerate(self.components):
            components.append(component.phase)

            def eval_sigma_s(si: SpectralIndex, n_component: int = i) -> np.ndarray:
                sigma_s = self._eval_sigma_s_impl(si, self.geometry.grid)
                return sigma_s[n_component].m_as(sigma_units)

            weights.append(eval_sigma_s)

        return MultiPhaseFunction(
            components=components,
            weights=weights,
            geometry=self.geometry,
            use_mis=self.use_phase_mis,
        )


# ------------------------------------------------------------------------------
#                               Merge policies
# ------------------------------------------------------------------------------

merge_policy_factory = Factory()
merge_policy_factory.register_lazy_batch(
    [
        ("MergeAll", "merge_all", {}),
        ("OverlapAll", "overlap_all", {}),
        ("MergeByGrid", "merge_by_grid", {}),
    ],
    cls_prefix="eradiate.scenes.atmosphere._merge",
)


def _same_geometry(a: SceneGeometry, b: SceneGeometry) -> bool:
    """
    ``True`` if both geometries have the same type, the same grid edges and
    the same values for all other fields. Grids are compared by value since
    they do not implement equality.
    """
    if a is b:
        return True

    if type(a) is not type(b) or type(a.grid) is not type(b.grid):
        return False

    for name in ("x_edges", "y_edges", "z_edges"):
        edges_a, edges_b = getattr(a.grid, name), getattr(b.grid, name)
        if not np.array_equal(edges_a.magnitude, edges_b.m_as(edges_a.units)):
            return False

    return all(
        np.all(getattr(a, field.name) == getattr(b, field.name))
        for field in attrs.fields(type(a))
        if field.name != "grid"
    )


def _merge(
    components: list[AtmosphereComponent], geometry: SceneGeometry
) -> MergedComponent:
    """
    Merge components on ``geometry``. The merged component uses the extremum
    resolution shared by all components; if they differ, it falls back to the
    resolution of the parent atmosphere.
    """
    resolutions = {component.extremum_resolution for component in components}

    if len(resolutions) == 1:
        extremum_resolution = resolutions.pop()
    else:
        extremum_resolution = None
        warnings.warn(
            "merged components set different extremum resolutions "
            f"{resolutions - {None}}; the resolution of the parent atmosphere "
            "is used instead",
            UserWarning,
            stacklevel=2,
        )

    return MergedComponent(
        components=components,
        geometry=geometry,
        extremum_resolution=extremum_resolution,
    )


@define
class MergePolicy(ABC):
    """
    Abstract base class for merge policies.

    A merge policy splits the components of a :class:`.HeterogeneousAtmosphere`
    into :class:`.MergedComponent` instances, each rendered as one medium.
    Several merged components are rendered as overlapping media.
    """

    @abstractmethod
    def merge(
        self, components: list[AtmosphereComponent], geometry: SceneGeometry
    ) -> list[MergedComponent]:
        """
        Merge components.

        Parameters
        ----------
        components : list of .AtmosphereComponent
            Components to merge, each set to its own geometry or, if it has
            none, to the geometry of the parent atmosphere.

        geometry : .SceneGeometry
            Geometry of the parent atmosphere.

        Returns
        -------
        list of .MergedComponent
        """


@define
class MergeAll(MergePolicy):
    """
    Merge all components into a single medium, evaluated on the geometry of
    the parent atmosphere [``merge_all``].
    """

    def merge(self, components, geometry):
        # Inherit docstring
        return [_merge(components, geometry)]


@define
class OverlapAll(MergePolicy):
    """
    Render each component as its own medium, evaluated on its own geometry
    [``overlap_all``].
    """

    def merge(self, components, geometry):
        # Inherit docstring
        return [_merge([component], component.geometry) for component in components]


@define
class MergeByGrid(MergePolicy):
    """
    Merge components which share the same geometry, including its grid
    [``merge_by_grid``].
    """

    def merge(self, components, geometry):
        # Inherit docstring
        groups: list[list[AtmosphereComponent]] = []

        for component in components:
            group = next(
                (
                    g
                    for g in groups
                    if _same_geometry(g[0].geometry, component.geometry)
                ),
                None,
            )
            if group is None:
                groups.append([component])
            else:
                group.append(component)

        return [_merge(group, group[0].geometry) for group in groups]


def merge_policy_converter(value: MergePolicy | dict | str) -> MergePolicy:
    """
    Convert a merge policy specification: a :class:`.MergePolicy` instance, a
    dictionary interpreted by :data:`.merge_policy_factory`, or a type ID
    registered to :data:`.merge_policy_factory`.
    """
    if isinstance(value, str):
        value = {"type": value}
    return merge_policy_factory.convert(value, allowed_cls=MergePolicy)
