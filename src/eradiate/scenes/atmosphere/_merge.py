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

from ._core import AtmosphereComponent, atmosphere_factory, geometry_on_grid
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

    Sums the radiative properties of its members on its own grid and
    blends their phase functions. A :class:`.HeterogeneousAtmosphere` renders
    each merged component as one medium; several merged components are
    rendered as overlapping media.

    The ``geometry`` field holds the render geometry: the parent
    :class:`.HeterogeneousAtmosphere` sets it on every update to its own
    geometry, with its grid replaced by ``grid`` if set.
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
        "evaluated on the grid of the merged component; phase functions of "
        "members are built on their own grid if set, on the geometry of the "
        "merged component otherwise.",
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

    @property
    def default_geometry(self) -> SceneGeometry:
        # Inherit docstring
        # Plain property: the render geometry is reassigned on every update
        if self.geometry is not None:
            return self.geometry
        return super().default_geometry

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
        grid = self.default_geometry.grid if grid is None else grid
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
        grid = self.default_geometry.grid if grid is None else grid
        return self._eval_sigma_s_impl(si, grid).sum(axis=0)

    # --------------------------------------------------------------------------
    #                       Phase function
    # --------------------------------------------------------------------------

    def get_phase(self, geometry: SceneGeometry | None = None) -> PhaseFunction:
        # Inherit docstring
        geometry = self.default_geometry if geometry is None else geometry
        phases = [
            component.get_phase(
                geometry_on_grid(geometry, component.grid, component.crop_to_grid)
            )
            for component in self.components
        ]

        if len(phases) == 1:
            return phases[0]

        weights = []
        sigma_units = ucc.get("collision_coefficient")

        for i in range(len(phases)):

            def eval_sigma_s(si: SpectralIndex, n_component: int = i) -> np.ndarray:
                sigma_s = self._eval_sigma_s_impl(si, geometry.grid)
                return sigma_s[n_component].m_as(sigma_units)

            weights.append(eval_sigma_s)

        return MultiPhaseFunction(
            components=phases,
            weights=weights,
            geometry=geometry,
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


def _same_grid(a: SpatialGrid | None, b: SpatialGrid | None) -> bool:
    """
    ``True`` if both grids have the same type and the same edges. Grids are
    compared by value since they do not implement equality. ``None`` only
    matches ``None``.
    """
    if a is b:
        return True

    if a is None or b is None or type(a) is not type(b):
        return False

    return all(
        np.array_equal(
            getattr(a, name).magnitude, getattr(b, name).m_as(getattr(a, name).units)
        )
        for name in ("x_edges", "y_edges", "z_edges")
    )


def _merge(
    components: list[AtmosphereComponent],
    grid: SpatialGrid | None = None,
    crop_to_grid: str = "",
) -> MergedComponent:
    """
    Merge components on ``grid``. The merged component uses the extremum
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
        grid=grid,
        crop_to_grid=crop_to_grid,
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
            Components to merge.

        geometry : .SceneGeometry
            Geometry of the parent atmosphere, which merged components are
            rendered on with their own grid.

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
        return [_merge(components)]


@define
class OverlapAll(MergePolicy):
    """
    Render each component as its own medium, evaluated on its own grid
    [``overlap_all``].
    """

    def merge(self, components, geometry):
        # Inherit docstring
        return [
            _merge([component], component.grid, component.crop_to_grid)
            for component in components
        ]


@define
class MergeByGrid(MergePolicy):
    """
    Merge components which share the same grid and ``crop_to_grid`` value
    [``merge_by_grid``]. Components without a grid are merged together.
    """

    def merge(self, components, geometry):
        # Inherit docstring
        groups: list[tuple[SpatialGrid | None, str, list[AtmosphereComponent]]] = []

        for component in components:
            group = next(
                (
                    members
                    for grid, crop_to_grid, members in groups
                    if _same_grid(grid, component.grid)
                    and crop_to_grid == component.crop_to_grid
                ),
                None,
            )
            if group is None:
                groups.append((component.grid, component.crop_to_grid, [component]))
            else:
                group.append(component)

        return [_merge(members, grid, crop) for grid, crop, members in groups]


def merge_policy_converter(value: MergePolicy | dict | str) -> MergePolicy:
    """
    Convert a merge policy specification: a :class:`.MergePolicy` instance, a
    dictionary interpreted by :data:`.merge_policy_factory`, or a type ID
    registered to :data:`.merge_policy_factory`.
    """
    if isinstance(value, str):
        value = {"type": value}
    return merge_policy_factory.convert(value, allowed_cls=MergePolicy)
