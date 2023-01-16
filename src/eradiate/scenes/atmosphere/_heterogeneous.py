"""
Heterogeneous atmospheres.
"""
from __future__ import annotations

import typing as t
from collections import abc as cabc
from functools import lru_cache

import attrs
import numpy as np
import pint

from ._core import AbstractHeterogeneousAtmosphere, atmosphere_factory
from ._molecular_atmosphere import MolecularAtmosphere
from ._particle_layer import ParticleLayer
from ..core import BoundingBox, traverse
from ..phase import BlendPhaseFunction, PhaseFunctionNode
from ..shapes import CuboidShape, SphereShape
from ...attrs import documented, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...radprops._core import ZGrid
from ...units import unit_context_config as ucc


def _molecular_converter(value):
    if isinstance(value, cabc.MutableMapping) and ("type" not in value):
        value["type"] = "molecular"
    return atmosphere_factory.convert(value, allowed_cls=MolecularAtmosphere)


def _particle_layer_converter(value):
    if not value:
        return []

    if not isinstance(value, (list, tuple)):
        return _particle_layer_converter([value])

    else:
        result = []

        for element in value:
            if isinstance(element, cabc.MutableMapping) and ("type" not in element):
                element["type"] = "particle_layer"
            result.append(
                atmosphere_factory.convert(element, allowed_cls=ParticleLayer)
            )

        return result


@parse_docs
@attrs.define(eq=False, slots=False)
class HeterogeneousAtmosphere(AbstractHeterogeneousAtmosphere):
    """
    Heterogeneous atmosphere scene element [``heterogeneous``].
    """

    molecular_atmosphere: t.Optional[MolecularAtmosphere] = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(_molecular_converter),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(MolecularAtmosphere)
            ),
        ),
        doc="Molecular atmosphere. May be specified as a dictionary interpreted "
        'by :data:`.atmosphere_factory`; in that case, the ``"type"`` parameter '
        'may be omitted and will automatically be set to ``"molecular"``.',
        type=":class:`.MolecularAtmosphere` or None",
        init_type=":class:`.MolecularAtmosphere` or dict, optional",
        default="None",
    )

    @molecular_atmosphere.validator
    def _molecular_atmosphere_validator(self, attribute, value):
        if value is None:
            return

        if value.scale is not None:
            raise ValueError(
                f"while validating {attribute.name}: components cannot be "
                "scaled individually"
            )

    particle_layers: t.List[ParticleLayer] = documented(
        attrs.field(
            factory=list,
            converter=_particle_layer_converter,
            validator=attrs.validators.deep_iterable(
                attrs.validators.instance_of(ParticleLayer)
            ),
        ),
        doc="List of particle layers. Elements may be specified as "
        "dictionaries interpreted by :data:`.atmosphere_factory`; in that "
        "case, the ``type`` parameter may be omitted and will automatically "
        'be set to ``"particle_layer"``.',
        type="list of :class:`.ParticleLayer`",
        init_type="list of :class:`.ParticleLayer`, optional",
        default="[]",
    )

    @particle_layers.validator
    def _particle_layers_validator(self, attribute, value):
        if not all(component.scale is None for component in value):
            raise ValueError(
                f"while validating {attribute.name}: components cannot be "
                "scaled individually"
            )

    #: A high-resolution layer altitude mesh to interpolate the components'
    #: radiative properties on, before computing the total radiative
    #: properties. This is an internal field that is automatically set by
    #: the :meth:`update` method.
    _zgrid: t.Optional[ZGrid] = attrs.field(
        default=None, converter=ZGrid, init=False, repr=False
    )

    @property
    def components(self) -> t.List[t.Union[MolecularAtmosphere, ParticleLayer]]:
        result = [self.molecular_atmosphere] if self.molecular_atmosphere else []
        result.extend(self.particle_layers)
        return result

    def update(self):
        # Force component IDs and geometry
        for i, component in enumerate(self.components):
            component.id = f"{self.id}_component_{i}"
            component.geometry = self.geometry
            component.update()

        # Set altitude grid
        # TODO: Add n_layers parameter to control the number of layers
        z_level = np.linspace(self.bottom, self.top, 11)
        self._zgrid = ZGrid(0.5 * (z_level[1:] + z_level[:-1]))

        if not self.components:
            raise ValueError("HeterogeneousAtmosphere must have at least one component")

        super().update()

    # --------------------------------------------------------------------------
    #              Spatial extension and thermophysical properties
    # --------------------------------------------------------------------------

    @property
    def zgrid(self) -> ZGrid:
        return self._zgrid

    @property
    def bottom(self) -> pint.Quantity:
        return min([component.bottom for component in self.components])

    @property
    def top(self) -> pint.Quantity:
        return max([component.top for component in self.components])

    def eval_mfp(self, ctx: KernelDictContext) -> pint.Quantity:
        mfp = [component.eval_mfp(ctx=ctx) for component in self.components]
        return max(mfp)

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def eval_albedo(
        self, sctx: SpectralContext, zgrid: t.Optional[ZGrid] = None
    ) -> pint.Quantity:
        return self.eval_sigma_s(sctx, zgrid) / self.eval_sigma_t(sctx, zgrid)

    @lru_cache(maxsize=1)
    def _eval_sigma_t_impl(self, sctx: SpectralContext, zgrid: ZGrid) -> pint.Quantity:
        result = np.zeros((len(self.components), len(zgrid.values)))
        sigma_units = ucc.get("collision_coefficient")

        # Retrieve scattering coefficient and corresponding altitude grid for
        # current component, interpolate collision coefficient on fine grid
        # TODO: Rewrite after updating component APIs with new zgrid parameter
        for i, component in enumerate(self.components):
            result[i] = np.interp(
                zgrid.values,
                component.zgrid.values,
                component.eval_sigma_t(sctx).m_as(sigma_units),
                left=0.0,
                right=0.0,
            )

        return result * sigma_units

    def eval_sigma_t(
        self, sctx: SpectralContext, zgrid: t.Optional[ZGrid] = None
    ) -> pint.Quantity:
        return self._eval_sigma_t_impl(sctx).sum(axis=0)

    def eval_sigma_a(
        self, sctx: SpectralContext, zgrid: t.Optional[ZGrid] = None
    ) -> pint.Quantity:
        return self._eval_sigma_t(sctx) - self._eval_sigma_s(sctx)

    @lru_cache(maxsize=1)
    def _eval_sigma_s_impl(self, sctx: SpectralContext, zgrid: ZGrid) -> pint.Quantity:
        result = np.zeros((len(self.components), len(zgrid.values)))
        sigma_units = ucc.get("collision_coefficient")

        # Retrieve scattering coefficient and corresponding altitude grid for
        # current component, interpolate collision coefficient on fine grid
        # TODO: Rewrite after updating component APIs with new zgrid parameter
        for i, component in enumerate(self.components):
            result[i] = np.interp(
                zgrid.values,
                component.zgrid.values,
                component.eval_sigma_s(sctx).m_as(sigma_units),
                left=0.0,
                right=0.0,
            )

        return result * sigma_units

    def _eval_sigma_s_component(
        self, sctx: SpectralContext, zgrid: ZGrid, n_component: int
    ) -> pint.Quantity:
        return self._eval_sigma_s_impl(sctx, zgrid)[n_component]

    def eval_sigma_s(
        self, sctx: SpectralContext, zgrid: t.Optional[ZGrid] = None
    ) -> pint.Quantity:
        return self._eval_sigma_s_impl(sctx, zgrid).sum(axis=0)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def _bbox(self) -> BoundingBox:
        shape = self._shape
        length_units = ucc.get("length")

        if isinstance(shape, CuboidShape):
            # In this case, the bounding box corresponds to the corners of
            # the cuboid
            min_x, min_y = (shape.center[0:2] - shape.edges[0:2] * 0.5).m_as(
                length_units
            )
            min_z = self.bottom.m_as(length_units)

            max_x, max_y = (shape.center[0:2] + shape.edges[0:2] * 0.5).m_as(
                length_units
            )
            max_z = self.top.m_as(length_units)

            return BoundingBox(
                [min_x, min_y, min_z] * length_units,
                [max_x, max_y, max_z] * length_units,
            )

        elif isinstance(shape, SphereShape):
            # In this case, the bounding box is the cuboid that contains the
            # sphere
            r = shape.radius.m_as(length_units)

            return BoundingBox(
                [-r, -r, -r] * length_units,
                [r, r, r] * length_units,
            )

        else:
            raise NotImplementedError

    @property
    def phase(self) -> PhaseFunctionNode:
        if len(self.components) == 1:
            return self.components[0].phase

        else:
            components, weights = [], []
            sigma_units = ucc.get("collision_coefficient")

            for i, component in enumerate(self.components):
                components.append(component.phase)

                def eval_sigma_s(
                    sctx: SpectralContext, n_component: int = i
                ) -> np.ndarray:
                    return self._eval_sigma_s_component(sctx, n_component).m_as(
                        sigma_units
                    )

                weights.append(eval_sigma_s)

            return BlendPhaseFunction(
                components=components, weights=weights, bbox=self._bbox
            )

    @property
    def _template_phase(self) -> dict:
        template, _ = traverse(self.phase)
        return template.data
