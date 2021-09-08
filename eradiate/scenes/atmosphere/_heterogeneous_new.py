"""
Heterogeneous atmospheres.
"""
from typing import Dict, List, Optional, Tuple, Union

import attr
import numpy as np
import pint
import xarray as xr

from ._core import Atmosphere, atmosphere_factory, write_binary_grid3d
from ._molecules import MolecularAtmosphere
from ._particles import ParticleLayer
from ...attrs import AUTO, documented, parse_docs
from ...contexts import KernelDictContext
from ...units import to_quantity
from ...units import unit_registry as ureg


@atmosphere_factory.register(type_id="heterogeneous_new")
@parse_docs
@attr.s
class HeterogeneousNewAtmosphere(Atmosphere):
    """
    Heterogeneous atmosphere scene element [``heterogeneous_new``].
    """

    molecular_atmosphere: Optional[MolecularAtmosphere] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(atmosphere_factory.convert),
            validator=attr.validators.optional(
                attr.validators.instance_of(MolecularAtmosphere)
            ),
        ),
        doc="Molecular atmosphere.",
        type=":class:`.MolecularAtmosphere` or None",
        default="None",
    )

    particle_layers: List[ParticleLayer] = documented(
        attr.ib(
            factory=list,
            converter=lambda value: [
                ParticleLayer.convert(element) for element in value
            ],
            validator=attr.validators.deep_iterable(
                attr.validators.instance_of(ParticleLayer)
            ),
        ),
        doc="Particle layers.",
        type="list[:class:`.ParticleLayer`]",
        default="[]",
    )

    @molecular_atmosphere.validator
    @particle_layers.validator
    def _validate_ids(instance, attribute, value):
        ids = set()

        for component in instance.components:
            if not component.id in ids:
                ids.add(component.id)
            else:
                raise ValueError(
                    f"while validating {attribute.name}: found duplicate component ID '{component.id}'; "
                    f"all components (molecular atmosphere and particle layers) must have different IDs"
                )

    @molecular_atmosphere.validator
    @particle_layers.validator
    def _validate_widths(instance, attribute, value):
        if not all([component.width is AUTO for component in instance.components]):
            raise ValueError(
                "all components must have their 'width' attribute set to 'AUTO'"
            )

    @property
    def components(self) -> List[Union[MolecularAtmosphere, ParticleLayer]]:
        result: list = [self.molecular_atmosphere] if self.molecular_atmosphere else []
        result.extend(self.particle_layers)
        return result

    # --------------------------------------------------------------------------
    #              Spatial extension and thermophysical properties
    # --------------------------------------------------------------------------

    @property
    def bottom(self) -> pint.Quantity:
        bottoms = [component.bottom for component in self.components]
        if bottoms:
            return min(bottoms)
        else:
            return 0.0 * ureg.km

    @property
    def top(self) -> pint.Quantity:
        tops = [component.top for component in self.components]
        if tops:
            return max(tops)
        else:
            return 10.0 * ureg.km

    def eval_width(self, ctx: KernelDictContext) -> pint.Quantity:
        widths = [component.eval_width(ctx=ctx) for component in self.components]

        if widths:
            return max(widths)
        else:
            return 1000.0 * ureg.km

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def kernel_phase(self, ctx: KernelDictContext) -> Dict:
        """
        .. note::
           One phase plugin specification per component (molecular atmosphere and
           particle layers) is generated.
           For example, if there are one molecular atmosphere and two particle
           layers, the returned kernel dictionary has three entries.
        """
        phases = {}

        if self.molecular_atmosphere is not None:
            phases.update(self.molecular_atmosphere.kernel_phase(ctx=ctx))

            # TODO: add support for overlapping layers
            if self.particle_layers is not None:
                for particle_layer in self.particle_layers:
                    # blend molecular atmosphere with particle layer
                    _, ratios = blend_radprops(
                        background=self.molecular_atmosphere.eval_radprops(
                            spectral_ctx=ctx.spectral_ctx
                        ),
                        foreground=particle_layer.eval_radprops(
                            spectral_ctx=ctx.spectral_ctx
                        ),
                    )
                    # write the weight volume data file
                    write_binary_grid3d(
                        filename=particle_layer.weight_file,
                        values=ratios.values[np.newaxis, np.newaxis, ...],
                    )
                    phases.update(
                        {
                            f"phase_{particle_layer.id}": {
                                "type": "blendphase",
                                "weight": {
                                    "type": "gridvolume",
                                    "filename": str(particle_layer.weight_file),
                                    "to_world": particle_layer._gridvolume_to_world_trafo(
                                        ctx=ctx
                                    ),
                                },
                                **self.molecular_atmosphere.kernel_phase(ctx=ctx),
                                **particle_layer.kernel_phase(ctx=ctx),
                            }
                        }
                    )
        else:
            if self.particle_layers is not None:
                # TODO: add support for overlapping layers
                for particle_layer in self.particle_layers:
                    phases.update(**particle_layer.kernel_phase(ctx=ctx))

        return phases

    def kernel_media(self, ctx: KernelDictContext) -> Dict:
        """
        .. note::
           One media plugin specification per component (molecular atmosphere and
           particle layers) is generated.
           For example, if there are one molecular atmosphere and two particle
           layers, the returned kernel dictionary has three entries.
        """
        media = {}

        if self.molecular_atmosphere is not None:
            # override component widths
            ctx = ctx.evolve(override_scene_width=self.eval_width(ctx=ctx))
            media.update(self.molecular_atmosphere.kernel_media(ctx=ctx))

            if not ctx.ref:
                phases = self.kernel_phase(ctx=ctx)

            # TODO: add support for overlapping layers
            if self.particle_layers is not None:
                for particle_layer in self.particle_layers:
                    # blend molecular atmosphere with particle layer
                    radprops, _ = blend_radprops(
                        background=self.molecular_atmosphere.eval_radprops(
                            spectral_ctx=ctx.spectral_ctx
                        ),
                        foreground=particle_layer.eval_radprops(
                            spectral_ctx=ctx.spectral_ctx
                        ),
                    )
                    # write the albedo volume data file
                    write_binary_grid3d(
                        filename=particle_layer.albedo_file,
                        values=radprops.albedo.values[np.newaxis, np.newaxis, ...],
                    )
                    # write the sigma_t volume data file
                    write_binary_grid3d(
                        filename=particle_layer.sigma_t_file,
                        values=radprops.sigma_t.values[np.newaxis, np.newaxis, ...],
                    )
                    trafo = particle_layer._gridvolume_to_world_trafo(ctx=ctx)
                    if ctx.ref:
                        phase = {"type": "ref", "id": f"phase_{particle_layer.id}"}
                    else:
                        phase = phases[f"phase_{particle_layer.id}"]
                    media.update(
                        {
                            f"medium_{particle_layer.id}": {
                                "type": "heterogeneous",
                                "phase": phase,
                                "sigma_t": {
                                    "type": "gridvolume",
                                    "filename": str(particle_layer.sigma_t_file),
                                    "to_world": trafo,
                                },
                                "albedo": {
                                    "type": "gridvolume",
                                    "filename": str(particle_layer.albedo_file),
                                    "to_world": trafo,
                                },
                            }
                        }
                    )
        else:
            if self.particle_layers is not None:
                # TODO: add support for overlapping layers
                for particle_layer in self.particle_layers:
                    media.update(**particle_layer.kernel_media(ctx=ctx))

        return media

    def kernel_shapes(self, ctx: KernelDictContext) -> Dict:
        """
        .. note::
           One shape plugin specification per component (molecular atmosphere and
           particle layers) is generated.
           For example, if there are one molecular atmosphere and two particle
           layers, the returned kernel dictionary has three entries.
        """
        shapes = {}

        # override componments' widths
        ctx.override_scene_width = self.eval_width(ctx=ctx)

        if self.molecular_atmosphere is not None:
            shapes.update(self.molecular_atmosphere.kernel_shapes(ctx=ctx))

        if self.particle_layers is not None:
            for particle_layer in self.particle_layers:
                shapes.update(particle_layer.kernel_shapes(ctx=ctx))

        return shapes


def blend_radprops(
    background: xr.Dataset, foreground: xr.Dataset
) -> Tuple[xr.Dataset, xr.DataArray]:
    """
    Blend radiative properties of two participating media.

    .. note::
       Assumes 'z_layer' is in same units in 'molecules' and in 'particles'.

    Returns → Tuple[:class:`~xarray.Dataset`, :class:`~xarray.DataArray`]:
        Radiative properties resulting from the blending of the two participating
        media, in the altitude range of the foreground participating medium.
    """
    background = interpolate_radprops(
        radprops=background, new_z_layer=to_quantity(foreground.z_layer)
    )
    blend_sigma_t = background.sigma_t + foreground.sigma_t
    background_ratio = xr.where(
        blend_sigma_t != 0.0, background.sigma_t / blend_sigma_t, 0.5
    )  # broadcast 0.5
    foreground_ratio = xr.where(
        blend_sigma_t != 0.0, foreground.sigma_t / blend_sigma_t, 0.5
    )  # broadcast 0.5
    blend_albedo = (
        background.albedo * background_ratio + foreground.albedo * foreground_ratio
    )
    blended_radprops = xr.Dataset(
        data_vars={"sigma_t": blend_sigma_t, "albedo": blend_albedo}
    )
    return (blended_radprops, foreground_ratio)


def interpolate_radprops(
    radprops: xr.Dataset, new_z_layer: pint.Quantity
) -> xr.Dataset:
    """
    Interpolate a radiative property data set onto a new level altitude grid.

    Out of bounds values are replaced with zeros.

    Parameter ``radprops`` (:class:`~xarray.Dataset`):
        Radiative property data set.

    Parameter ``new_z_layer`` (:class:`~pint.Quantity`):
        Layer altitude grid to interpolate onto.

    Returns → :class:`~xarray.Dataset`:
        Interpolated radiative propery data set.
    """
    mask = (new_z_layer >= to_quantity(radprops.z_level).min()) & (
        new_z_layer <= to_quantity(radprops.z_level).max()
    )
    masked = new_z_layer[mask]  # altitudes that fall within the bounds of radprops

    # interpolate within radprops altitude bounds (safe)
    interpolated_safe = radprops.interp(
        z_layer=masked.m_as(radprops.z_layer.units),
        kwargs=dict(fill_value="extrapolate"),  # handle floating point arithmetic issue
        method="nearest",  # radiative properties are assumed uniform in altitude layers
    )

    # interpolate over the full range
    interpolated = interpolated_safe.interp(
        z_layer=new_z_layer.m_as(radprops.z_layer.units),
        kwargs=dict(fill_value=0.0),
        method="nearest",
    )

    return interpolated


def overlapping(
    particle_layers: List[ParticleLayer], particle_layer: ParticleLayer
) -> List[ParticleLayer]:
    """
    Return the particle layers that are overlapping a given particle layer.

    Parameter ``particle_layers`` (list[:class:`.ParticleLayer`]):
        List of particle layers to check for overlap.

    Parameter ``particle_layer`` (:class:`.ParticleLayer`):
        The given particle layer.

    Returns → list[:class:`.ParticleLayer`]:
        Overlapping particle layers.
    """
    return [
        layer
        for layer in particle_layers
        if particle_layer.top >= layer.top > particle_layer.bottom
        or particle_layer.top > layer.bottom >= particle_layer.bottom
    ]
