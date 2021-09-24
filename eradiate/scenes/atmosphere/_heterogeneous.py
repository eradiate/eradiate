"""
Heterogeneous atmospheres.
"""
import typing as t
from collections import abc as cabc

import attr
import pint
import xarray as xr

from ._core import AbstractHeterogeneousAtmosphere, atmosphere_factory
from ._molecules import MolecularAtmosphere
from ._particles import ParticleLayer
from ...attrs import AUTO, documented, parse_docs
from ...contexts import KernelDictContext
from ...units import to_quantity
from ...units import unit_registry as ureg


def _heterogeneous_atmosphere_molecular_converter(value):
    if isinstance(value, cabc.MutableMapping) and ("type" not in value):
        value["type"] = "molecular"
    return atmosphere_factory.convert(value, allowed_cls=MolecularAtmosphere)


def _heterogeneous_atmosphere_particle_converter(value):
    if not value:
        return []

    if not isinstance(value, (list, tuple)):
        return _heterogeneous_atmosphere_particle_converter([value])

    else:
        result = []

        for element in value:
            if isinstance(element, cabc.MutableMapping) and ("type" not in element):
                element["type"] = "particle_layer"
            result.append(
                atmosphere_factory.convert(element, allowed_cls=ParticleLayer)
            )

        return result


@atmosphere_factory.register(type_id="heterogeneous")
@parse_docs
@attr.s
class HeterogeneousAtmosphere(AbstractHeterogeneousAtmosphere):
    """
    Heterogeneous atmosphere scene element [``heterogeneous``].
    """

    molecular_atmosphere: t.Optional[MolecularAtmosphere] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(
                _heterogeneous_atmosphere_molecular_converter
            ),
            validator=attr.validators.optional(
                attr.validators.instance_of(MolecularAtmosphere)
            ),
        ),
        doc="Molecular atmosphere. Can be specified as a dictionary interpreted "
        'by :data:`.atmosphere_factory`; in that case, the ``"type"`` parameter '
        'can be ommitted and will automatically be set to \'``"molecular"``.',
        type=":class:`.MolecularAtmosphere`, optional",
        init_type=":class:`.MolecularAtmosphere` or dict, optional",
        default="None",
    )

    particle_layers: t.List[ParticleLayer] = documented(
        attr.ib(
            factory=list,
            converter=_heterogeneous_atmosphere_particle_converter,
            validator=attr.validators.deep_iterable(
                attr.validators.instance_of(ParticleLayer)
            ),
        ),
        doc="Particle layers. Can be specified as a dictionary interpreted "
        "by :data:`.atmosphere_factory`; in that case, the ``type`` parameter "
        "can be ommitted and will automatically be set to "
        "``particle_layer``.",
        type="list of :class:`.ParticleLayer`",
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
                    f"while validating {attribute.name}: found duplicate "
                    f"component ID '{component.id}'; "
                    f"all components (molecular atmosphere and particle layers) "
                    f"must have different IDs"
                )

    @molecular_atmosphere.validator
    @particle_layers.validator
    def _validate_widths(instance, attribute, value):
        if not all([component.width is AUTO for component in instance.components]):
            raise ValueError(
                "all components must have their 'width' attribute set to 'AUTO'"
            )

    @property
    def components(self) -> t.List[t.Union[MolecularAtmosphere, ParticleLayer]]:
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
        if self.width is not AUTO:
            return self.width

        else:
            widths = [component.eval_width(ctx=ctx) for component in self.components]

            if widths:
                return max(widths)
            else:
                return 1000.0 * ureg.km


def blend_radprops(
    background: xr.Dataset, foreground: xr.Dataset
) -> t.Tuple[xr.Dataset, xr.DataArray]:
    """
    Blend radiative properties of two participating media.

    Returns
    -------
    tuple[:class:`~xarray.Dataset`, :class:`~xarray.DataArray`]
        Radiative properties resulting from the blending of the two participating
        media, in the altitude range of the foreground participating medium.

    Notes
    -----
    Assumes 'z_layer' is in same units in 'molecules' and in 'particles'.
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

    Parameters
    ----------
    radprops : :class:`~xarray.Dataset`
        Radiative property data set.

    new_z_layer : :class:`~pint.Quantity`)
        Layer altitude grid to interpolate onto.

    Returns
    -------
    :class:`~xarray.Dataset`
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
    particle_layers: t.List[ParticleLayer], particle_layer: ParticleLayer
) -> t.List[ParticleLayer]:
    """
    Return the particle layers that are overlapping a given particle layer.

    Parameters
    ----------
     particle_layers : list of :class:`.ParticleLayer`
        List of particle layers to check for overlap.

    particle_layer : :class:`.ParticleLayer`
        The given particle layer.

    Returns
    -------
    list of :class:`.ParticleLayer`
        Overlapping particle layers.
    """
    return [
        layer
        for layer in particle_layers
        if particle_layer.top >= layer.top > particle_layer.bottom
        or particle_layer.top > layer.bottom >= particle_layer.bottom
    ]
