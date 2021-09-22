"""
Heterogeneous atmospheres.
"""
import pathlib
import typing as t
from collections import abc as cabc

import attr
import numpy as np
import pint
import xarray as xr

from ._core import (
    AbstractHeterogeneousAtmosphere,
    atmosphere_factory,
    write_binary_grid3d,
)
from ._molecules import MolecularAtmosphere
from ._particles import ParticleLayer
from ..core import KernelDict
from ...attrs import AUTO, documented, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...kernel.transform import map_unit_cube
from ...units import to_quantity
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


def _zero_radprops(spectral_ctx: SpectralContext) -> xr.Dataset:
    """
    Returns a radiative properties data set with extinction coefficient and
    albedo set to zero.
    """
    return xr.Dataset(
        data_vars={
            "sigma_t": (
                "z_layer",
                np.zeros(2),
                dict(
                    standard_name="volume_extinction_coefficient",
                    long_name="extinction coefficient",
                    units="km^-1",
                ),
            ),
            "albedo": (
                "z_layer",
                np.zeros(2),
                dict(
                    standard_name="albedo",
                    long_name="albedo",
                    units="",
                ),
            ),
        },
        coords={
            "z_layer": (
                "z_layer",
                np.array([0, 1]),
                dict(standard_name="altitude", long_name="altitude", units="km"),
            ),
            "w": (
                "w",
                [spectral_ctx.wavelength.m_as(ureg.nm)],
                dict(
                    standard_name="radiation_wavelength",
                    long_name="wavelength",
                    units="nm",
                ),
            ),
        },
    )


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

    weight_filename: str = documented(
        attr.ib(
            default="weight.vol",
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Name of the weight volume data file",
        type="str",
        default='"weight.vol"',
    )

    @particle_layers.validator
    def _validate_particle_layer(instance, attribute, value):
        if len(value) > 1:
            raise NotImplementedError(
                "Support for more than one particle layer is not yet available."
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
    #                        Volume data files
    # --------------------------------------------------------------------------

    @property
    def weight_file(self) -> pathlib.Path:
        return self.cache_dir / self.weight_filename

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

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def _high_res_z_layer(self) -> pint.Quantity:
        """
        A high-resolution layer altitude mesh to interpolate the components'
        radiative properties on, before computinf the total radiative
        properties.
        """
        z_level = np.linspace(self.bottom, self.top, 100001)
        z_layer = (z_level[1:] + z_level[:-1]) / 2.0
        return z_layer

    def eval_radprops(self, spectral_ctx: SpectralContext) -> t.Optional[xr.Dataset]:
        """
        Evaluates the total (molecular atmosphere + particle layer) radiative
        properties.

        If there is no molecular atmosphere, the radiative properties of the
        particle layer are returned.
        If there is no particle layer, the radiative properties of the molecular
        atmosphere are returned.
        If there is a molecular atmosphere and a particle layer, their
        radiative properties are interpolated on a high resolution altitude
        mesh before the total radiative properties are computed.
        """
        # Nothing
        if self.molecular_atmosphere is None and not self.particle_layers:
            return _zero_radprops(spectral_ctx)
        # Only a particle layer
        elif self.molecular_atmosphere is None and self.particle_layers:
            return self.particle_layers[0].eval_radprops(spectral_ctx)
        # Only a molecular atmosphere
        elif self.molecular_atmosphere is not None and not self.particle_layers:
            return self.molecular_atmosphere.eval_radprops(spectral_ctx)
        # Both a molecular atmosphere and a particle layer
        else:
            hrz = self._high_res_z_layer()

            molecular = interpolate_radprops(
                self.molecular_atmosphere.eval_radprops(spectral_ctx),
                new_z_layer=hrz,
            )

            layer = interpolate_radprops(
                radprops=self.particle_layers[0].eval_radprops(spectral_ctx),
                new_z_layer=hrz,
            )

            # total radiative properties
            with xr.set_options(keep_attrs=True):
                sigma_t = molecular.sigma_t + layer.sigma_t
                albedo = np.divide(
                    layer.albedo * layer.sigma_t + molecular.albedo * molecular.sigma_t,
                    sigma_t,
                    where=sigma_t != 0.0,
                    out=np.ones_like(sigma_t),
                )
                weight = np.divide(
                    layer.sigma_t,
                    sigma_t,
                    where=sigma_t != 0.0,
                    out=np.ones_like(sigma_t),
                )

            return xr.Dataset(
                data_vars={
                    "sigma_t": sigma_t,
                    "albedo": albedo,
                    "weight": weight,
                }
            )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _gridvolume_to_world_trafo(self, ctx: KernelDictContext) -> t.Any:
        """
        Returns the 'to_world' transformation for gridvolume plugins.
        """
        length_units = uck.get("length")
        width = self.kernel_width(ctx).m_as(length_units)
        top = self.top.m_as(length_units)
        bottom = self.bottom.m_as(length_units)
        return map_unit_cube(
            xmin=-width / 2.0,
            xmax=width / 2.0,
            ymin=-width / 2.0,
            ymax=width / 2.0,
            zmin=bottom,
            zmax=top,
        )

    def kernel_phase(self, ctx: KernelDictContext) -> KernelDict:
        """
        Return phase function plugin specifications only.

        Parameters
        ----------
        ctx : :class:`.KernelDictContext`
            A context data structure containing parameters relevant for kernel
            dictionary generation.

        Returns
        -------
        :class:`.KernelDict`
            A kernel dictionary containing all the phase functions attached to
            the atmosphere.
        """
        # nothing
        if self.molecular_atmosphere is None and not self.particle_layers:
            return {}

        # only a particle particle
        elif self.molecular_atmosphere is None:
            particle_layer = self.particle_layers[0]
            return KernelDict(
                {
                    f"phase_{self.id}": particle_layer.kernel_phase(ctx).data[
                        f"phase_{particle_layer.id}"
                    ]
                }
            )
        # only a molecular atmosphere
        elif not self.particle_layers:
            return KernelDict(
                {
                    f"phase_{self.id}": self.molecular_atmosphere.kernel_phase(
                        ctx
                    ).data[f"phase_{self.molecular_atmosphere.id}"]
                }
            )
        # a molecular atmosphere and a molecular particle layer
        else:
            radprops = self.eval_radprops(ctx.spectral_ctx)
            write_binary_grid3d(
                filename=self.weight_file,
                values=radprops.weight.values[np.newaxis, np.newaxis, ...],
            )
            return KernelDict(
                {
                    f"phase_{self.id}": {
                        "type": "blendphase",
                        "weight": {
                            "type": "gridvolume",
                            "filename": str(self.weight_file),
                            "to_world": self._gridvolume_to_world_trafo(ctx=ctx),
                        },
                        **self.molecular_atmosphere.kernel_phase(ctx=ctx),
                        **self.particle_layers[0].kernel_phase(ctx=ctx),
                    }
                }
            )


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

    with xr.set_options(keep_attrs=True):
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
    with xr.set_options(keep_attrs=True):
        interpolated_safe = radprops.interp(
            z_layer=masked.m_as(radprops.z_layer.units),
            kwargs=dict(
                fill_value="extrapolate"
            ),  # handle floating point arithmetic issue
            method="nearest",  # radiative properties are assumed uniform in altitude layers
        )

    # interpolate over the full range
    with xr.set_options(keep_attrs=True):
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
