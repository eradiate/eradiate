"""
Heterogeneous atmospheres.
"""
from __future__ import annotations

import typing as t
import warnings
from collections import abc as cabc

import attr
import numpy as np
import pint
import xarray as xr

from ._core import AbstractHeterogeneousAtmosphere, atmosphere_factory
from ._molecular_atmosphere import MolecularAtmosphere
from ._particle_layer import ParticleLayer
from ..core import KernelDict
from ..phase import BlendPhaseFunction
from ..._util import onedict_value
from ...attrs import AUTO, documented, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...units import to_quantity
from ...units import unit_context_config as ucc
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
        'can be omitted and will automatically be set to \'``"molecular"``.',
        type=":class:`.MolecularAtmosphere` or None",
        init_type=":class:`.MolecularAtmosphere` or dict, optional",
        default="None",
    )

    @molecular_atmosphere.validator
    def _molecular_atmosphere_validator(self, attribute, value):
        if value is None:
            return

        if value.width is not AUTO:
            raise ValueError(
                f"while validating {attribute.name}: all components must have "
                "their 'width' field set to AUTO"
            )

        if value.scale is not None:
            raise ValueError(
                f"while validating {attribute.name}: components cannot be "
                "scaled individually"
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
        "can be omitted and will automatically be set to "
        "``particle_layer``.",
        type="list of :class:`.ParticleLayer`",
        init_type="list of :class:`.ParticleLayer`, optional",
        default="[]",
    )

    @particle_layers.validator
    def _component_validator(self, attribute, value):
        if not all([component.width is AUTO for component in value]):
            raise ValueError(
                f"while validating {attribute.name}: all components must have "
                f"their 'width' field set to AUTO"
            )

        if not all(component.scale is None for component in value):
            raise ValueError(
                f"while validating {attribute.name}: components cannot be "
                "scaled individually"
            )

    @property
    def components(self) -> t.List[t.Union[MolecularAtmosphere, ParticleLayer]]:
        result = [self.molecular_atmosphere] if self.molecular_atmosphere else []
        result.extend(self.particle_layers)
        return result

    def update(self):
        super().update()

        # Force IDs and cache directories
        for i, component in enumerate(self.components):
            component.id = f"{self.id}_component_{i}"
            component.cache_dir = self.cache_dir

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
        radiative properties on, before computing the total radiative
        properties.
        """
        z_level = np.linspace(self.bottom, self.top, 100001)
        z_layer = 0.5 * (z_level[1:] + z_level[:-1])
        return z_layer

    def eval_radprops(self, spectral_ctx: SpectralContext) -> xr.Dataset:
        """
        Evaluate the extinction coefficients and albedo profiles.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        Dataset
            A dataset containing with the following variables for the specified
            spectral context:

            * ``sigma_t``: extinction coefficient;
            * ``albedo``: albedo.

            Coordinates are the following:

            * ``z``: altitude.
        """
        components = self.components

        # Nothing: return zeros
        if len(components) == 0:
            return _zero_radprops(spectral_ctx)

        # Single component: just forward encapsulated component
        elif len(components) == 1:
            return components[0].eval_radprops(spectral_ctx)

        # Two components or more: interpolate all components on a fine grid and
        # aggregate collision coefficients
        else:
            hrz = self._high_res_z_layer()
            sigma_units = ucc.get("collision_coefficient")
            sigma_ss = []
            sigma_ts = []

            for component in components:
                radprops = interpolate_radprops(
                    component.eval_radprops(spectral_ctx), new_z_layer=hrz
                )
                # We store only the magnitude
                sigma_ts.append(to_quantity(radprops.sigma_t).m_as(sigma_units))
                sigma_ss.append(
                    (to_quantity(radprops.sigma_t) * radprops.albedo.values).m_as(
                        sigma_units
                    )
                )

            sigma_t = sum(sigma_ts)
            sigma_s = sum(sigma_ss)

            albedo = np.divide(
                sigma_s,
                sigma_t,
                where=sigma_t != 0.0,
                out=np.ones_like(sigma_t),
            )

            result = xr.Dataset(
                data_vars={
                    "sigma_t": (
                        "z_layer",
                        sigma_t,
                        {
                            "units": f"{sigma_units:~P}",
                            "standard_name": "extinction_coefficient",
                            "long_name": "extinction coefficient",
                        },
                    ),
                    "albedo": (
                        "z_layer",
                        albedo,
                        {
                            "standard_name": "albedo",
                            "long_name": "albedo",
                            "units": "",
                        },
                    ),
                },
                coords={
                    "z_layer": (
                        "z_layer",
                        hrz.magnitude,
                        {
                            "units": f"{hrz.units:~P}",
                            "standard_name": "layer_altitude",
                            "long_name": "layer altitude",
                        },
                    )
                },
            )

            return result

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def kernel_shapes(self, ctx: KernelDictContext) -> KernelDict:
        if len(self.components) == 0:
            return KernelDict()

        else:
            return super().kernel_shapes(ctx)

    def kernel_media(self, ctx: KernelDictContext) -> KernelDict:
        if len(self.components) == 0:
            return KernelDict()

        else:
            return super().kernel_media(ctx)

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
        components = self.components

        # Nothing: return empty dict
        if len(components) == 0:
            return KernelDict()

        # Single component: just forward encapsulated component
        elif len(components) == 1:
            return KernelDict(
                {self.id_phase: onedict_value(components[0].kernel_phase(ctx).data)}
            )

        # Two components or more: blend phase functions
        else:
            # Component weights are given by the scattering coefficient:
            # we collect values and interpolate on the global grid
            hrz = self._high_res_z_layer()
            sigma_ss = []

            for component in components:
                radprops = interpolate_radprops(
                    component.eval_radprops(ctx.spectral_ctx), new_z_layer=hrz
                )
                sigma_ss.append(radprops.sigma_t * radprops.albedo)

            # Construct a blended phase function based on those weighting values
            phase = BlendPhaseFunction(
                components=[component.phase for component in components],
                weights=sigma_ss,
            )

            return phase.kernel_dict(ctx)

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        # Shortcut: an empty HeterogeneousAtmosphere returns no kernel object
        components = self.components
        if len(components) == 0:
            return KernelDict()

        else:
            return super().kernel_dict(ctx)


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
    interpolated : Dataset
        Interpolated radiative property data set.
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
