"""
Particle layers.
"""
from __future__ import annotations

import pathlib
import tempfile
from typing import Any, Dict, MutableMapping, Optional, Union

import attr
import numpy as np
import pint
import pinttr
import xarray as xr

import eradiate

from ._core import Atmosphere, write_binary_grid3d
from ._particle_dist import (
    ParticleDistribution,
    UniformParticleDistribution,
    particle_distribution_factory,
)
from ... import path_resolver
from ..._mode import ModeFlags
from ..._util import onedict_value
from ...attrs import AUTO, documented, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...exceptions import UnsupportedModeError
from ...kernel.transform import map_cube, map_unit_cube
from ...units import to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg
from ...validators import is_positive


@parse_docs
@attr.s
class ParticleLayer(Atmosphere):
    """
    Particle layer.
    The particle layer has a vertical extension specified by a bottom altitude
    (set by ``bottom``) and a top altitude (set by ``top``).
    Inside the layer, the particles number is distributed according to a
    distribution (set by ``distribution``).
    See :mod:`~eradiate.scenes.atmosphere.particle_dist` for the available distribution
    types and corresponding parameters.
    The particle layer is itself divided into a number of (sub-)layers
    (``n_layers``) to allow to describe the variations of the particles number
    with altitude.
    The total number of particles in the layer is adjusted so that the
    particle layer's optical thickness at 550 nm meet a specified value
    (``tau_550``).
    The particles radiative properties are specified by a data set
    (``dataset``).
    """

    _bottom: Optional[pint.Quantity] = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.km),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
            units=ucc.deferred("length"),
        ),
        doc="Bottom altitude of the particle layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length])",
        type="float",
        default="0 km",
    )

    _top: Optional[pint.Quantity] = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            default=ureg.Quantity(1.0, ureg.km),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Top altitude of the particle layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="1 km.",
    )

    @_bottom.validator
    @_top.validator
    def _bottom_and_top_validator(instance, attribute, value):
        if instance.bottom >= instance.top:
            raise ValueError("bottom altitude must be lower than top altitude")

    distribution: Optional[ParticleDistribution] = documented(
        attr.ib(
            factory=UniformParticleDistribution,
            converter=particle_distribution_factory.convert,
            validator=attr.validators.instance_of(ParticleDistribution),
        ),
        doc="Particle distribution.",
        type="dict or :class:`ParticleDistribution`",
        default=":class:`Uniform`",
    )

    tau_550: Optional[pint.Quantity] = documented(
        pinttr.ib(
            units=ucc.deferred("dimensionless"),
            default=ureg.Quantity(0.2, ureg.dimensionless),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Extinction optical thickness at the wavelength of 550 nm.\n"
        "\n"
        "Unit-enabled field (default: ucc[dimensionless]).",
        type="float",
        default="0.2",
    )

    n_layers: Optional[int] = documented(
        attr.ib(
            default=16,
            converter=int,
            validator=attr.validators.instance_of(int),
        ),
        doc="Number of layers inside the particle layer.\n",
        type="int",
        default="16",
    )

    dataset: Optional[str] = documented(
        attr.ib(
            default=path_resolver.resolve("tests/radprops/rtmom_aeronet_desert.nc"),
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Particles radiative properties data set path.",
        type="str",
    )

    albedo_filename: Optional[str] = documented(
        attr.ib(
            default="albedo.vol",
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Name of the albedo volume data file.",
        type="str",
        default='"albedo.vol"',
    )

    sigma_t_filename: Optional[str] = documented(
        attr.ib(
            default="sigma_t.vol",
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Name of the extinction coefficient volume data file.",
        type="str",
        default='"sigma_t.vol"',
    )

    weight_filename: Optional[str] = documented(
        attr.ib(
            default="weight.vol",
            converter=attr.converters.optional(str),
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="Name of the weight volume data file if the phase function is of type ``blendphase``.",
        type="str",
        default='"weight.vol"',
    )

    cache_dir: Optional[pathlib.Path] = documented(
        attr.ib(
            default=pathlib.Path(tempfile.mkdtemp()),
            converter=pathlib.Path,
            validator=attr.validators.instance_of(pathlib.Path),
        ),
        doc="Path to a cache directory where volume data files will be created.",
        type="path-like",
        default="Temporary directory",
    )

    def __attrs_post_init__(self):
        # Prepare cache directory in case we'd need it
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    #                        Volume data files
    # --------------------------------------------------------------------------

    @property
    def albedo_file(self) -> pathlib.Path:
        return self.cache_dir / self.albedo_filename

    @property
    def sigma_t_file(self) -> pathlib.Path:
        return self.cache_dir / self.sigma_t_filename

    @property
    def weight_file(self) -> pathlib.Path:
        return self.cache_dir / self.weight_filename

    # --------------------------------------------------------------------------
    #                    Spatial and thermophysical properties
    # --------------------------------------------------------------------------

    @property
    def top(self) -> pint.Quantity:
        return self._top

    @property
    def bottom(self) -> pint.Quantity:
        return self._bottom

    def eval_width(self, ctx: Optional[KernelDictContext]) -> pint.Quantity:
        if self.width is AUTO:
            spectral_ctx = ctx.spectral_ctx if ctx is not None else None
            return 10.0 / self.eval_sigma_s(spectral_ctx=spectral_ctx).min()
        else:
            return self.width

    @property
    def z_level(self) -> pint.Quantity:
        """
        Compute the level altitude mesh within the particle layer.

        The level altitude mesh corresponds to a regular level altitude mesh
        from the layer's bottom altitude to the layer's top altitude with
        a number of points specified by ``n_layer + 1``.

        Returns → :class:`~pint.Quantity`:
            Level altitude mesh.
        """
        return np.linspace(start=self.bottom, stop=self.top, num=self.n_layers + 1)

    @property
    def z_layer(self) -> pint.Quantity:
        """
        Compute the layer altitude mesh within the particle layer.

        The layer altitude mesh corresponds to a regular level altitude mesh
        from the layer's bottom altitude to the layer's top altitude with
        a number of points specified by ``n_layer``.

        Returns → :class:`~pint.Quantity`:
            Layer altitude mesh.
        """
        z_level = self.z_level
        return (z_level[:-1] + z_level[1:]) / 2.0

    def eval_fractions(self, z_layer: Optional[ureg.Quantity] = None) -> np.ndarray:
        """
        Compute the particle number fraction in the particle layer.

        Parameter ``z_layer`` (:class:`~pint.Quantity` or None):
            Layer altitude mesh onto which the fractions must be computed.

        Returns → :class:`~numpy.ndarray`:
            Particles fractions.
        """
        z_layer = self.z_layer if z_layer is None else z_layer
        return self.distribution.eval_fraction(z_layer)

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def eval_phase(self, spectral_ctx: SpectralContext) -> xr.DataArray:
        """
        Evaluate the phase function.

        The phase function is represented by a :class:`~xarray.DataArray` with
        a :math:`\\mu` (``mu``) coordinate for the scattering angle cosine
        (:math:`\\mu \\in [-1, 1]`).

        Returns → :class:`~xarray.DataArray`:
            Phase function.
        """
        ds = xr.open_dataset(self.dataset)
        return (
            ds.phase.sel(i=0)
            .sel(j=0)
            .interp(w=spectral_ctx.wavelength.magnitude, kwargs=dict(bounds_error=True))
        )

    def eval_albedo(
        self,
        spectral_ctx: SpectralContext,
        z_level: Optional[ureg.Quantity] = None,
    ) -> pint.Quantity:
        """
        Evaluate albedo given a spectral context.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext`):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Parameter ``z_level`` (:class:`~pint.Quantity` or None):
            Level altitude mesh onto which the fractions must be computed.

        Returns → :class:`~pint.Quantity`:
            Particle layer albedo.
        """
        wavelength = spectral_ctx.wavelength.magnitude
        ds = xr.open_dataset(self.dataset)
        interpolated_albedo = ds.albedo.interp(w=wavelength)
        albedo = to_quantity(interpolated_albedo)
        n_layers = self.n_layers if z_level is None else len(z_level) - 1
        albedo_array = albedo * np.ones(n_layers)
        return albedo_array

    def eval_sigma_t(
        self,
        spectral_ctx: SpectralContext,
        z_level: Optional[ureg.Quantity] = None,
    ) -> pint.Quantity:
        """
        Evaluate extinction coefficient given a spectral context.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext`):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Parameter ``z_level`` (:class:`~pint.Quantity` or None):
            Level altitude mesh onto which the fractions must be computed.

        Returns → :class:`~pint.Quantity`:
            Particle layer extinction coefficient.
        """
        wavelength = spectral_ctx.wavelength.magnitude
        ds = xr.open_dataset(self.dataset)
        interpolated_sigma_t = ds.sigma_t.interp(w=wavelength)
        sigma_t = to_quantity(interpolated_sigma_t)
        z_layer = None if z_level is None else (z_level[1:] + z_level[:-1]) / 2.0
        fractions = self.eval_fractions(z_layer=z_layer)
        sigma_t_array = sigma_t * fractions
        dz = (
            (self.top - self.bottom) / self.n_layers
            if z_level is None
            else z_level[1:] - z_level[:-1]
        )
        normalized_sigma_t_array = self._normalize_to_tau(
            ki=sigma_t_array.magnitude,
            dz=dz,
            tau=self.tau_550,
        )
        return normalized_sigma_t_array

    def eval_sigma_a(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext`):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns → :class:`~pint.Quantity`:
            Particle layer absorption coefficient.
        """
        return self.eval_sigma_t(spectral_ctx) - self.eval_sigma_a(spectral_ctx)

    def eval_sigma_s(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate scattering coefficient given a spectral context.

        Returns → :class:`~pint.Quantity`:
            Particle layer scattering coefficient.
        """
        return self.eval_sigma_t(spectral_ctx) * self.eval_albedo(spectral_ctx)

    def eval_radprops(
        self,
        spectral_ctx: SpectralContext,
        z_level: Optional[ureg.Quantity] = None,
    ) -> xr.Dataset:
        """
        Return a dataset that holds the radiative properties profile of the
        particle layer.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext`):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Parameter ``z_level`` (:class:`~pint.Quantity`):
            Level altitude mesh.
            This parameter allows to compute the radiative properties of the
            particle layer on a level altitude mesh that is different from
            the "native" (specified by ``n_layers``) particle layer mesh.
            If ``None``, the level altitude mesh is the regular mesh defined
            by ``bottom``, ``top`` and ``n_layers`` for the start value,
            stop value and number of points, respectively.

        Returns → :class:`~xarray.Dataset`:
            Particle layer radiative properties profile dataset.
        """
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            z_level = self.z_level if z_level is None else z_level
            z_layer = (z_level[1:] + z_level[:-1]) / 2.0
            sigma_t = self.eval_sigma_t(spectral_ctx=spectral_ctx, z_level=z_level)
            albedo = self.eval_albedo(spectral_ctx=spectral_ctx, z_level=z_level)
            wavelength = spectral_ctx.wavelength
            return xr.Dataset(
                data_vars={
                    "sigma_t": (
                        ("z_layer"),
                        sigma_t.magnitude,
                        dict(
                            standard_name="extinction_coefficient",
                            long_name="extinction coefficient",
                            units=sigma_t.units,
                        ),
                    ),
                    "albedo": (
                        ("z_layer"),
                        albedo.magnitude,
                        dict(
                            standard_name="albedo",
                            long_name="albedo",
                            units=albedo.units,
                        ),
                    ),
                },
                coords={
                    "z_layer": (
                        "z_layer",
                        z_layer.magnitude,
                        dict(
                            standard_name="layer_altitude",
                            long_name="layer altitude",
                            units=z_layer.units,
                        ),
                    ),
                    "z_level": (
                        "z_level",
                        z_level.magnitude,
                        dict(
                            standard_name="level_altitude",
                            long_name="level altitude",
                            units=z_level.units,
                        ),
                    ),
                    "wavelength": (
                        "w",
                        [wavelength.magnitude],
                        dict(
                            standard_name="wavelength",
                            long_name="wavelength",
                            units=wavelength.units,
                        ),
                    ),
                },
            ).isel(w=0)

        else:
            raise UnsupportedModeError(supported="monochromatic")

    @staticmethod
    @ureg.wraps(ret="km^-1", args=("", "km", ""), strict=False)
    def _normalize_to_tau(ki: np.ndarray, dz: np.ndarray, tau: float) -> np.ndarray:
        r"""
        Normalise extinction coefficient values :math:`k_i` so that:

        .. math::
            \sum_i k_i \Delta z = \tau_{550}

        where :math:`tau` is the particle layer optical thickness.

        Parameter ``ki`` (:class:`~pint.Quantity` or :class:`~np.ndarray`):
            Dimensionless extinction coefficients values [].

        Parameter ``dz`` (:class:`~pint.Quantity` or :class:`~np.ndarray`):
            Layer divisions thickness [km].

        Parameter ``tau`` (float):
            Layer optical thickness (dimensionless).

        Returns → :class:`~pint.Quantity`
            Normalised extinction coefficients.
        """
        return ki * tau / (np.sum(ki) * dz)

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    def _gridvolume_to_world_trafo(self, ctx: KernelDictContext) -> Any:
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

    def kernel_phase(self, ctx: KernelDictContext) -> MutableMapping:
        particles_phase = self.eval_phase(spectral_ctx=ctx.spectral_ctx)
        return {
            f"phase_{self.id}": {
                "type": "lutphase",
                "values": ",".join([str(value) for value in particles_phase.data]),
            }
        }

    def kernel_media(self, ctx: KernelDictContext) -> Dict:
        radprops = self.eval_radprops(spectral_ctx=ctx.spectral_ctx)
        albedo = to_quantity(radprops.albedo).m_as(uck.get("albedo"))
        sigma_t = to_quantity(radprops.sigma_t).m_as(uck.get("collision_coefficient"))
        write_binary_grid3d(
            filename=str(self.albedo_file), values=albedo[np.newaxis, np.newaxis, ...]
        )
        write_binary_grid3d(
            filename=str(self.sigma_t_file), values=sigma_t[np.newaxis, np.newaxis, ...]
        )
        trafo = self._gridvolume_to_world_trafo(ctx=ctx)

        if ctx.ref:
            phase = {"type": "ref", "id": f"phase_{self.id}"}
        else:
            phase = onedict_value(self.kernel_phase(ctx=ctx))

        return {
            f"medium_{self.id}": {
                "type": "heterogeneous",
                "phase": phase,
                "sigma_t": {
                    "type": "gridvolume",
                    "filename": str(self.sigma_t_file),
                    "to_world": trafo,
                },
                "albedo": {
                    "type": "gridvolume",
                    "filename": str(self.albedo_file),
                    "to_world": trafo,
                },
            }
        }

    def kernel_shapes(self, ctx: Optional[KernelDictContext]) -> Dict:
        if ctx.ref:
            medium = {"type": "ref", "id": f"medium_{self.id}"}
        else:
            medium = self.kernel_media(ctx)[f"medium_{self.id}"]

        length_units = uck.get("length")
        width = self.kernel_width(ctx).m_as(length_units)
        bottom = self.bottom.m_as(length_units)
        top = self.top.m_as(length_units)
        offset = self.kernel_offset(ctx).m_as(length_units)
        trafo = map_cube(
            xmin=-width / 2.0,
            xmax=width / 2.0,
            ymin=-width / 2.0,
            ymax=width / 2.0,
            zmin=bottom - offset,
            zmax=top,
        )

        return {
            f"shape_{self.id}": {
                "type": "cube",
                "to_world": trafo,
                "bsdf": {"type": "null"},
                "interior": medium,
            }
        }

    # --------------------------------------------------------------------------
    #                               Miscellaneous
    # --------------------------------------------------------------------------

    @classmethod
    def convert(cls: ParticleLayer, value: Union[ParticleLayer, dict]) -> ParticleLayer:
        """
        Object converter method.

        If ``value`` is a dictionary, this method forwards it to
        :meth:`.from_dict`. Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return cls.from_dict(value)

        return value
