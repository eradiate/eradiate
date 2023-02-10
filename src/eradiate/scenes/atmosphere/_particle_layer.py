"""
Particle layers.
"""
from __future__ import annotations

import typing as t

import attrs
import numpy as np
import pint
import pinttr
import xarray as xr

import eradiate

from ._core import AbstractHeterogeneousAtmosphere
from ._particle_dist import ParticleDistribution, particle_distribution_factory
from ..core import traverse
from ..phase import TabulatedPhaseFunction
from ... import converters, data
from ...attrs import documented, parse_docs
from ...ckd import Bindex
from ...contexts import KernelDictContext, SpectralContext
from ...exceptions import UnsupportedModeError
from ...kernel import UpdateParameter
from ...radprops import ZGrid
from ...units import to_quantity
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg
from ...validators import is_positive


def _particle_layer_distribution_converter(value):
    if isinstance(value, str):
        if value == "uniform":
            return particle_distribution_factory.convert({"type": "uniform"})
        elif value == "gaussian":
            return particle_distribution_factory.convert({"type": "gaussian"})
        elif value == "exponential":
            return particle_distribution_factory.convert({"type": "exponential"})

    return particle_distribution_factory.convert(value)


@parse_docs
@attrs.define(eq=False, slots=False)
class ParticleLayer(AbstractHeterogeneousAtmosphere):
    """
    Particle layer scene element [``particle_layer``].

    The particle layer has a vertical extension specified by a bottom altitude
    (set by ``bottom``) and a top altitude (set by ``top``).
    Inside the layer, the particles number is distributed according to a
    distribution (set by ``distribution``).
    See :mod:`~eradiate.scenes.atmosphere.particle_dist` for the available
    distribution types and corresponding parameters.
    The particle layer is itself divided into a number of (sub-)layers
    (``n_layers``) to allow to describe the variations of the particles number
    with altitude.
    The particle density in the layer is adjusted so that the particle layer's
    optical thickness at a specified reference wavelength (``w_ref``) meets a
    specified value (``tau_ref``).
    The particles radiative properties are specified by a data set
    (``dataset``).
    """

    _bottom: pint.Quantity = documented(
        pinttr.field(
            default=ureg.Quantity(0.0, ureg.km),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
            units=ucc.deferred("length"),
        ),
        doc="Bottom altitude of the particle layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length])",
        type="quantity",
        init_type="float or quantity",
        default="0 km",
    )

    _top: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("length"),
            default=ureg.Quantity(1.0, ureg.km),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Top altitude of the particle layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="float or quantity",
        default="1 km.",
    )

    @_bottom.validator
    @_top.validator
    def _bottom_top_validator(self, attribute, value):
        if self.bottom >= self.top:
            raise ValueError(
                f"while validating '{attribute.name}': bottom altitude must be "
                "lower than top altitude "
                f"(got bottom={self.bottom}, top={self.top})"
            )

    distribution: ParticleDistribution = documented(
        attrs.field(
            default="uniform",
            converter=_particle_layer_distribution_converter,
            validator=attrs.validators.instance_of(ParticleDistribution),
        ),
        doc="Particle distribution. Simple defaults can be set using a string: "
        '``"uniform"`` (resp. ``"gaussian"``, ``"exponential"``) is converted to '
        ":class:`UniformParticleDistribution() <.UniformParticleDistribution>` "
        "(resp. :class:`GaussianParticleDistribution() <.GaussianParticleDistribution>`, "
        ":class:`ExponentialParticleDistribution() <.ExponentialParticleDistribution>`).",
        init_type=":class:`.ParticleDistribution` or dict or "
        '{"uniform", "gaussian", "exponential"}, optional',
        type=":class:`.ParticleDistribution`",
        default='"uniform"',
    )

    w_ref: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("length"),
            default=550 * ureg.nm,
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Reference wavelength at which the extinction optical thickness is "
        "specified.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="quantity",
        init_type="quantity or float",
        default="550.0",
    )

    tau_ref: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("dimensionless"),
            default=ureg.Quantity(0.2, ureg.dimensionless),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Extinction optical thickness at the reference wavelength.\n"
        "\n"
        "Unit-enabled field (default: ucc[dimensionless]).",
        type="quantity",
        init_type="quantity or float",
        default="0.2",
    )

    n_layers: int = documented(
        attrs.field(
            default=16,
            converter=int,
            validator=attrs.validators.instance_of(int),
        ),
        doc="Number of layers in which the particle layer is discretised by "
        "default.",
        type="int",
        default="16",
    )

    dataset: xr.Dataset = documented(
        attrs.field(
            default="govaerts_2021-continental",
            converter=converters.to_dataset(
                load_from_id=lambda x: data.load_dataset(f"spectra/particles/{x}.nc")
            ),
            validator=attrs.validators.instance_of(xr.Dataset),
        ),
        doc="Particle radiative property data set."
        "If an xarray dataset is passed, the dataset is used as is "
        "(refer to the data guide for the format requirements of this dataset)."
        "If a path is passed, the converter tries to open the corresponding "
        "file on the hard drive; should that fail, it queries the Eradiate data"
        "store with that path."
        "If a string is passed, it is interpreted as an identifier for a "
        "particle radiative property dataset in the Eradiate data store.",
        type="Dataset",
        init_type="Dataset or path-like or str, optional",
        default="govaerts_2021-continental",
    )

    _phase: t.Optional[TabulatedPhaseFunction] = attrs.field(default=None, init=False)
    _zgrid: t.Optional[ZGrid] = attrs.field(default=None, init=False)

    def update(self) -> None:
        super().update()

        ds = self.dataset
        self._phase = TabulatedPhaseFunction(id=self.phase_id, data=ds.phase)
        self._zgrid = ZGrid(np.linspace(self.bottom, self.top, self.n_layers + 1))

    # --------------------------------------------------------------------------
    #                    Spatial and thermophysical properties
    # --------------------------------------------------------------------------

    @property
    def top(self) -> pint.Quantity:
        # Inherit docstring
        return self._top

    @property
    def bottom(self) -> pint.Quantity:
        # Inherit docstring
        return self._bottom

    @property
    def zgrid(self) -> ZGrid:
        # Inherit docstring
        return self._zgrid

    @property
    def z_level(self) -> pint.Quantity:
        return self.zgrid.levels

    @property
    def z_layer(self) -> pint.Quantity:
        return self.zgrid.layers

    def eval_fractions(self, zgrid: ZGrid) -> np.ndarray:
        """
        Compute the particle number fraction in the particle layer.

        Returns
        -------
        ndarray
            Particle fractions.
        """
        x = (zgrid.layers - self.bottom) / (self.top - self.bottom)
        fractions = self.distribution(x.m_as(ureg.dimensionless))
        fractions /= np.sum(fractions)
        return fractions

    def eval_mfp(self, ctx: KernelDictContext) -> pint.Quantity:
        min_sigma_s = self.eval_sigma_s(sctx=ctx.spectral_ctx).min()
        return 1.0 / min_sigma_s if min_sigma_s != 0.0 else np.inf * ureg.m

    # --------------------------------------------------------------------------
    #                       Radiative properties
    # --------------------------------------------------------------------------

    def eval_albedo(
        self, sctx: SpectralContext, zgrid: t.Optional[ZGrid] = None
    ) -> pint.Quantity:
        # Inherit docstring

        if eradiate.mode().is_mono:
            return self.eval_albedo_mono(
                sctx.wavelength,
                self.zgrid if zgrid is None else zgrid,
            )

        elif eradiate.mode().is_ckd:
            return self.eval_albedo_ckd(
                sctx.bindex,
                self.zgrid if zgrid is None else zgrid,
            )

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_albedo_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        ds = self.dataset
        wavelengths = w.m_as(ds.w.attrs["units"])
        interpolated_albedo = ds.albedo.interp(w=wavelengths)
        albedo = to_quantity(interpolated_albedo)

        # Albedo is constant vs spatial dimension
        return np.full_like(zgrid.layers, albedo)

    def eval_albedo_ckd(
        self, bindexes: t.Union[Bindex, t.List[Bindex]], zgrid: ZGrid
    ) -> pint.Quantity:
        w_units = ureg.nm
        if isinstance(bindexes, Bindex):
            bindexes = [bindexes]

        w = [bindex.bin.wcenter.m_as(w_units) for bindex in bindexes] * w_units
        return self.eval_albedo_mono(w, zgrid)

    def eval_sigma_t(
        self, sctx: SpectralContext, zgrid: t.Optional[ZGrid] = None
    ) -> pint.Quantity:
        # Inherit docstring

        if eradiate.mode().is_mono:
            return self.eval_sigma_t_mono(
                sctx.wavelength,
                self.zgrid if zgrid is None else zgrid,
            )

        elif eradiate.mode().is_ckd:
            return self.eval_sigma_t_ckd(
                sctx.bindex,
                self.zgrid if zgrid is None else zgrid,
            )

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_sigma_t_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        # Prepare input data
        ds = self.dataset
        ds_w_units = ureg(ds.w.attrs["units"])
        wavelength = w.m_as(ds_w_units)
        xs_t = to_quantity(ds.sigma_t.interp(w=wavelength))
        xs_t_ref = to_quantity(ds.sigma_t.interp(w=self.w_ref.m_as(ds_w_units)))

        # Compute volume fractions on the requested altitude grid
        fractions = self.eval_fractions(zgrid)
        sigma_t_array = xs_t_ref * fractions

        # Normalise the extinction coefficient to the nominal optical thickness
        # so that Σ_i (k_i Δz_i) == τ
        normalized_sigma_t_array = (
            sigma_t_array.magnitude
            * self.tau_ref
            / (np.sum(sigma_t_array.magnitude) * zgrid.layer_height.magnitude)
        ) * zgrid.layer_height.units**-1
        result = np.atleast_1d(normalized_sigma_t_array * xs_t / xs_t_ref)

        return result

    def eval_sigma_t_ckd(
        self, bindexes: t.Union[Bindex, t.List[Bindex]], zgrid: ZGrid
    ) -> pint.Quantity:
        w_units = ureg.nm
        if isinstance(bindexes, Bindex):
            bindexes = [bindexes]
        w = [bindex.bin.wcenter.m_as(w_units) for bindex in bindexes] * w_units
        return self.eval_sigma_t_mono(w, zgrid)

    def eval_sigma_a(self, sctx: SpectralContext, zgrid: ZGrid = None) -> pint.Quantity:
        # Inherit docstring

        if eradiate.mode().is_mono:
            return self.eval_sigma_a_mono(
                sctx.wavelength,
                self.zgrid if zgrid is None else zgrid,
            )

        elif eradiate.mode().is_ckd:
            return self.eval_sigma_a_ckd(
                sctx.bindex,
                self.zgrid if zgrid is None else zgrid,
            )

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_sigma_a_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_sigma_t_mono(w, zgrid) - self.eval_sigma_s_mono(w, zgrid)

    def eval_sigma_a_ckd(self, bindexes: t.Union[Bindex, t.List[Bindex]], zgrid: ZGrid):
        return self.eval_sigma_t_ckd(bindexes, zgrid) - self.eval_sigma_s_ckd(
            bindexes, zgrid
        )

    def eval_sigma_s(
        self, sctx: SpectralContext, zgrid: t.Optional[ZGrid] = None
    ) -> pint.Quantity:
        # Inherit docstring

        if eradiate.mode().is_mono:
            return self.eval_sigma_s_mono(
                sctx.wavelength,
                self.zgrid if zgrid is None else zgrid,
            )

        elif eradiate.mode().is_ckd:
            return self.eval_sigma_s_ckd(
                sctx.bindex,
                self.zgrid if zgrid is None else zgrid,
            )

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_sigma_s_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_sigma_t_mono(w, zgrid) * self.eval_albedo_mono(w, zgrid)

    def eval_sigma_s_ckd(self, bindexes: Bindex, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_sigma_t_ckd(bindexes, zgrid) * self.eval_albedo_ckd(
            bindexes, zgrid
        )

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def phase(self) -> TabulatedPhaseFunction:
        # Inherit docstring
        return self._phase

    @property
    def _template_phase(self) -> dict:
        result, _ = traverse(self.phase)
        return result.data

    @property
    def _params_phase(self) -> t.Dict[str, UpdateParameter]:
        _, result = traverse(self.phase)
        return result.data
