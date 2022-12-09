"""
AFGL (1986) radiative profile definition.
"""

from __future__ import annotations

import typing as t
import warnings
from functools import partial, singledispatchmethod
from os import PathLike

import attrs
import numpy as np
import pandas as pd
import pint
import xarray as xr

from ._core import RadProfile, make_dataset
from .rayleigh import compute_sigma_s_air
from .. import converters
from ..attrs import documented, parse_docs
from ..quad import Quad
from ..spectral_index import CKDSpectralIndex, SpectralIndex
from ..thermoprops import afgl_1986
from ..thermoprops.util import (
    column_mass_density,
    column_number_density,
    volume_mixing_ratio_at_surface,
)
from ..units import to_quantity
from ..units import unit_registry as ureg

G16 = Quad.gauss_legendre(16).eval_nodes(interval=[0.0, 1.0])

absorption_dataset_convert = converters.to_dataset(load_from_id=None)


def _convert_thermoprops_afgl_1986(
    value: t.Union[t.MutableMapping, xr.Dataset]
) -> xr.Dataset:
    if isinstance(value, dict):
        return afgl_1986.make_profile(**value)
    else:
        return value


@parse_docs
@attrs.define
class AFGL1986RadProfile(RadProfile):
    """
    Radiative properties profile corresponding to the AFGL (1986) atmospheric
    thermophysical properties profiles
    :cite:`Anderson1986AtmosphericConstituentProfiles`.

    Warnings
    --------
    This class does not support ``mono`` modes.
    """

    _thermoprops: xr.Dataset = documented(
        attrs.field(
            factory=lambda: afgl_1986.make_profile(),
            converter=_convert_thermoprops_afgl_1986,
            validator=attrs.validators.instance_of(xr.Dataset),
        ),
        doc="Thermophysical properties.",
        type=":class:`~xarray.Dataset`",
        default=":func:`~eradiate.thermoprops.afgl_1986.make_profile`",
    )

    has_absorption: bool = documented(
        attrs.field(
            default=True,
            converter=bool,
            validator=attrs.validators.instance_of(bool),
        ),
        doc="Absorption switch. If ``True``, the absorption coefficient is "
        "computed. Else, the absorption coefficient is not computed and "
        "instead set to zero.",
        type="bool",
        default="True",
    )

    has_scattering: bool = documented(
        attrs.field(
            default=True,
            converter=bool,
            validator=attrs.validators.instance_of(bool),
        ),
        doc="Scattering switch. If ``True``, the scattering coefficient is "
        "computed. Else, the scattering coefficient is not computed and "
        "instead set to zero.",
        type="bool",
        default="True",
    )

    absorption_dataset: t.Optional[t.Union[PathLike, xr.Dataset]] = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(absorption_dataset_convert),  # TODO: open dataset (not load)
            validator=attrs.validators.optional(attrs.validators.instance_of(xr.Dataset)),
        ),
        doc="Absorption coefficient dataset. If ``None``, the absorption "
        "coefficient is set to zero.",
        type=":class:`~xarray.Dataset`",
        default="None",
    )

    @absorption_dataset.validator
    @has_absorption.validator
    def _check_absorption_dataset(self, attribute, value):
        if not self.has_absorption and self.absorption_dataset is not None:
            warnings.warn(
                "When validating attribute 'absorption_dataset': specified "
                "absorption dataset will be ignored because absorption is "
                "disabled."
            )

    @property
    def thermoprops(self) -> xr.Dataset:
        return self._thermoprops

    @property
    def levels(self) -> pint.Quantity:
        return to_quantity(self.thermoprops.z_level)

    @singledispatchmethod
    def eval_albedo(self, spectral_index: SpectralIndex) -> pint.Quantity:
        raise NotImplementedError
    
    @eval_albedo.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_albedo_ckd(spectral_index.bindexes, spectral_index.bin_set_id)

    @singledispatchmethod
    def eval_sigma_a(self, spectral_index: SpectralIndex) -> pint.Quantity:
        raise NotImplementedError
    
    @eval_sigma_a.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_a_ckd(w=spectral_index.w, g=spectral_index.g)
    
    @singledispatchmethod
    def eval_sigma_s(self, spectral_index: SpectralIndex) -> pint.Quantity:
        raise NotImplementedError
    
    @eval_sigma_s.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_s_ckd(w=spectral_index.w, g=spectral_index.g)
    
    @singledispatchmethod
    def eval_sigma_t(self, spectral_index: SpectralIndex) -> pint.Quantity:
        raise NotImplementedError
    
    @eval_sigma_t.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_t_ckd(w=spectral_index.w, g=spectral_index.g)

    def eval_sigma_a_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        if not self.has_absorption:
            return ureg.Quantity(np.zeros(self.thermoprops.z_layer.size), "km^-1")

        ds = self.absorption_dataset

        # Combine the 'bin' and 'index' coordinates into a multi-index, then reindex dataset
        idx = pd.MultiIndex.from_arrays(
            (ds.bin.values, ds.index.values), names=("bin", "index")
        )
        ds = ds.drop_vars(("bin", "index"))
        ds = ds.reindex({"bd": idx})
        
        # This table maps species to the function used to compute
        # corresponding physical quantities used for concentration rescaling
        compute = {
            "H2O": partial(column_mass_density, species="H2O"),
            "CO2": partial(volume_mixing_ratio_at_surface, species="CO2"),
            "O3": partial(column_number_density, species="O3"),
        }

        # Evaluate species concentrations
        # Note: This includes a conversion to units appropriate for interpolation
        rescaled_species = list(compute.keys())
        concentrations = {
            species: compute[species](ds=self.thermoprops).m_as(ds[species].units)
            for species in rescaled_species
        }

        # Convert altitude to units appropriate for interpolation
        z = to_quantity(self.thermoprops.z_layer).m_as(ds.z.units)

        # Interpolate absorption coefficient on concentration coordinates
        bin_id = str(int(w.m_as("nm")))
        g_index = int(np.where(np.isclose(g, G16))[0])

        sigma_a_value = ds.k.sel(bd=(bin_id, g_index)).interp(
            z=z,
            kwargs=dict(fill_value=0.0),  # extrapolate to 0.0 for altitude out of bounds
        ).interp(
            **concentrations,
            kwargs=dict(
                bounds_error=True  # raise when concentration are out of bounds
            ),
        ).values

        return ureg.Quantity(sigma_a_value, ds.k.units)

    def eval_sigma_s_mono(self, w: pint.Quantity) -> pint.Quantity:
        thermoprops = self.thermoprops
        if self.has_scattering:
            return compute_sigma_s_air(
                wavelength=w,
                number_density=to_quantity(thermoprops.n),
            )
        else:
            return ureg.Quantity(np.zeros((1, thermoprops.z_layer.size)), "km^-1")

    def eval_sigma_s_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        return self.eval_sigma_s_mono(w=w)

    def eval_albedo_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        sigma_s = self.eval_sigma_s_ckd(w=w, g=g)
        sigma_t = self.eval_sigma_t_ckd(w=w, g=g)
        return np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        ).to("dimensionless")

    def eval_sigma_t_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        return self.eval_sigma_a_ckd(w=w, g=g) + self.eval_sigma_s_ckd(w=w, g=g)

    def eval_dataset_mono(self, w: pint.Quantity) -> xr.Dataset:
        raise NotImplementedError

    def eval_dataset_ckd(self, w: pint.Quantity, g: float) -> xr.Dataset:
        if w.size > 1:
            raise NotImplementedError
        else:
            return make_dataset(
                wavelength=w,
                z_level=to_quantity(self.thermoprops.z_level),
                z_layer=to_quantity(self.thermoprops.z_layer),
                sigma_a=self.eval_sigma_a_ckd(w=w, g=g),
                sigma_s=self.eval_sigma_s_ckd(w=w, g=g),
            ).squeeze()
