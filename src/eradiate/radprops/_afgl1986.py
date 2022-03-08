"""
AFGL (1986) radiative profile definition.
"""

from __future__ import annotations

import functools
import typing as t

import attr
import numpy as np
import pint
import xarray as xr

from . import _util_ckd
from ._core import RadProfile, make_dataset, rad_profile_factory
from .rayleigh import compute_sigma_s_air
from .._mode import UnsupportedModeError
from ..attrs import documented, parse_docs
from ..ckd import Bindex
from ..thermoprops import afgl_1986
from ..thermoprops.util import (
    compute_column_mass_density,
    compute_column_number_density,
    compute_volume_mixing_ratio_at_surface,
)
from ..units import to_quantity
from ..units import unit_registry as ureg


def _convert_thermoprops_afgl_1986(
    value: t.Union[t.MutableMapping, xr.Dataset]
) -> xr.Dataset:
    if isinstance(value, dict):
        return afgl_1986.make_profile(**value)
    else:
        return value


@rad_profile_factory.register(type_id="afgl_1986")
@parse_docs
@attr.s
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
        attr.ib(
            factory=lambda: afgl_1986.make_profile(),
            converter=_convert_thermoprops_afgl_1986,
            validator=attr.validators.instance_of(xr.Dataset),
        ),
        doc="Thermophysical properties.",
        type=":class:`~xarray.Dataset`",
        default=":func:`~eradiate.thermoprops.afgl_1986.make_profile`",
    )

    has_absorption: bool = documented(
        attr.ib(
            default=True,
            converter=bool,
            validator=attr.validators.instance_of(bool),
        ),
        doc="Absorption switch. If ``True``, the absorption coefficient is "
        "computed. Else, the absorption coefficient is not computed and "
        "instead set to zero.",
        type="bool",
        default="True",
    )

    has_scattering: bool = documented(
        attr.ib(
            default=True,
            converter=bool,
            validator=attr.validators.instance_of(bool),
        ),
        doc="Scattering switch. If ``True``, the scattering coefficient is "
        "computed. Else, the scattering coefficient is not computed and "
        "instead set to zero.",
        type="bool",
        default="True",
    )

    @property
    def thermoprops(self) -> xr.Dataset:
        return self._thermoprops

    @property
    def levels(self) -> pint.Quantity:
        return to_quantity(self.thermoprops.z_level)

    def eval_sigma_a_ckd(self, *bindexes: Bindex, bin_set_id: str) -> pint.Quantity:
        if bin_set_id is None:
            raise ValueError("argument 'bin_set_id' is required")

        if not self.has_absorption:
            return ureg.Quantity(np.zeros(self.thermoprops.z_layer.size), "km^-1")

        ds_id = f"{self.thermoprops.attrs['identifier']}-{bin_set_id}-v3"
        with _util_ckd.open_dataset(ds_id) as ds:
            # This table maps species to the function used to compute
            # corresponding physical quantities used for concentration rescaling
            compute = {
                "H2O": functools.partial(compute_column_mass_density, species="H2O"),
                "CO2": functools.partial(compute_volume_mixing_ratio_at_surface, species="CO2"),
                "O3": functools.partial(compute_column_number_density, species="O3"),
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
            result = ureg.Quantity(
                [
                    ds.k.sel(bd=(bindex.bin.id, bindex.index))
                    .interp(z=z, **concentrations, kwargs=dict(fill_value=0.0))
                    .values
                    for bindex in bindexes
                ],
                ds.k.units,
            )

            return result

    def eval_sigma_s_mono(self, w: pint.Quantity) -> pint.Quantity:
        thermoprops = self.thermoprops
        if self.has_scattering:
            return compute_sigma_s_air(
                wavelength=w,
                number_density=to_quantity(thermoprops.n),
            )
        else:
            return ureg.Quantity(np.zeros((1, thermoprops.z_layer.size)), "km^-1")

    def eval_sigma_s_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        wavelengths = ureg.Quantity(
            np.array([bindex.bin.wcenter.m_as("nm") for bindex in bindexes]), "nm"
        )
        return self.eval_sigma_s_mono(w=wavelengths)

    def eval_albedo_mono(self, w: pint.Quantity) -> pint.Quantity:
        raise UnsupportedModeError(supported="ckd")

    def eval_albedo_ckd(self, *bindexes: Bindex, bin_set_id: str) -> pint.Quantity:
        sigma_s = self.eval_sigma_s_ckd(*bindexes)
        sigma_t = self.eval_sigma_t_ckd(*bindexes, bin_set_id=bin_set_id)
        return np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        ).to("dimensionless")

    def eval_sigma_t_mono(self, w: pint.Quantity) -> pint.Quantity:
        raise UnsupportedModeError(supported="ckd")

    def eval_sigma_t_ckd(self, *bindexes: Bindex, bin_set_id: str) -> pint.Quantity:
        return self.eval_sigma_a_ckd(
            *bindexes, bin_set_id=bin_set_id
        ) + self.eval_sigma_s_ckd(*bindexes)

    def eval_dataset_mono(self, w: pint.Quantity) -> xr.Dataset:
        raise UnsupportedModeError(supported="ckd")

    def eval_dataset_ckd(self, *bindexes: Bindex, bin_set_id: str) -> xr.Dataset:
        if len(bindexes) > 1:
            raise NotImplementedError
        else:
            return make_dataset(
                wavelength=bindexes[0].bin.wcenter,
                z_level=to_quantity(self.thermoprops.z_level),
                z_layer=to_quantity(self.thermoprops.z_layer),
                sigma_a=self.eval_sigma_a_ckd(*bindexes, bin_set_id=bin_set_id),
                sigma_s=self.eval_sigma_s_ckd(*bindexes),
            ).squeeze()
