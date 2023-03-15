"""
AFGL (1986) radiative profile definition.
"""

from __future__ import annotations

import functools
import typing as t

import attrs
import numpy as np
import pint
import xarray as xr

from . import _util_ckd
from ._core import RadProfile, ZGrid, make_dataset
from .rayleigh import compute_sigma_s_air
from .._mode import UnsupportedModeError
from ..attrs import documented, parse_docs
from ..ckd import Bindex
from ..thermoprops import afgl_1986
from ..thermoprops.util import (
    column_mass_density,
    column_number_density,
    volume_mixing_ratio_at_surface,
)
from ..units import to_quantity
from ..units import unit_registry as ureg


def _convert_thermoprops_afgl_1986(value: t.MutableMapping | xr.Dataset) -> xr.Dataset:
    if isinstance(value, dict):
        return afgl_1986.make_profile(**value)
    else:
        return value


@parse_docs
@attrs.define(eq=False)
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

    _zgrid: ZGrid | None = attrs.field(default=None, init=False)

    def __attrs_post_init__(self):
        self.update()

    def update(self) -> None:
        self._zgrid = ZGrid(levels=self.levels)

    @property
    def thermoprops(self) -> xr.Dataset:
        return self._thermoprops

    @property
    def levels(self) -> pint.Quantity:
        return to_quantity(self.thermoprops.z_level)

    @property
    def zgrid(self) -> ZGrid:
        # Inherit docstring
        return self._zgrid

    @functools.lru_cache(maxsize=1)
    def _thermoprops_interp(self, zgrid: ZGrid) -> xr.Dataset:
        # Interpolate thermophysical profile on specified altitude grid
        # Note: we use a nearest neighbour scheme (so far, it doesn't seem to make a difference)
        # Note: this value is cached so that repeated calls with the same zgrid
        #       won't trigger an unnecessary computation.
        with xr.set_options(keep_attrs=True):
            result = self.thermoprops.interp(
                z_layer=zgrid.layers.m_as(self.thermoprops.z_layer.units),
                method="nearest",
                kwargs={"fill_value": "extrapolate"},
            )

        return result.assign_coords(
            z_level=(
                "z_level",
                zgrid.levels.m_as(self.thermoprops.z_level.units),
                self.thermoprops.z_level.attrs,
            )
        )

    def eval_albedo_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        raise UnsupportedModeError(supported="ckd")

    def eval_albedo_ckd(self, bindexes: list[Bindex], zgrid: ZGrid) -> pint.Quantity:
        sigma_s = self.eval_sigma_s_ckd(bindexes, zgrid)
        sigma_t = self.eval_sigma_t_ckd(bindexes, zgrid)
        return np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        ).to("dimensionless")

    def eval_sigma_t_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        raise UnsupportedModeError(supported="ckd")

    def eval_sigma_t_ckd(self, bindexes: list[Bindex], zgrid: ZGrid) -> pint.Quantity:
        return self.eval_sigma_a_ckd(bindexes, zgrid) + self.eval_sigma_s_ckd(
            bindexes, zgrid
        )

    def eval_sigma_a_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        raise UnsupportedModeError(supported="ckd")

    def eval_sigma_a_ckd(self, bindexes: list[Bindex], zgrid: ZGrid) -> pint.Quantity:
        thermoprops = self._thermoprops_interp(zgrid)

        bin_set_ids: set[str] = {bindex.bin.bin_set_id for bindex in bindexes}
        if len(bin_set_ids) != 1:
            raise ValueError(
                f"all bindexes must be from the same bin set, got {bin_set_ids}"
            )

        bin_set_id = bin_set_ids.pop()

        if not self.has_absorption:
            return ureg.Quantity(np.zeros(thermoprops.z_layer.size), "km^-1")

        ds_id = f"{thermoprops.attrs['identifier']}-{bin_set_id}-v3"
        with _util_ckd.open_dataset(ds_id) as ds:
            # This table maps species to the function used to compute
            # corresponding physical quantities used for concentration rescaling
            compute = {
                "H2O": functools.partial(column_mass_density, species="H2O"),
                "CO2": functools.partial(volume_mixing_ratio_at_surface, species="CO2"),
                "O3": functools.partial(column_number_density, species="O3"),
            }

            # Evaluate species concentrations
            # Note: This includes a conversion to units appropriate for interpolation
            rescaled_species = list(compute.keys())
            concentrations = {
                species: compute[species](ds=thermoprops).m_as(ds[species].units)
                for species in rescaled_species
            }

            # Convert altitude to units appropriate for interpolation
            z = ureg.convert(thermoprops.z_layer, thermoprops.z_layer.units, ds.z.units)

            # Interpolate absorption coefficient on concentration coordinates
            result: pint.Quantity = [
                ds.k.sel(bd=(bindex.bin.id, bindex.index))
                .interp(
                    z=z,
                    kwargs=dict(fill_value=0.0),
                )  # extrapolate to 0.0 for altitude out of bounds
                .interp(
                    **concentrations,
                    kwargs=dict(
                        bounds_error=True
                    ),  # raise when concentration are out of bounds
                )
                .values
                for bindex in bindexes
            ] * ureg(ds.k.units)

            return result

    def eval_sigma_s_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        thermoprops = self._thermoprops_interp(zgrid)

        if self.has_scattering:
            return compute_sigma_s_air(
                wavelength=w,
                number_density=to_quantity(thermoprops.n),
            )
        else:
            return ureg.Quantity(np.zeros((1, thermoprops.z_layer.size)), "km^-1")

    def eval_sigma_s_ckd(self, bindexes: list[Bindex], zgrid: ZGrid) -> pint.Quantity:
        wavelengths = (
            np.array([bindex.bin.wcenter.m_as(ureg.nm) for bindex in bindexes])
            * ureg.nm
        )

        return self.eval_sigma_s_mono(wavelengths, zgrid)

    def eval_dataset_mono(self, w: pint.Quantity, zgrid: ZGrid) -> xr.Dataset:
        raise UnsupportedModeError(supported="ckd")

    def eval_dataset_ckd(self, bindexes: list[Bindex], zgrid: ZGrid) -> xr.Dataset:
        if len(bindexes) > 1:
            raise NotImplementedError

        else:
            return make_dataset(
                wavelength=bindexes[0].bin.wcenter,
                z_level=zgrid.levels,
                z_layer=zgrid.layers,
                sigma_a=self.eval_sigma_a_ckd(bindexes, zgrid),
                sigma_s=self.eval_sigma_s_ckd(bindexes, zgrid),
            ).squeeze()
