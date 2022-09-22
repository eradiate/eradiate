from __future__ import annotations

import pathlib
import typing as t

import attrs
import pint
import pinttr
import xarray as xr

import eradiate

from ._core import RadProfile, make_dataset
from .. import validators
from ..attrs import documented, parse_docs
from ..exceptions import UnsupportedModeError
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg


@parse_docs
@attrs.define
class ArrayRadProfile(RadProfile):
    """
    A flexible 1D radiative property profile whose level altitudes, albedo
    and extinction coefficient are specified as numpy arrays.
    """

    levels: pint.Quantity = documented(
        pinttr.field(units=ucc.deferred("length")),
        doc="Level altitudes. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="array",
    )

    albedo_values: pint.Quantity = documented(
        pinttr.field(
            validator=[validators.all_positive, pinttr.validators.has_compatible_units],
            units=ureg.dimensionless,
        ),
        doc="An array specifying albedo values. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (dimensionless).",
        type="array",
    )

    sigma_t_values: pint.Quantity = documented(
        pinttr.field(
            validator=[validators.all_positive, pinttr.validators.has_compatible_units],
            units=ucc.deferred("collision_coefficient"),
        ),
        doc="An array specifying extinction coefficient values. **Required, no "
        "default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['collision_coefficient']).",
        type="array",
    )

    @albedo_values.validator
    @sigma_t_values.validator
    def _validator_values(instance, attribute, value):
        if value.ndim != 1:
            raise ValueError(
                f"while setting {attribute.name}: "
                f"must have 1 dimension only "
                f"(got shape {value.shape})"
            )

        if instance.albedo_values.shape != instance.sigma_t_values.shape:
            raise ValueError(
                f"while setting {attribute.name}: "
                f"'albedo_values' and 'sigma_t_values' must have "
                f"the same length"
            )

    def __attrs_pre_init__(self):
        if not eradiate.mode().is_mono:
            raise UnsupportedModeError(supported="monochromatic")

    def eval_albedo_mono(self, w: pint.Quantity) -> pint.Quantity:
        return self.albedo_values

    def eval_sigma_t_mono(self, w: pint.Quantity) -> pint.Quantity:
        return self.sigma_t_values

    def eval_sigma_a_mono(self, w: pint.Quantity) -> pint.Quantity:
        return self.eval_sigma_t_mono(w) * (1.0 - self.eval_albedo_mono(w))

    def eval_sigma_s_mono(self, w: pint.Quantity) -> pint.Quantity:
        return self.eval_sigma_t_mono(w) * self.eval_albedo_mono(w)

    def eval_dataset_mono(self, w: pint.Quantity) -> xr.Dataset:
        return make_dataset(
            wavelength=w,
            z_level=self.levels,
            sigma_t=self.eval_sigma_t_mono(w),
            albedo=self.eval_albedo_mono(w),
        ).squeeze()

    @classmethod
    def from_dataset(cls, path: t.Union[str, pathlib.Path]) -> ArrayRadProfile:
        try:
            ds = xr.open_dataset(path)
        except FileNotFoundError:
            ds = eradiate.data.data_store.open_dataset(path)

        z_level = to_quantity(ds.z_level)
        albedo = to_quantity(ds.albedo)
        sigma_t = to_quantity(ds.sigma_t)
        ds.close()
        return cls(albedo_values=albedo, sigma_t_values=sigma_t, levels=z_level)
