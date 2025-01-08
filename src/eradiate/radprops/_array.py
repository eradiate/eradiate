"""
Array radiative profile.
"""

from __future__ import annotations

import typing as t
import warnings

import attrs
import numpy as np
import pint
import xarray as xr

from ._core import RadProfile, ZGrid, make_dataset
from ..attrs import define, documented
from ..units import to_quantity
from ..units import unit_registry as ureg


@define(eq=False)
class ArrayRadProfile(RadProfile):
    """
    Array radiative profile.

    This class provides an interface to generate vertical profiles of
    atmospheric volume radiative properties (also sometimes referred to as
    collision coefficients).

    The array radiative profile is built from absorption and scattering
    coefficient data provided as input. This can be useful for debugging and
    benchmarking.
    """

    sigma_a: xr.DataArray | None = documented(
        attrs.field(
            default=None,
        ),
        doc="``DataArray`` of absorption coefficients. "
        "The ``DataArray`` is composed of two dimensions ``{w, z}`` representing the "
        "wavelength and altitude respectively. Note that ``w`` must be of length "
        "2 minimum to be correctly interpolated.",
        type="DataArray or None",
        init_type="DataArray or None",
        default="None",
    )

    sigma_s: xr.DataArray | None = documented(
        attrs.field(
            default=None,
        ),
        doc="``DataArray`` of scattering coefficients. "
        "The ``DataArray`` is composed of two dimensions ``{w, z}`` representing the "
        "wavelength and altitude respectively. Note that ``w`` must be of length "
        "2 minimum to be correctly interpolated.",
        type="DataArray or None",
        init_type="DataArray or None",
        default="None",
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

    rayleigh_depolarization: np.ndarray = documented(
        attrs.field(
            converter=lambda x: np.array(x, dtype=np.float64),
            kw_only=True,
            factory=lambda: np.array(0.0),
        ),
        type="ndarray",
        doc="Depolarization factor of the rayleigh phase function. "
        "A ``ndarray`` will be interpreted as a description of the depolarization "
        "factor at different levels of the atmosphere. Must be shaped (N,) with "
        "N the number of layers.",
        init_type="array-like, optional",
        default="[0]",
    )

    interpolation_method: t.Literal["nearest", "linear"] = documented(
        attrs.field(
            default="nearest",
            converter=str,
            validator=attrs.validators.in_(["nearest", "linear"]),
        ),
        doc="Method of interpolation of the absorption and scattering coefficients.",
        type="str",
        init_type="{'nearest', 'linear'}",
        default="nearest",
    )

    interpolation_kwargs: dict[str, t.Any] = documented(
        attrs.field(
            factory=dict,
            converter=dict,
            validator=attrs.validators.instance_of(dict),
        ),
        doc="Keyword arguments passed to :meth:`xarray.DataArray.interp` when called.",
        type="dict",
        init_type="dict, optional",
    )

    def __attrs_post_init__(self):
        self.update()

    def update(self) -> None:
        pass

    @property
    def zbounds(self) -> tuple[pint.Quantity, pint.Quantity]:
        z_max = np.inf
        z_min = -1.0
        if self.sigma_s is not None and self.has_scattering:
            z_sigma_s = to_quantity(self.sigma_s.z)
            z_max = pint.Quantity(z_max, z_sigma_s.units)
            z_min = pint.Quantity(z_min, z_sigma_s.units)
            z_max = min(z_max, z_sigma_s.max())
            z_min = max(z_min, z_sigma_s.min())

        if self.sigma_a is not None and self.has_absorption:
            z_sigma_a = to_quantity(self.sigma_a.z)
            z_max = pint.Quantity(z_max, z_sigma_a.units)
            z_min = pint.Quantity(z_min, z_sigma_a.units)
            z_max = min(z_max, z_sigma_a.max())
            z_min = max(z_min, z_sigma_a.min())

        return z_min, z_max

    @property
    def levels(self) -> pint.Quantity:
        NotImplementedError

    @property
    def zgrid(self) -> ZGrid:
        NotImplementedError

    def eval_albedo_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        # Inherit docstring
        sigma_s = self.eval_sigma_s_mono(w, zgrid)
        sigma_t = self.eval_sigma_t_mono(w, zgrid)
        return np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        ).to("dimensionless")

    def eval_albedo_ckd(
        self, w: pint.Quantity, g: float, zgrid: ZGrid
    ) -> pint.Quantity:
        sigma_s = self.eval_sigma_s_ckd(w=w, g=g, zgrid=zgrid)
        sigma_t = self.eval_sigma_t_ckd(w=w, g=g, zgrid=zgrid)
        return np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        ).to("dimensionless")

    def eval_sigma_a_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        # NOTE: this method accepts 'w'-arrays and is vectorized as far as
        # each individual absorption dataset is concerned, namely when the
        # wavelengths span multiple datasets we for-loop over them.
        w = np.atleast_1d(w)
        if self.has_absorption and self.sigma_a is not None:
            values = self.sigma_a.interp(
                coords={
                    "z": zgrid.layers.m_as(self.sigma_a.z.attrs["units"]),
                    "w": w.m_as(self.sigma_a.w.attrs["units"]),
                },
                method=self.interpolation_method,
                kwargs=self.interpolation_kwargs,
            )

            values = to_quantity(values)
            return values.squeeze()
        else:
            return np.zeros((w.size, zgrid.n_layers)).squeeze() / ureg.km

    def eval_sigma_a_ckd(
        self, w: pint.Quantity, g: float, zgrid: ZGrid
    ) -> pint.Quantity:
        warnings.warn(
            "You are using ArrayRadProps in CKD mode, this is not standard "
            "behaviour and might provide erroneous results."
        )
        return self.eval_sigma_a_mono(w=w, zgrid=zgrid)

    def eval_sigma_s_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        w = np.atleast_1d(w)
        if self.has_scattering and self.sigma_s is not None:
            sigma_s = self.sigma_s.interp(
                coords={
                    "z": zgrid.layers.m_as(self.sigma_s.z.attrs["units"]),
                    "w": w.m_as(self.sigma_s.w.attrs["units"]),
                },
                method=self.interpolation_method,
                kwargs=self.interpolation_kwargs,
            )

            sigma_s = to_quantity(sigma_s)
            return sigma_s.squeeze()
        else:
            return np.zeros((1, zgrid.n_layers)) / ureg.km

    def eval_sigma_s_ckd(
        self, w: pint.Quantity, g: float, zgrid: ZGrid
    ) -> pint.Quantity:
        return self.eval_sigma_s_mono(w=w, zgrid=zgrid)

    def eval_sigma_t_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        sigma_a = self.eval_sigma_a_mono(w=w, zgrid=zgrid)
        sigma_s = self.eval_sigma_s_mono(w=w, zgrid=zgrid)
        return sigma_a + sigma_s

    def eval_sigma_t_ckd(
        self,
        w: pint.Quantity,
        g: float,
        zgrid: ZGrid,
    ) -> pint.Quantity:
        sigma_a = self.eval_sigma_a_ckd(w=w, g=g, zgrid=zgrid)
        sigma_s = self.eval_sigma_s_ckd(w=w, g=g, zgrid=zgrid)
        return sigma_a + sigma_s

    def eval_dataset_mono(self, w: pint.Quantity, zgrid: ZGrid) -> xr.Dataset:
        return make_dataset(
            wavelength=w,
            z_level=zgrid.levels,
            z_layer=zgrid.layers,
            sigma_a=self.eval_sigma_a_mono(w, zgrid),
            sigma_s=self.eval_sigma_s_mono(w, zgrid),
        ).squeeze()

    def eval_dataset_ckd(
        self,
        w: pint.Quantity,
        g: float,
        zgrid: ZGrid,
    ) -> xr.Dataset:
        return make_dataset(
            wavelength=w,
            z_level=zgrid.levels,
            z_layer=zgrid.layers,
            sigma_a=self.eval_sigma_a_ckd(w=w, g=g, zgrid=zgrid),
            sigma_s=self.eval_sigma_s_ckd(w=w, g=g, zgrid=zgrid),
        ).squeeze()

    def eval_depolarization_factor_mono(
        self, w: pint.Quantity, zgrid: ZGrid
    ) -> pint.Quantity:
        if self.has_scattering:
            if isinstance(self.rayleigh_depolarization, np.ndarray):
                return np.atleast_1d(self.rayleigh_depolarization) * ureg.dimensionless
            else:
                raise NotImplementedError

        else:
            return np.atleast_1d(0.0) * ureg.dimensionless

    def eval_depolarization_factor_ckd(
        self,
        w: pint.Quantity,
        g: float,
        zgrid: ZGrid,
    ) -> pint.Quantity:
        return self.eval_depolarization_factor_mono(w=w, zgrid=zgrid)
