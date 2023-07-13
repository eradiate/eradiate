"""
Atmosphere's radiative profile.
"""
from __future__ import annotations

import attrs
import joseki
import numpy as np
import pint
import portion as P
import xarray as xr
from joseki.profiles.core import interp

from ._core import RadProfile, ZGrid, make_dataset
from .absorption import (
    DEFAULT_HANDLER_CONFIG,
    eval_sigma_a_ckd_impl,
    eval_sigma_a_mono_impl,
)
from .rayleigh import compute_sigma_s_air
from ..attrs import documented, parse_docs
from ..converters import convert_absorption_data, convert_thermoprops
from ..units import to_quantity
from ..units import unit_registry as ureg
from ..util.misc import cache_by_id, summary_repr
from ..validators import validate_absorption_data


def _absorption_data_repr(value: dict[P.Interval, xr.Dataset]) -> dict(str, str):
    def repr_k(value):
        "Representation for keys which are wavelength intervals."

        return f"{value.lower:~.3f} {value.upper:~.3f}"

    def repr_v(value):
        "Representation for values which are absorption dataset"
        return summary_repr(value)

    return "\n".join([repr_k(k) + ": " + repr_v(v) for k, v in value.items()])


@parse_docs
@attrs.define(eq=False)
class AtmosphereRadProfile(RadProfile):
    """
    Atmospheric radiative profile.

    Notes
    -----
    The radiative profile is defined by atmospheric thermophysical and
    absorption coefficient data.
    """

    absorption_data: xr.Dataset | list[xr.Dataset] = documented(
        attrs.field(
            converter=convert_absorption_data,
            validator=validate_absorption_data,
            repr=_absorption_data_repr,
        ),
        doc="Absorption coefficient data. "
        "If a file path, the absorption coefficient is loaded from the "
        "specified file (must be a NetCDF file)."
        "If a tuple, the first element is the dataset codename and the"
        "second is the desired working wavelength range.",
        type="Dataset or list of Dataset",
        init_type="Dataset or list of Dataset or :class:`.PathLike`",
    )

    thermoprops: xr.Dataset = documented(
        attrs.field(
            factory=lambda: joseki.make(
                identifier="afgl_1986-us_standard",
                z=np.linspace(0.0, 120.0, 121) * ureg.km,
                additional_molecules=False,
            ),
            converter=convert_thermoprops,
            repr=summary_repr,
        ),
        doc="Atmosphere's thermophysical properties.",
        type="Dataset",
        default=":meth:`joseki.make() <joseki.make>` with "
        "``identifier='afgl_1986-us_standard'`` and "
        "``z=np.linspace(0, 120, 61) * ureg.km``"
        "``additional_moleculs=False``.",
    )

    @thermoprops.validator
    def _check_thermoprops(self, attribute, value):
        if not value.joseki.is_valid:
            raise ValueError(
                f"Invalid thermophysical properties dataset."  # TODO: explain what is invalid
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

    error_handler_config: dict[str, dict[str, str]] = documented(
        attrs.field(
            factory=lambda: DEFAULT_HANDLER_CONFIG,
            validator=attrs.validators.deep_mapping(
                key_validator=attrs.validators.instance_of(str),
                value_validator=attrs.validators.deep_mapping(
                    key_validator=attrs.validators.instance_of(str),
                    value_validator=attrs.validators.instance_of(str),
                ),
            ),
        ),
        doc="Error handler configuration for absorption data interpolation.",
        type="dict",
        default=DEFAULT_HANDLER_CONFIG,
    )

    def __attrs_post_init__(self):
        self.update()

    def update(self) -> None:
        self._zgrid = ZGrid(levels=self.levels)

    @property
    def levels(self) -> pint.Quantity:
        return to_quantity(self.thermoprops.z)

    @property
    def zgrid(self) -> ZGrid:
        # Inherit docstring
        return self._zgrid

    @cache_by_id
    def _thermoprops_interp(self, zgrid: ZGrid) -> xr.Dataset:
        # Interpolate thermophysical profile on specified altitude grid
        # Note: this value is cached so that repeated calls with the same zgrid
        #       won't trigger an unnecessary computation.
        return interp(
            self.thermoprops,
            z_new=zgrid.levels.m * ureg(str(zgrid.levels.units)),
            method={"default": "nearest"},  # TODO: revisit
        )

    def eval_albedo_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        # Inherit docstring
        sigma_s = self.eval_sigma_s_mono(w, zgrid)
        sigma_t = self.eval_sigma_t_mono(w, zgrid)
        return np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        ).to("dimensionless")

    def eval_albedo_ckd(
        self,
        w: pint.Quantity,
        g: float,
        zgrid: ZGrid,
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
        if self.has_absorption:
            values = eval_sigma_a_mono_impl(  # values at altitude levels
                self.absorption_data,
                thermoprops=self._thermoprops_interp(zgrid),
                w=w,
                error_handler_config=self.error_handler_config,
            )  # axis order = (w, z)
            # project on altitude layers
            return 0.5 * (values[:, 1:] + values[:, :-1]).squeeze()
        else:
            return np.zeros((w.size, zgrid.n_layers)).squeeze() / ureg.km

    def eval_sigma_a_ckd(
        self,
        w: pint.Quantity,
        g: float,
        zgrid: ZGrid,
    ) -> pint.Quantity:
        # NOTE: this method accepts 'w'-arrays and is vectorized as far as
        # each individual absorption dataset is concerned, namely when the
        # wavelengths span multiple datasets we for-loop over them.
        w = np.atleast_1d(w)
        if self.has_absorption:
            values = eval_sigma_a_ckd_impl(  # values at altitude levels
                self.absorption_data,
                thermoprops=self._thermoprops_interp(zgrid),
                w=w,
                g=g,
                error_handler_config=self.error_handler_config,
            )  # axis order = (w, z)
            # project on altitude layers
            return 0.5 * (values[:, 1:] + values[:, :-1]).squeeze()
        else:
            return np.zeros((w.size, zgrid.n_layers)).squeeze() / ureg.km

    def eval_sigma_s_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:

        if self.has_scattering:
            thermoprops = self._thermoprops_interp(zgrid)
            sigma_s = compute_sigma_s_air(
                wavelength=w,
                number_density=to_quantity(thermoprops.n),
            )
            # project on altitude layers
            return 0.5 * (sigma_s[1:] + sigma_s[:-1])
        else:
            return np.zeros((1, zgrid.n_layers)) / ureg.km

    def eval_sigma_s_ckd(
        self,
        w: pint.Quantity,
        g: float,
        zgrid: ZGrid,
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
