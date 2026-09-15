"""
Atmosphere's radiative profile.
"""

from __future__ import annotations

import attrs
import numpy as np
import pint
import xarray as xr
from axsdb import AbsorptionDatabase
from joseki.profiles.core import interp

from ._absorption import get_default_absdb
from ._core import RadProfile, make_dataset
from .rayleigh import compute_sigma_s_air, depolarization_bates, depolarization_bodhaine
from .. import converters
from ..attrs import define, documented
from ..grid import GridCoords
from ..units import to_quantity
from ..units import unit_registry as ureg
from ..util.misc import cache_by_id, summary_repr

_THERMOPROPS_DEFAULT = {
    "identifier": "afgl_1986-us_standard",
    "z": np.linspace(0.0, 120.0, 121) * ureg.km,
    "additional_molecules": False,
}


def _validate_thermoprops_grid_shape(
    thermoprops: xr.Dataset, grid: GridCoords
) -> xr.Dataset:
    """Check any 'x'/'y' dimensions of thermoprops against the render grid."""
    horizontal_dims = set(thermoprops.dims) & {"x", "y"}
    if not horizontal_dims:
        return thermoprops

    result = thermoprops.transpose(..., "z")
    expected_sizes = {"x": grid.n_cells_x, "y": grid.n_cells_y}
    for dim in horizontal_dims:
        if result.sizes[dim] != expected_sizes[dim]:
            raise ValueError(
                f"thermoprops dimension '{dim}' has size {result.sizes[dim]}, "
                f"expected {expected_sizes[dim]} to match the render grid"
            )

    return result


def _average_adjacent_levels(values):
    """
    Average adjacent altitude levels (the last axis) into per-layer values,
    then drop the leading axis (the wavelength axis) if it has size 1.
    """
    result = 0.5 * (values[..., 1:] + values[..., :-1])
    return result[0] if result.shape[0] == 1 else result


def _layers_from_levels(values: xr.DataArray) -> pint.Quantity:
    """
    Average adjacent altitude levels into per-layer values. ``values`` is
    transposed so that ``w`` comes first and ``z`` last, whatever other
    dimensions (e.g. ``x``, ``y``) sit in between and in whatever order they
    originally came in.
    """
    return _average_adjacent_levels(to_quantity(values.transpose("w", ..., "z")))


@define(eq=False)
class AtmosphereRadProfile(RadProfile):
    """
    Atmospheric radiative profile.

    This class provides an interface to generate vertical profiles of
    atmospheric volume radiative properties (also sometimes referred to as
    collision coefficients).

    The atmospheric radiative profile is built from a thermophysical profile,
    which provides the temperature, pressure and species concentrations as a
    function of altitude, and an absorption coefficient database indexed by
    those thermophysical variables.
    """

    absorption_data: AbsorptionDatabase = documented(
        attrs.field(
            factory=get_default_absdb,
            converter=converters.convert_absdb,
            validator=attrs.validators.instance_of(AbsorptionDatabase),
        ),
        doc="Absorption coefficient data. The passed value is pre-processed by "
        ":meth:`.AbsorptionDatabase.convert`.",
        type="AbsorptionDatabase",
        init_type="str or path-like or dict or .AbsorptionDatabase",
        default=":meth:`AbsorptionDatabase.default() <.AbsorptionDatabase.default>`",
    )

    thermoprops: xr.Dataset = documented(
        attrs.field(
            default=_THERMOPROPS_DEFAULT,
            converter=converters.convert_thermoprops,
            repr=summary_repr,
        ),
        doc="Thermophysical property dataset. If a path is passed, Eradiate will "
        "look it up and load it. If a dictionary is passed, it will be passed "
        "as keyword argument to ``joseki.make()``. The default is "
        '``{"identifier": "afgl_1986-us_standard",  "z": np.linspace(0, 120, 121) * ureg.km), "additional_molecules": False}``. '
        "See the `Joseki documentation <https://joseki.readthedocs.io/en/latest/reference/joseki/index.html#joseki.make>`__ "
        "for details.",
        type="Dataset",
        init_type="Dataset or path-like or dict",
    )

    @thermoprops.validator
    def _check_thermoprops(self, attribute, value):
        if not value.joseki.is_valid:
            raise ValueError(
                "Invalid thermophysical properties dataset."  # TODO: explain what is invalid
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

    rayleigh_depolarization: np.ndarray | str = documented(
        attrs.field(
            converter=lambda x: (
                x if isinstance(x, str) else np.array(x, dtype=np.float64)
            ),
            kw_only=True,
            factory=lambda: np.array(0.0),
        ),
        type='ndarray or {"bates", "bodhaine"}',
        doc="Depolarization factor of the rayleigh phase function. "
        "``str`` will be interpreted as the name of the function used to "
        "calculate the depolarization factor from atmospheric properties. "
        "A ``ndarray`` will be interpreted as a description of the depolarization "
        "factor at different levels of the atmosphere. Must be shaped (N,) with "
        "N the number of layers.",
        init_type="array-like or str, optional",
        default="[0]",
    )

    _grid: GridCoords | None = attrs.field(default=None, init=False)

    def __attrs_post_init__(self):
        self.update()

    def update(self) -> None:
        self._grid = GridCoords.make_default()

    @property
    def zbounds(self) -> tuple[pint.Quantity, pint.Quantity]:
        z = to_quantity(self.thermoprops.z)
        return tuple(z[[0, -1]])

    @property
    def levels(self) -> pint.Quantity:
        return to_quantity(self.thermoprops.z)

    @property
    def grid(self) -> GridCoords:
        # Inherit docstring
        return self._grid

    @cache_by_id
    def _thermoprops_interp(self, grid: GridCoords) -> xr.Dataset:
        # Interpolate thermophysical profile on specified altitude grid
        # Note: this value is cached so that repeated calls with the same grid
        #       won't trigger an unnecessary computation.
        result = interp(
            self.thermoprops,
            z_new=grid.levels.m * ureg(str(grid.levels.units)),
            method={"default": "nearest"},  # TODO: revisit
        )
        return _validate_thermoprops_grid_shape(result, grid)

    def eval_albedo_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        # Inherit docstring
        sigma_s = self.eval_sigma_s_mono(w, grid)
        sigma_t = self.eval_sigma_t_mono(w, grid)
        return np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        ).to("dimensionless")

    def eval_albedo_ckd(
        self, w: pint.Quantity, g: float, grid: GridCoords
    ) -> pint.Quantity:
        sigma_s = self.eval_sigma_s_ckd(w=w, g=g, grid=grid)
        sigma_t = self.eval_sigma_t_ckd(w=w, g=g, grid=grid)
        return np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        ).to("dimensionless")

    def eval_sigma_a_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        w = np.atleast_1d(w)
        if self.has_absorption:
            thermoprops = self._thermoprops_interp(grid)
            values = self.absorption_data.eval_sigma_a_mono(w, thermoprops)
            return _layers_from_levels(values)
        else:
            return np.zeros((w.size, grid.n_layers)).squeeze() / ureg.km

    def eval_sigma_a_ckd(
        self, w: pint.Quantity, g: float, grid: GridCoords
    ) -> pint.Quantity:
        w = np.atleast_1d(w)
        if self.has_absorption:
            values = self.absorption_data.eval_sigma_a_ckd(
                w, g, self._thermoprops_interp(grid)
            )
            return _layers_from_levels(values)
        else:
            return np.zeros((w.size, grid.n_layers)).squeeze() / ureg.km

    def eval_sigma_s_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        if self.has_scattering:
            w = np.atleast_1d(w)
            thermoprops = self._thermoprops_interp(grid)
            sigma_s = compute_sigma_s_air(
                wavelength=w,
                number_density=to_quantity(thermoprops.n),
            )
            # project on altitude layers
            return _average_adjacent_levels(sigma_s)
        else:
            return np.zeros((1, grid.n_layers)) / ureg.km

    def eval_sigma_s_ckd(
        self, w: pint.Quantity, g: float, grid: GridCoords
    ) -> pint.Quantity:
        return self.eval_sigma_s_mono(w=w, grid=grid)

    def eval_sigma_t_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        sigma_a = self.eval_sigma_a_mono(w=w, grid=grid)
        sigma_s = self.eval_sigma_s_mono(w=w, grid=grid)
        return sigma_a + sigma_s

    def eval_sigma_t_ckd(
        self,
        w: pint.Quantity,
        g: float,
        grid: GridCoords,
    ) -> pint.Quantity:
        sigma_a = self.eval_sigma_a_ckd(w=w, g=g, grid=grid)
        sigma_s = self.eval_sigma_s_ckd(w=w, g=g, grid=grid)
        return sigma_a + sigma_s

    def eval_dataset_mono(self, w: pint.Quantity, grid: GridCoords) -> xr.Dataset:
        return make_dataset(
            wavelength=w,
            z_level=grid.levels,
            z_layer=grid.layers,
            sigma_a=self.eval_sigma_a_mono(w, grid),
            sigma_s=self.eval_sigma_s_mono(w, grid),
        ).squeeze()

    def eval_dataset_ckd(
        self,
        w: pint.Quantity,
        g: float,
        grid: GridCoords,
    ) -> xr.Dataset:
        return make_dataset(
            wavelength=w,
            z_level=grid.levels,
            z_layer=grid.layers,
            sigma_a=self.eval_sigma_a_ckd(w=w, g=g, grid=grid),
            sigma_s=self.eval_sigma_s_ckd(w=w, g=g, grid=grid),
        ).squeeze()

    def eval_depolarization_factor_mono(
        self, w: pint.Quantity, grid: GridCoords
    ) -> pint.Quantity:
        if self.has_scattering:
            if isinstance(self.rayleigh_depolarization, np.ndarray):
                return np.atleast_1d(self.rayleigh_depolarization) * ureg.dimensionless

            elif isinstance(self.rayleigh_depolarization, str):
                if self.rayleigh_depolarization == "bates":
                    return depolarization_bates(wavelength=w)

                elif self.rayleigh_depolarization == "bodhaine":
                    thermoprops = self._thermoprops_interp(grid)
                    depol = depolarization_bodhaine(
                        wavelength=w,
                        x_CO2=to_quantity(thermoprops.x_CO2),
                    )
                    return 0.5 * (depol[..., 1:] + depol[..., :-1])
                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

        else:
            return np.atleast_1d(0.0) * ureg.dimensionless

    def eval_depolarization_factor_ckd(
        self,
        w: pint.Quantity,
        g: float,
        grid: GridCoords,
    ) -> pint.Quantity:
        return self.eval_depolarization_factor_mono(w=w, grid=grid)
