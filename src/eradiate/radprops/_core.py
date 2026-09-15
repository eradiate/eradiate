from __future__ import annotations

from abc import ABC, abstractmethod
from functools import singledispatchmethod

import attrs
import numpy as np
import pint
import xarray as xr

import eradiate

from ..grid import GridCoords
from ..spectral.index import CKDSpectralIndex, MonoSpectralIndex, SpectralIndex
from ..units import unit_registry as ureg
from ..util.misc import get_utcnow


@ureg.wraps(
    ret=None, args=("nm", "km", "km", "km^-1", "km^-1", "km^-1", ""), strict=False
)
def make_dataset(
    wavelength: pint.Quantity | float,
    z_level: pint.Quantity | float,
    z_layer: pint.Quantity | float | None = None,
    sigma_a: pint.Quantity | float | None = None,
    sigma_s: pint.Quantity | float | None = None,
    sigma_t: pint.Quantity | float | None = None,
    albedo: pint.Quantity | float | None = None,
) -> xr.Dataset:
    """
    Makes an atmospheric radiative properties data set.

    Parameters
    ----------
    wavelength : float
        Wavelength [nm].

    z_level : array
        Level altitudes [km].

    z_layer : array
        Layer altitudes [km].

        If ``None``, the layer altitudes are computed automatically, so that
        they are halfway between the adjacent altitude levels.

    sigma_a : array
        Absorption coefficient values [km^-1].

    sigma_s : array
        Scattering coefficient values [km^-1].

    sigma_t : array
        Extinction coefficient values [km^-1].

    albedo : array
        Albedo values [/].

    Returns
    -------
    Dataset
        Atmosphere radiative properties data set.
    """
    if z_layer is None:
        z_layer = 0.5 * (z_level[1:] + z_level[:-1])

    if sigma_a is not None and sigma_s is not None:
        sigma_t = sigma_a + sigma_s
        albedo = np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        )
    elif sigma_t is not None and albedo is not None:
        sigma_s = albedo * sigma_t
        sigma_a = sigma_t - sigma_s
    else:
        raise ValueError(
            "You must provide either one of the two pairs of arguments "
            "('sigma_a', 'sigma_s') or ('sigma_t', 'albedo')."
        )

    # sigma_a/sigma_s/sigma_t/albedo are shaped either (z_layer,) or, for a
    # 3D atmosphere, (x, y, z_layer)
    horizontal_shape = np.shape(sigma_a)[:-1]
    dims = ("w",) + (("x", "y") if horizontal_shape else ()) + ("z_layer",)
    shape = (1,) + horizontal_shape + (z_layer.size,)

    coords = {
        "z_level": (
            "z_level",
            z_level,
            {
                "standard_name": "level_altitude",
                "units": "km",
                "long_name": "level altitude",
            },
        ),
        "z_layer": (
            "z_layer",
            z_layer,
            {
                "standard_name": "layer_altitude",
                "units": "km",
                "long_name": "layer altitude",
            },
        ),
        "w": (
            "w",
            [wavelength],
            {
                "standard_name": "radiation_wavelength",
                "units": "nm",
                "long_name": "wavelength",
            },
        ),
    }
    if horizontal_shape:
        coords["x"] = ("x", np.arange(horizontal_shape[0]))
        coords["y"] = ("y", np.arange(horizontal_shape[1]))

    return xr.Dataset(
        data_vars={
            "sigma_a": (
                dims,
                sigma_a.reshape(shape),
                {
                    "standard_name": "absorption_coefficient",
                    "units": "km^-1",
                    "long_name": "absorption coefficient",
                },
            ),
            "sigma_s": (
                dims,
                sigma_s.reshape(shape),
                {
                    "standard_name": "scattering_coefficient",
                    "units": "km^-1",
                    "long_name": "scattering coefficient",
                },
            ),
            "sigma_t": (
                dims,
                sigma_t.reshape(shape),
                {
                    "standard_name": "extinction_coefficient",
                    "units": "km^-1",
                    "long_name": "extinction coefficient",
                },
            ),
            "albedo": (
                dims,
                albedo.reshape(shape),
                {
                    "standard_name": "albedo",
                    "units": "",
                    "long_name": "albedo",
                },
            ),
        },
        coords=coords,
        attrs={
            "convention": "CF-1.8",
            "title": "Atmospheric monochromatic radiative properties",
            "history": f"{get_utcnow().replace(microsecond=0)} - "
            f"data set creation - "
            f"{__name__}.make_dataset",
            "source": f"eradiate, version {eradiate.__version__}",
            "references": "",
        },
    )


@attrs.define(eq=False)
class RadProfile(ABC):
    """
    An abstract base class for radiative property profiles. Classes deriving
    from this one must implement methods which return the albedo and collision
    coefficients as Pint-wrapped Numpy arrays.
    """

    @property
    @abstractmethod
    def zbounds(self) -> tuple[pint.Quantity, pint.Quantity]:
        """
        Bounds of the z profile.
        """

    @property
    @abstractmethod
    def grid(self) -> GridCoords:
        """
        Default altitude grid used for profile evaluation.
        """

    @singledispatchmethod
    def eval_albedo(
        self,
        si: SpectralIndex,
        grid: GridCoords | None = None,
    ) -> pint.Quantity:
        """
        Evaluate albedo at given spectral index.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .GridCoords, optional
            The altitude grid for which the albedo is evaluated. If unset, a
            profile-specific default is used.

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        raise NotImplementedError

    @eval_albedo.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords) -> pint.Quantity:
        return self.eval_albedo_mono(w=si.w, grid=grid)

    @eval_albedo.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords) -> pint.Quantity:
        return self.eval_albedo_ckd(w=si.w, g=si.g, grid=grid)

    def eval_albedo_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        raise NotImplementedError

    def eval_albedo_ckd(
        self,
        w: pint.Quantity,
        g: float,
        grid: GridCoords,
    ) -> pint.Quantity:
        raise NotImplementedError

    @singledispatchmethod
    def eval_sigma_t(
        self,
        si: SpectralIndex,
        grid: GridCoords | None = None,
    ) -> pint.Quantity:
        """
        Evaluate extinction coefficient at given spectral index.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .GridCoords, optional
            The altitude grid for which the extinction coefficient is evaluated.
            If unset, a profile-specific default is used.

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        raise NotImplementedError

    @eval_sigma_t.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords) -> pint.Quantity:
        return self.eval_sigma_t_mono(w=si.w, grid=grid)

    @eval_sigma_t.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords) -> pint.Quantity:
        return self.eval_sigma_t_ckd(w=si.w, g=si.g, grid=grid)

    def eval_sigma_t_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        raise NotImplementedError

    def eval_sigma_t_ckd(
        self,
        w: pint.Quantity,
        g: float,
        grid: GridCoords,
    ) -> pint.Quantity:
        raise NotImplementedError

    @singledispatchmethod
    def eval_sigma_a(
        self,
        si: SpectralIndex,
        grid: GridCoords | None = None,
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient at given spectral index.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        grid : .GridCoords, optional
            The altitude grid for which the absorption coefficient is evaluated.
            If unset, a profile-specific default is used.

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        raise NotImplementedError

    @eval_sigma_a.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords) -> pint.Quantity:
        return self.eval_sigma_a_mono(w=si.w, grid=grid)

    @eval_sigma_a.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords) -> pint.Quantity:
        return self.eval_sigma_a_ckd(w=si.w, g=si.g, grid=grid)

    def eval_sigma_a_mono(
        self,
        w: pint.Quantity,
        grid: GridCoords,
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient spectrum in monochromatic mode.
        """
        raise NotImplementedError

    def eval_sigma_a_ckd(
        self,
        w: pint.Quantity,
        g: float,
        grid: GridCoords,
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient spectrum in CKD modes.
        """
        raise NotImplementedError

    @singledispatchmethod
    def eval_sigma_s(
        self,
        si: SpectralIndex,
        grid: GridCoords | None = None,
    ) -> pint.Quantity:
        """
        Evaluate scattering coefficient at given spectral index.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        grid : .GridCoords, optional
            The altitude grid for which the scattering coefficient is evaluated.
            If unset, a profile-specific default is used.

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        raise NotImplementedError

    @eval_sigma_s.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords) -> pint.Quantity:
        return self.eval_sigma_s_mono(w=si.w, grid=grid)

    @eval_sigma_s.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords) -> pint.Quantity:
        return self.eval_sigma_s_ckd(w=si.w, g=si.g, grid=grid)

    def eval_sigma_s_mono(self, w: pint.Quantity, grid: GridCoords) -> pint.Quantity:
        raise NotImplementedError

    def eval_sigma_s_ckd(
        self, w: pint.Quantity, g: float, grid: GridCoords
    ) -> pint.Quantity:
        raise NotImplementedError

    @singledispatchmethod
    def eval_depolarization_factor(
        self,
        si: SpectralIndex,
        grid: GridCoords | None = None,
    ) -> pint.Quantity:
        """
        Evaluate depolarization factor at given spectral index.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        grid : .GridCoords, optional
            The altitude grid for which the depolarization factor is evaluated.
            If unset, a profile-specific default is used.

        Returns
        -------
        quantity
            Evaluated depolarization factor as an array with length equal
            to the number of layers if it is parametrized over layers,
            otherwise as an array of length 1.
        """
        raise NotImplementedError

    @eval_depolarization_factor.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords) -> pint.Quantity:
        return self.eval_depolarization_factor_mono(w=si.w, grid=grid)

    @eval_depolarization_factor.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords) -> pint.Quantity:
        return self.eval_depolarization_factor_ckd(w=si.w, g=si.g, grid=grid)

    def eval_depolarization_factor_mono(
        self, w: pint.Quantity, grid: GridCoords
    ) -> pint.Quantity:
        raise NotImplementedError

    def eval_depolarization_factor_ckd(
        self, w: pint.Quantity, g: float, grid: GridCoords
    ) -> pint.Quantity:
        raise NotImplementedError

    @singledispatchmethod
    def eval_dataset(
        self,
        si: SpectralIndex,
        grid: GridCoords | None = None,
    ) -> xr.Dataset:
        """
        Evaluate radiative properties at given spectral index.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        grid : .GridCoords, optional
            The altitude grid for which the radiative profile is evaluated.
            If unset, a profile-specific default is used.

        Returns
        -------
        Dataset
            Radiative property dataset.
        """
        raise NotImplementedError

    @eval_dataset.register(MonoSpectralIndex)
    def _(self, si, grid: GridCoords) -> xr.Dataset:
        return self.eval_dataset_mono(w=si.w, grid=grid)

    @eval_dataset.register(CKDSpectralIndex)
    def _(self, si, grid: GridCoords) -> xr.Dataset:
        return self.eval_dataset_ckd(w=si.w, g=si.g, grid=grid)

    def eval_dataset_mono(self, w: pint.Quantity, grid: GridCoords) -> xr.Dataset:
        raise NotImplementedError

    def eval_dataset_ckd(
        self,
        w: pint.Quantity,
        g: float,
        grid: GridCoords,
    ) -> xr.Dataset:
        raise NotImplementedError
