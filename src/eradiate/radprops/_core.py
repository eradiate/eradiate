from __future__ import annotations

import datetime
import typing as t
from abc import ABC, abstractmethod
from functools import singledispatchmethod

import attrs
import numpy as np
import pint
import xarray as xr

import eradiate

from .._factory import Factory
from ..spectral_index import CKDSpectralIndex, MonoSpectralIndex, SpectralIndex
from ..units import unit_registry as ureg

rad_profile_factory = Factory()
rad_profile_factory.register_lazy_batch(
    [
        ("_afgl1986.AFGL1986RadProfile", "afgl_1986", {}),
        ("_array.ArrayRadProfile", "array", {}),
        ("_us76_approx.US76ApproxRadProfile", "us76_approx", {}),
    ],
    cls_prefix="eradiate.radprops",
)


@ureg.wraps(
    ret=None, args=("nm", "km", "km", "km^-1", "km^-1", "km^-1", ""), strict=False
)
def make_dataset(
    wavelength: t.Union[pint.Quantity, float],
    z_level: t.Union[pint.Quantity, float],
    z_layer: t.Optional[t.Union[pint.Quantity, float]] = None,
    sigma_a: t.Optional[t.Union[pint.Quantity, float]] = None,
    sigma_s: t.Optional[t.Union[pint.Quantity, float]] = None,
    sigma_t: t.Optional[t.Union[pint.Quantity, float]] = None,
    albedo: t.Optional[t.Union[pint.Quantity, float]] = None,
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
        z_layer = (z_level[1:] + z_level[:-1]) / 2.0

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
            "'sigma_a' and 'sigma_s' or 'sigma_t' and 'albedo'."
        )

    return xr.Dataset(
        data_vars={
            "sigma_a": (
                ("w", "z_layer"),
                sigma_a.reshape(1, len(z_layer)),
                dict(
                    standard_name="absorption_coefficient",
                    units="km^-1",
                    long_name="absorption coefficient",
                ),
            ),
            "sigma_s": (
                ("w", "z_layer"),
                sigma_s.reshape(1, len(z_layer)),
                dict(
                    standard_name="scattering_coefficient",
                    units="km^-1",
                    long_name="scattering coefficient",
                ),
            ),
            "sigma_t": (
                ("w", "z_layer"),
                sigma_t.reshape(1, len(z_layer)),
                dict(
                    standard_name="extinction_coefficient",
                    units="km^-1",
                    long_name="extinction coefficient",
                ),
            ),
            "albedo": (
                ("w", "z_layer"),
                albedo.reshape(1, len(z_layer)),
                dict(
                    standard_name="albedo",
                    units="",
                    long_name="albedo",
                ),
            ),
        },
        coords={
            "z_level": (
                "z_level",
                z_level,
                dict(
                    standard_name="level_altitude",
                    units="km",
                    long_name="level altitude",
                ),
            ),
            "z_layer": (
                "z_layer",
                z_layer,
                dict(
                    standard_name="layer_altitude",
                    units="km",
                    long_name="layer altitude",
                ),
            ),
            "w": (
                "w",
                [wavelength],
                dict(standard_name="wavelength", units="nm", long_name="wavelength"),
            ),
        },
        attrs={
            "convention": "CF-1.8",
            "title": "Atmospheric monochromatic radiative properties",
            "history": f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
            f"data set creation - "
            f"{__name__}.make_dataset",
            "source": f"eradiate, version {eradiate.__version__}",
            "references": "",
        },
    )


@attrs.define
class RadProfile(ABC):
    """
    An abstract base class for radiative property profiles. Classes deriving
    from this one must implement methods which return the albedo and collision
    coefficients as Pint-wrapped Numpy arrays.

    Warnings
    --------
    Arrays returned by the :meth:`albedo`, :meth:`sigma_a`, :meth:`sigma_s`
    and :meth:`sigma_t` methods **must** be 3D. Should the profile
    be one-dimensional, invariant dimensions can be set to 1.

    See Also
    --------
    :class:`.RadProfileFactory`
    """

    @singledispatchmethod
    def eval_albedo(self, spectral_index: SpectralIndex) -> pint.Quantity:
        """Evaluate albedo at given spectral index.

        Parameters
        ----------
        spectral_index : :class:`.SpectralIndex`
            Spectral index.
        
        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        raise NotImplementedError
    
    @eval_albedo.register
    def _(self, spectral_index: MonoSpectralIndex) -> pint.Quantity:
        return self.eval_albedo_mono(spectral_index.w).squeeze()
    
    @eval_albedo.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_albedo_ckd(spectral_index.w, spectral_index.g).squeeze()

    def eval_albedo_mono(self, w: pint.Quantity) -> pint.Quantity:
        """Evaluate albedo spectrum in monochromatic modes."""
        raise NotImplementedError

    def eval_albedo_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        """Evaluate albedo spectrum in CKD modes."""
        raise NotImplementedError

    @singledispatchmethod
    def eval_sigma_t(self, spectral_index: SpectralIndex) -> pint.Quantity:
        """
        Evaluate extinction coefficient at given spectral index.

        Parameters
        ----------
        spectral_index : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        raise NotImplementedError
    
    @eval_sigma_t.register
    def _(self, spectral_index: MonoSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_t_mono(spectral_index.w).squeeze()
    
    @eval_sigma_t.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_t_ckd(spectral_index.w, spectral_index.g).squeeze()

    def eval_sigma_t_mono(self, w: pint.Quantity) -> pint.Quantity:
        """Evaluate extinction coefficient spectrum in monochromatic modes."""
        raise NotImplementedError

    def eval_sigma_t_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        """Evaluate extinction coefficient spectrum in CKD modes."""
        raise NotImplementedError

    @singledispatchmethod
    def eval_sigma_a(self, spectral_index: SpectralIndex) -> pint.Quantity:
        """
        Evaluate absorption coefficient at given spectral index.

        Parameters
        ----------
        spectral_index : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        raise NotImplementedError
    
    @eval_sigma_a.register
    def _(self, spectral_index: MonoSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_a_mono(spectral_index.w).squeeze()
    
    @eval_sigma_a.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_a_ckd(spectral_index.w, spectral_index.g).squeeze()

    def eval_sigma_a_mono(self, w: pint.Quantity) -> pint.Quantity:
        """Evaluate absorption coefficient spectrum in monochromatic modes."""
        raise NotImplementedError

    def eval_sigma_a_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        """Evaluate absorption coefficient spectrum in CKD modes."""
        raise NotImplementedError

    @singledispatchmethod
    def eval_sigma_s(self, spectral_index: SpectralIndex) -> pint.Quantity:
        """
        Evaluate scattering coefficient at given spectral index.

        Parameters
        ----------
        spectral_index : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        raise NotImplementedError

    @eval_sigma_s.register
    def _(self, spectral_index: MonoSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_s_mono(spectral_index.w).squeeze()
    
    @eval_sigma_s.register
    def _(self, spectral_index: CKDSpectralIndex) -> pint.Quantity:
        return self.eval_sigma_s_ckd(spectral_index.w, spectral_index.g).squeeze()

    def eval_sigma_s_mono(self, w: pint.Quantity) -> pint.Quantity:
        """Evaluate scattering coefficient spectrum in monochromatic modes."""
        raise NotImplementedError

    def eval_sigma_s_ckd(self, w: pint.Quantity, g: float) -> pint.Quantity:
        """Evaluate scattering coefficient spectrum in CKD modes."""
        raise NotImplementedError

    @singledispatchmethod
    def eval_dataset(self, spectral_index: SpectralIndex) -> xr.Dataset:
        """
        Evaluate radiative properties at given spectral index.

        Parameters
        ----------
        spectral_index : :class:`.SpectralIndex`
            Spectral index.

        Returns
        -------
        Dataset
            Radiative properties dataset.
        """
        raise NotImplementedError
    
    @eval_dataset.register
    def _(self, spectral_index: MonoSpectralIndex) -> xr.Dataset:
        return self.eval_dataset_mono(w=spectral_index.w)
    
    @eval_dataset.register
    def _(self, spectral_index: CKDSpectralIndex) -> xr.Dataset:
        return self.eval_dataset_ckd(w=spectral_index.w, g=spectral_index.g)

    @abstractmethod
    def eval_dataset_mono(self, w: pint.Quantity) -> xr.Dataset:
        """Evaluate radiative properties in monochromatic modes."""
        pass

    @abstractmethod
    def eval_dataset_ckd(self, w: pint.Quantity, g: float) -> xr.Dataset:
        """Evaluate radiative properties in CKD modes."""
        pass
