from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from functools import singledispatchmethod

import attrs
import numpy as np
import pint
import pinttr
import xarray as xr
from pinttr.util import ensure_units

import eradiate

from ..attrs import documented, frozen
from ..spectral.index import CKDSpectralIndex, MonoSpectralIndex, SpectralIndex
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg


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

    return xr.Dataset(
        data_vars={
            "sigma_a": (
                ("w", "z_layer"),
                sigma_a.reshape(1, z_layer.size),
                dict(
                    standard_name="absorption_coefficient",
                    units="km^-1",
                    long_name="absorption coefficient",
                ),
            ),
            "sigma_s": (
                ("w", "z_layer"),
                sigma_s.reshape(1, z_layer.size),
                dict(
                    standard_name="scattering_coefficient",
                    units="km^-1",
                    long_name="scattering coefficient",
                ),
            ),
            "sigma_t": (
                ("w", "z_layer"),
                sigma_t.reshape(1, z_layer.size),
                dict(
                    standard_name="extinction_coefficient",
                    units="km^-1",
                    long_name="extinction coefficient",
                ),
            ),
            "albedo": (
                ("w", "z_layer"),
                albedo.reshape(1, z_layer.size),
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
                dict(
                    standard_name="radiation_wavelength",
                    units="nm",
                    long_name="wavelength",
                ),
            ),
        },
        attrs={
            "convention": "CF-1.8",
            "title": "Atmospheric monochromatic radiative properties",
            "history": f"{datetime.datetime.utcnow().replace(microsecond=0)} - "
            f"data set creation - "
            f"{__name__}.make_dataset",
            "source": f"eradiate, version {eradiate.__version__}",
            "references": "",
        },
    )


@frozen(eq=False, init=False)
class ZGrid:
    """
    A container for a regular altitude grid.

    Notes
    -----
    * Instances are immutable.
    * Instances are hashable by ID. This is required to allow for using them as
      an argument of an LRU-cached function.
    * This class is used as the argument of the ``eval()`` family of methods.
    """

    levels: pint.Quantity = documented(
        pinttr.field(
            units=ucc.deferred("length"),
            on_setattr=None,  # frozen instance: on_setattr must be disabled
        ),
        type="pint.Quantity",
        init_type="quantity or array-like",
        doc="Grid node altitudes.\n\nUnit-enabled field (default: ``ucc['length']``).",
    )

    _layers: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
        on_setattr=None,  # frozen instance: on_setattr must be disabled
    )

    _layer_height: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
        on_setattr=None,  # frozen instance: on_setattr must be disabled
    )

    @_layer_height.validator
    def _layer_height_validator(self, attribute, value):
        if not np.isscalar(value.magnitude):
            raise ValueError("layer height must be a scalar")

    _total_height: pint.Quantity = pinttr.field(
        units=ucc.deferred("length"),
        on_setattr=None,  # frozen instance: on_setattr must be disabled
    )

    def __init__(self, levels: np.typing.ArrayLike):
        levels = ensure_units(levels, ucc.get("length"))
        layer_height = np.diff(levels)
        if not np.allclose(layer_height, layer_height[0]):
            raise ValueError("levels must be regularly spaced")
        layers = levels[:-1] + 0.5 * layer_height
        self.__attrs_init__(
            levels=levels,
            layers=layers,
            layer_height=layer_height[0],
            total_height=(levels.m[-1] - levels.m[0]) * levels.u,
        )

    @property
    def layers(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Vector of altitudes of layer centres.
        """
        return self._layers

    @property
    def layer_height(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Layer height.
        """
        return self._layer_height

    @property
    def n_levels(self) -> int:
        """
        Returns
        -------
        int
            Number of levels.
        """
        return len(self.levels)

    @property
    def n_layers(self) -> int:
        """
        Returns
        -------
        int
            Number of layers.
        """
        return len(self.layers)

    @property
    def total_height(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Total height covered by the altitude grid.
        """
        return self._total_height


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
        pass

    @property
    @abstractmethod
    def zgrid(self) -> ZGrid:
        """
        Default altitude grid used for profile evaluation.
        """
        pass

    @singledispatchmethod
    def eval_albedo(
        self,
        si: SpectralIndex,
        zgrid: ZGrid | None = None,
    ) -> pint.Quantity:
        """
        Evaluate albedo at given spectral index.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        zgrid : .ZGrid, optional
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
    def _(self, si, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_albedo_mono(w=si.w, zgrid=zgrid).squeeze()

    @eval_albedo.register(CKDSpectralIndex)
    def _(self, si, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_albedo_ckd(w=si.w, g=si.g, zgrid=zgrid).squeeze()

    def eval_albedo_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        raise NotImplementedError

    def eval_albedo_ckd(
        self,
        w: pint.Quantity,
        g: float,
        zgrid: ZGrid,
    ) -> pint.Quantity:
        raise NotImplementedError

    @singledispatchmethod
    def eval_sigma_t(
        self,
        si: SpectralIndex,
        zgrid: ZGrid | None = None,
    ) -> pint.Quantity:
        """
        Evaluate extinction coefficient at given spectral index.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        zgrid : .ZGrid, optional
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
    def _(self, si, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_sigma_t_mono(w=si.w, zgrid=zgrid).squeeze()

    @eval_sigma_t.register(CKDSpectralIndex)
    def _(self, si, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_sigma_t_ckd(w=si.w, g=si.g, zgrid=zgrid).squeeze()

    def eval_sigma_t_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        raise NotImplementedError

    def eval_sigma_t_ckd(
        self,
        w: pint.Quantity,
        g: float,
        zgrid: ZGrid,
    ) -> pint.Quantity:
        raise NotImplementedError

    @singledispatchmethod
    def eval_sigma_a(
        self,
        si: SpectralIndex,
        zgrid: ZGrid | None = None,
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient at given spectral index.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        zgrid : .ZGrid, optional
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
    def _(self, si, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_sigma_a_mono(w=si.w, zgrid=zgrid).squeeze()

    @eval_sigma_a.register(CKDSpectralIndex)
    def _(self, si, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_sigma_a_ckd(w=si.w, g=si.g, zgrid=zgrid).squeeze()

    def eval_sigma_a_mono(
        self,
        w: pint.Quantity,
        zgrid: ZGrid,
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient spectrum in monochromatic mode.
        """
        raise NotImplementedError

    def eval_sigma_a_ckd(
        self,
        w: pint.Quantity,
        g: float,
        zgrid: ZGrid,
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient spectrum in CKD modes.
        """
        raise NotImplementedError

    @singledispatchmethod
    def eval_sigma_s(
        self,
        si: SpectralIndex,
        zgrid: ZGrid | None = None,
    ) -> pint.Quantity:
        """
        Evaluate scattering coefficient at given spectral index.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        zgrid : .ZGrid, optional
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
    def _(self, si, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_sigma_s_mono(w=si.w, zgrid=zgrid).squeeze()

    @eval_sigma_s.register(CKDSpectralIndex)
    def _(self, si, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_sigma_s_ckd(w=si.w, g=si.g, zgrid=zgrid).squeeze()

    def eval_sigma_s_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        raise NotImplementedError

    def eval_sigma_s_ckd(
        self, w: pint.Quantity, g: float, zgrid: ZGrid
    ) -> pint.Quantity:
        raise NotImplementedError

    @singledispatchmethod
    def eval_depolarization_factor(
        self,
        si: SpectralIndex,
        zgrid: ZGrid | None = None,
    ) -> pint.Quantity:
        """
        Evaluate depolarization factor at given spectral index.

        Parameters
        ----------
        si : .SpectralIndex
            Spectral index.

        zgrid : .ZGrid, optional
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
    def _(self, si, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_depolarization_factor_mono(w=si.w, zgrid=zgrid)

    @eval_depolarization_factor.register(CKDSpectralIndex)
    def _(self, si, zgrid: ZGrid) -> pint.Quantity:
        return self.eval_depolarization_factor_ckd(w=si.w, g=si.g, zgrid=zgrid)

    def eval_depolarization_factor_mono(
        self, w: pint.Quantity, zgrid: ZGrid
    ) -> pint.Quantity:
        raise NotImplementedError

    def eval_depolarization_factor_ckd(
        self, w: pint.Quantity, g: float, zgrid: ZGrid
    ) -> pint.Quantity:
        raise NotImplementedError

    @singledispatchmethod
    def eval_dataset(
        self,
        si: SpectralIndex,
        zgrid: ZGrid | None = None,
    ) -> xr.Dataset:
        """
        Evaluate radiative properties at given spectral index.

        Parameters
        ----------
        si : :class:`.SpectralIndex`
            Spectral index.

        zgrid : .ZGrid, optional
            The altitude grid for which the radiative profile is evaluated.
            If unset, a profile-specific default is used.

        Returns
        -------
        Dataset
            Radiative property dataset.
        """
        raise NotImplementedError

    @eval_dataset.register(MonoSpectralIndex)
    def _(self, si, zgrid: ZGrid) -> xr.Dataset:
        return self.eval_dataset_mono(w=si.w, zgrid=zgrid)

    @eval_dataset.register(CKDSpectralIndex)
    def _(self, si, zgrid: ZGrid) -> xr.Dataset:
        return self.eval_dataset_ckd(w=si.w, g=si.g, zgrid=zgrid)

    def eval_dataset_mono(self, w: pint.Quantity, zgrid: ZGrid) -> xr.Dataset:
        raise NotImplementedError

    def eval_dataset_ckd(
        self,
        w: pint.Quantity,
        g: float,
        zgrid: ZGrid,
    ) -> xr.Dataset:
        raise NotImplementedError
