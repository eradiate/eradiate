from __future__ import annotations

import datetime
from abc import ABC, abstractmethod

import attrs
import numpy as np
import pint
import pinttr
import xarray as xr

import eradiate

from .._factory import Factory
from ..attrs import documented, parse_docs
from ..ckd import Bindex
from ..contexts import SpectralContext
from ..exceptions import UnsupportedModeError
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg

rad_profile_factory = Factory()
rad_profile_factory.register_lazy_batch(
    [
        ("_afgl1986.AFGL1986RadProfile", "afgl_1986", {}),
        ("_us76_approx.US76ApproxRadProfile", "us76_approx", {}),
    ],
    cls_prefix="eradiate.radprops",
)


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
            "('sigma_a', 'sigma_s') or ('sigma_t', 'albedo')."
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


@parse_docs
@attrs.frozen(eq=False, init=False)
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
        doc="Grid node altitudes.\n"
        "\n"
        "Unit-enabled field (default: ``ucc['length']``).",
    )

    _layers: pint.Quantity = pinttr.field(
        default=None,
        units=ucc.deferred("length"),
        on_setattr=None,  # frozen instance: on_setattr must be disabled
    )

    _layer_height: pint.Quantity = pinttr.field(
        default=None,
        units=ucc.deferred("length"),
        on_setattr=None,  # frozen instance: on_setattr must be disabled
    )

    @_layer_height.validator
    def _layer_height_validator(self, attribute, value):
        if not np.isscalar(value.magnitude):
            raise ValueError("layer height must be a scalar")

    def __init__(self, levels: pint.Quantity):
        layer_height = np.diff(levels)
        if not np.allclose(layer_height, layer_height[0]):
            raise ValueError("levels must be regularly spaced")
        layers = levels[:-1] + 0.5 * layer_height
        self.__attrs_init__(levels=levels, layers=layers, layer_height=layer_height[0])

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


@attrs.define(eq=False)
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

    @property
    @abstractmethod
    def zgrid(self) -> ZGrid:
        """
        Default altitude grid used for profile evaluation.
        """
        pass

    def eval_albedo(
        self, spectral_ctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate albedo spectrum based on a spectral context. This method
        dispatches evaluation to specialised methods depending on the active
        mode.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        if eradiate.mode().is_mono:
            return self.eval_albedo_mono(
                spectral_ctx.wavelength,
                self.zgrid if zgrid is None else zgrid,
            ).squeeze()

        elif eradiate.mode().is_ckd:
            return self.eval_albedo_ckd(
                [spectral_ctx.bindex],
                self.zgrid if zgrid is None else zgrid,
            ).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    @abstractmethod
    def eval_albedo_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        """
        Evaluate albedo spectrum in monochromatic modes.

        Parameters
        ----------
        w : quantity
            Wavelength values at which the spectrum is to be evaluated.

        Returns
        -------
        quantity
            Evaluated profile albedo as an array with shape (n_layers, len(w)).
        """
        pass

    @abstractmethod
    def eval_albedo_ckd(self, bindexes: list[Bindex], zgrid: ZGrid) -> pint.Quantity:
        """
        Evaluate albedo spectrum in CKD modes.

        Parameters
        ----------
        bindexes : list of :class:`.Bindex`
            CKD bindexes for which to evaluate the spectrum.

        Returns
        -------
        quantity
            Evaluated profile albedo as an array with shape (n_layers, len(bindexes)).
        """
        pass

    def eval_sigma_t(
        self, spectral_ctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate extinction coefficient spectrum based on a spectral context.
        This method dispatches evaluation to specialised methods depending on
        the active mode.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        if eradiate.mode().is_mono:
            return self.eval_sigma_t_mono(
                spectral_ctx.wavelength,
                self.zgrid if zgrid is None else zgrid,
            ).squeeze()

        elif eradiate.mode().is_ckd:
            return self.eval_sigma_t_ckd(
                [spectral_ctx.bindex],
                self.zgrid if zgrid is None else zgrid,
            ).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    @abstractmethod
    def eval_sigma_t_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        """
        Evaluate extinction coefficient spectrum in monochromatic modes.

        Parameters
        ----------
        w : quantity
            Wavelength values at which the spectrum is to be evaluated.

        Returns
        -------
        quantity
            Evaluated profile extinction coefficient as an array with shape
            (n_layers, len(w)).
        """
        pass

    @abstractmethod
    def eval_sigma_t_ckd(self, bindexes: list[Bindex], zgrid: ZGrid) -> pint.Quantity:
        """
        Evaluate extinction coefficient spectrum in CKD modes.

        Parameters
        ----------
        bindexes : list of :class:`.Bindex`
            One or several CKD bindexes for which to evaluate the spectrum.

        Returns
        -------
        quantity
            Evaluated profile extinction coefficient as an array with shape
            (n_layers, len(bindexes)).
        """
        pass

    def eval_sigma_a(
        self, spectral_ctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient spectrum based on a spectral context.
        This method dispatches evaluation to specialised methods depending on
        the active mode.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        if eradiate.mode().is_mono:
            return self.eval_sigma_a_mono(
                spectral_ctx.wavelength,
                self.zgrid if zgrid is None else zgrid,
            ).squeeze()

        elif eradiate.mode().is_ckd:
            return self.eval_sigma_a_ckd(
                [spectral_ctx.bindex],
                self.zgrid if zgrid is None else zgrid,
            ).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    @abstractmethod
    def eval_sigma_a_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        """
        Evaluate absorption coefficient spectrum in monochromatic modes.

        Parameters
        ----------
        w : quantity
            Wavelength values at which the spectrum is to be evaluated.

        Returns
        -------
        quantity
            Evaluated profile absorption coefficient as an array with shape
            (n_layers, len(w)).
        """
        pass

    @abstractmethod
    def eval_sigma_a_ckd(self, bindexes: Bindex, zgrid: ZGrid) -> pint.Quantity:
        """
        Evaluate absorption coefficient spectrum in CKD modes.

        Parameters
        ----------
        bindexes : :class:`.Bindex`
            One or several CKD bindexes for which to evaluate the spectrum.

        Returns
        -------
        quantity
            Evaluated profile absorption coefficient as an array with shape
            (n_layers, len(bindexes)).
        """
        pass

    def eval_sigma_s(
        self, spectral_ctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        """
        Evaluate scattering coefficient spectrum based on a spectral context.
        This method dispatches evaluation to specialised methods depending on
        the active mode.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode, bin and
            quadrature point index in CKD mode).

        Returns
        -------
        quantity
            Evaluated spectrum as an array with length equal to the number of
            layers.
        """
        if eradiate.mode().is_mono:
            return self.eval_sigma_s_mono(
                spectral_ctx.wavelength,
                self.zgrid if zgrid is None else zgrid,
            ).squeeze()

        elif eradiate.mode().is_ckd:
            return self.eval_sigma_s_ckd(
                [spectral_ctx.bindex],
                self.zgrid if zgrid is None else zgrid,
            ).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    @abstractmethod
    def eval_sigma_s_mono(self, w: pint.Quantity, zgrid: ZGrid) -> pint.Quantity:
        """
        Evaluate scattering coefficient spectrum in monochromatic modes.

        Parameters
        ----------
        w : quantity
            Wavelength values at which the spectrum is to be evaluated.

        Returns
        -------
        quantity
            Evaluated profile scattering coefficient as an array with shape
            (n_layers, len(w)).
        """
        pass

    @abstractmethod
    def eval_sigma_s_ckd(self, bindexes: list[Bindex], zgrid: ZGrid) -> pint.Quantity:
        """
        Evaluate scattering coefficient spectrum in CKD modes.

        Parameters
        ----------
        bindexes : list of :class:`.Bindex`
            One or several CKD bindexes for which to evaluate the spectrum.

        Returns
        -------
        quantity
            Evaluated profile scattering coefficient as an array with shape
            (n_layers, len(bindexes)).
        """
        pass

    def eval_dataset(
        self, spectral_ctx: SpectralContext, zgrid: ZGrid | None = None
    ) -> xr.Dataset:
        """
        Return a dataset that holds the radiative properties of the corresponding
        atmospheric profile. This method dispatches evaluation to specialised
        methods depending on the active mode.

        Parameters
        ----------
        spectral_ctx : :class:`.SpectralContext`
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns
        -------
        Dataset
            Radiative property dataset.
        """
        if eradiate.mode().is_mono:
            return self.eval_dataset_mono(
                spectral_ctx.wavelength,
                self.zgrid if zgrid is None else zgrid,
            ).squeeze()

        elif eradiate.mode().is_ckd:
            return self.eval_dataset_ckd(
                [spectral_ctx.bindex],
                self.zgrid if zgrid is None else zgrid,
            ).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    @abstractmethod
    def eval_dataset_mono(self, w: pint.Quantity, zgrid: ZGrid) -> xr.Dataset:
        """
        Return a dataset that holds the radiative properties of the corresponding
        atmospheric profile in monochromatic modes.

        Parameters
        ----------
        w : quantity
            Wavelength values at which spectra are to be evaluated.

        Returns
        -------
        Dataset
            Radiative properties dataset.
        """
        pass

    @abstractmethod
    def eval_dataset_ckd(self, bindexes: list[Bindex], zgrid: ZGrid) -> xr.Dataset:
        """
        Return a dataset that holds the radiative properties of the
        corresponding atmospheric profile in CKD modes.

        Parameters
        ----------
        bindexes : list of :class:`.Bindex`
            One or several CKD bindexes for which to evaluate spectra.

        Returns
        -------
        Dataset
            Radiative properties dataset.
        """
        pass
