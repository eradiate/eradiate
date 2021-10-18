"""
Radiative property profiles.
"""
from __future__ import annotations

import datetime
import pathlib
import typing as t
from abc import ABC

import attr
import numpy as np
import pint
import pinttr
import xarray as xr

import eradiate

from .absorption import compute_sigma_a
from .rayleigh import compute_sigma_s_air
from .. import data, validators
from .._factory import Factory
from .._mode import ModeFlags
from .._presolver import path_resolver
from ..attrs import documented, parse_docs
from ..ckd import Bindex
from ..contexts import SpectralContext
from ..data.absorption_spectra import Absorber, Engine, find_dataset
from ..exceptions import UnsupportedModeError
from ..thermoprops import afgl1986, us76
from ..thermoprops.util import (
    compute_column_mass_density,
    compute_column_number_density,
)
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg

rad_profile_factory = Factory()


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


@attr.s
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

    def eval_albedo(self, spectral_ctx: SpectralContext) -> pint.Quantity:
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

        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return self.eval_albedo_mono(spectral_ctx.wavelength).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return self.eval_albedo_ckd(spectral_ctx.bindex).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_albedo_mono(self, w: pint.Quantity) -> pint.Quantity:
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
        raise NotImplementedError

    def eval_albedo_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        """
        Evaluate albedo spectrum in CKD modes.

        Parameters
        ----------
        *bindexes : :class:`.Bindex`
            One or several CKD bindexes for which to evaluate the spectrum.

        Returns
        -------
        quantity
            Evaluated profile albedo as an array with shape (n_layers, len(bindexes)).
        """
        raise NotImplementedError

    def eval_sigma_t(self, spectral_ctx: SpectralContext) -> pint.Quantity:
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

        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return self.eval_sigma_t_mono(spectral_ctx.wavelength).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return self.eval_sigma_t_ckd(spectral_ctx.bindex).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_sigma_t_mono(self, w: pint.Quantity) -> pint.Quantity:
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
        raise NotImplementedError

    def eval_sigma_t_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        """
        Evaluate extinction coefficient spectrum in CKD modes.

        Parameters
        ----------
        *bindexes : :class:`.Bindex`
            One or several CKD bindexes for which to evaluate the spectrum.

        Returns
        -------
        quantity
            Evaluated profile extinction coefficient as an array with shape
            (n_layers, len(bindexes)).
        """
        raise NotImplementedError

    def eval_sigma_a(self, spectral_ctx: SpectralContext) -> pint.Quantity:
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

        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return self.eval_sigma_a_mono(spectral_ctx.wavelength).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return self.eval_sigma_a_ckd(spectral_ctx.bindex).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_sigma_a_mono(self, w: pint.Quantity) -> pint.Quantity:
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
        raise NotImplementedError

    def eval_sigma_a_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        """
        Evaluate absorption coefficient spectrum in CKD modes.

        Parameters
        ----------
        *bindexes : :class:`.Bindex`
            One or several CKD bindexes for which to evaluate the spectrum.

        Returns
        -------
        quantity
            Evaluated profile absorption coefficient as an array with shape
            (n_layers, len(bindexes)).
        """
        raise NotImplementedError

    def eval_sigma_s(self, spectral_ctx: SpectralContext) -> pint.Quantity:
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

        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return self.eval_sigma_s_mono(spectral_ctx.wavelength).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return self.eval_sigma_s_ckd(spectral_ctx.bindex).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_sigma_s_mono(self, w: pint.Quantity) -> pint.Quantity:
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
        raise NotImplementedError

    def eval_sigma_s_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        """
        Evaluate scattering coefficient spectrum in CKD modes.

        Parameters
        ----------
        *bindexes : :class:`.Bindex`
            One or several CKD bindexes for which to evaluate the spectrum.

        Returns
        -------
        quantity
            Evaluated profile scattering coefficient as an array with shape
            (n_layers, len(bindexes)).
        """
        raise NotImplementedError

    def eval_dataset(self, spectral_ctx: SpectralContext) -> xr.Dataset:
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
            Radiative properties dataset.
        """
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return self.eval_dataset_mono(spectral_ctx.wavelength).squeeze()

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return self.eval_dataset_ckd(spectral_ctx.bindex).squeeze()

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    def eval_dataset_mono(self, w: pint.Quantity) -> xr.Dataset:
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
        raise NotImplementedError

    def eval_dataset_ckd(self, *bindexes: Bindex) -> xr.Dataset:
        """
        Return a dataset that holds the radiative properties of the corresponding
        atmospheric profile in CKD modes

        Parameters
        ----------
        *bindexes : :class:`.Bindex`
            One or several CKD bindexes for which to evaluate spectra.

        Returns
        -------
        Dataset
            Radiative properties dataset.
        """
        raise NotImplementedError


@rad_profile_factory.register(type_id="array")
@parse_docs
@attr.s
class ArrayRadProfile(RadProfile):
    """
    A flexible 1D radiative property profile whose level altitudes, albedo
    and extinction coefficient are specified as numpy arrays.
    """

    levels: pint.Quantity = documented(
        pinttr.ib(
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=pinttr.validators.has_compatible_units,
            units=ucc.deferred("length"),
        ),
        doc="Level altitudes. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="array",
    )

    albedo_values: pint.Quantity = documented(
        pinttr.ib(
            validator=validators.all_positive,
            units=ureg.dimensionless,
        ),
        doc="An array specifying albedo values. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (dimensionless).",
        type="array",
    )

    sigma_t_values: pint.Quantity = documented(
        pinttr.ib(
            validator=validators.all_positive,
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
        if not eradiate.mode().has_flags(ModeFlags.ANY_MONO):
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
        ds = xr.open_dataset(path_resolver.resolve(path))
        z_level = to_quantity(ds.z_level)
        albedo = to_quantity(ds.albedo)
        sigma_t = to_quantity(ds.sigma_t)
        return cls(albedo_values=albedo, sigma_t_values=sigma_t, levels=z_level)


def _convert_thermoprops_us76_approx(
    value: t.Union[t.MutableMapping, xr.Dataset]
) -> xr.Dataset:
    if isinstance(value, dict):
        return us76.make_profile(**value)
    else:
        return value


@rad_profile_factory.register(type_id="us76_approx")
@parse_docs
@attr.s
class US76ApproxRadProfile(RadProfile):
    """
    Radiative properties profile approximately corresponding to an
    atmospheric profile based on the original U.S. Standard Atmosphere 1976
    atmosphere model.

    .. note::

       Instantiating this class requires to download the absorption dataset
       ``spectra-us76_u86_4`` and place it in ``$ERADIATE_DIR/resources/data/``.

    The :mod:`~eradiate.thermoprops.us76` module implements the original *U.S.
    Standard Atmosphere 1976* atmosphere model, as defined by the
    :cite:`NASA1976USStandardAtmosphere` technical report.
    In the original atmosphere model, the gases are assumed well-mixed below
    the altitude of 86 kilometers.
    In the present radiative properties profile, the absorption coefficient is
    computed using the ``spectra-us76_u86_4`` absorption dataset.
    This dataset provides the absorption cross section of a specific mixture
    of N2, O2, CO2 and CH4, the mixing ratio of which are those defined by the
    *U.S. Standard Atmosphere 1976* model for the region of altitudes under
    86 kilometers, where these four gas species are well-mixed.
    As a result, the dataset is representative of the *U.S. Standard Atmosphere
    1976* model only below 86 kilometers.
    Since the atmosphere is typically a hundred kilometers high or more in
    radiative transfer applications, and in order to make the radiative
    properties profile reach these altitudes, the absorption coefficient
    is nevertheless computed using the ``spectra-us76_u86_4`` dataset.
    This approximation assumes that the absorption coefficient does not vary
    much whether the mixing ratios of the absorbing gas mixture are those
    below or above 86 km.
    Furthermore, the *U.S. Standard Atmosphere 1976* model includes other gas
    species than N2, O2, CO2 and CH4.
    They are: Ar, He, Ne, Kr, H, O, Xe, He and H2.
    All these species except H2 are absent from the
    `HITRAN <https://hitran.org/>`_ spectroscopic database.
    Since the absorption datasets are computed using HITRAN, the atomic species
    could not be included in ``spectra-us76_u86_4``.
    H2 was mistakenly forgotten and should be added to the dataset in a future
    revision.


    .. note::
       We refer to the *U.S. Standard Atmosphere 1976* atmosphere model as the
       model defined by the set of assumptions and equations in part 1 of the
       report, and "numerically" illustrated by the extensive tables in part
       4 of the report.
       In particular, the part 3, entitled *Trace constituents*, which
       provides rough estimates and discussions on the amounts of trace
       constituents such as ozone, water vapor, nitrous oxide, methane, and so
       on, is not considered as part of the *U.S. Standard Atmosphere 1976*
       atmosphere model because it does not clearly defines the concentration
       values of all trace constituents at all altitudes, neither does it
       provide a way to compute them.

    .. note::
       It seems that the identifier "US76" is commonly used to refer to a
       standard atmospheric profile used in radiative transfer applications.
       However, there appears to be some confusion around the definition of
       that standard atmospheric profile.
       In our understanding, what is called the "US76 standard atmospheric
       profile", or "US76" in short, **is not the U.S. Standard Atmosphere
       1976 atmosphere model** but instead the so-called "U.S. Standard (1976)
       atmospheric constituent profile model" in a AFGL technical report
       entitled *AFGL Atmospheric Constituent Profiles (0-120km)* and
       published in 1986 by Anderson et al
       :cite:`Anderson1986AtmosphericConstituentProfiles`.
       Although the "U.S. Standard (1976) atmospheric profile model" of the
       AFGL's report is based on the *U.S. Standard Atmosphere* 1976 atmosphere
       model (hence the name), it is significantly different when it comes
       about the gas species concentration profiles.
       Notably, the "U.S. Standard (1976) atmospheric profile model" of the
       AFGL's report include radiatively active gases such as H2O, O3, N2O,
       and CO, that the *U.S. Standard Atmosphere 1976* atmosphere model does
       not include.
    """

    _thermoprops: xr.Dataset = documented(
        attr.ib(
            factory=lambda: us76.make_profile(),
            converter=_convert_thermoprops_us76_approx,
            validator=attr.validators.instance_of(xr.Dataset),
        ),
        doc="Thermophysical properties.",
        type=":class:`~xarray.Dataset`",
        default="us76.make_profile",
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

    absorption_data_set: t.Optional[str] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(str),
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="Absorption data set file path. If ``None``, the default "
        "absorption data sets will be used to compute the absorption "
        "coefficient. Otherwise, the absorption data set whose path is "
        "provided will be used to compute the absorption coefficient.",
        type="str",
    )

    @property
    def thermoprops(self) -> xr.Dataset:
        """
        Return thermophysical properties.
        """
        return self._thermoprops

    @property
    def levels(self) -> pint.Quantity:
        """
        Return level altitudes.
        """
        return to_quantity(self.thermoprops.z_level)

    @staticmethod
    def default_absorption_data_set(wavelength: pint.Quantity) -> xr.Dataset:
        """
        Return default absorption data set.
        """
        # ! This method is never tested because it requires large absorption
        # data sets to be downloaded
        wavenumber = 1.0 / wavelength
        dataset_id = find_dataset(
            wavenumber=wavenumber,
            absorber=Absorber.us76_u86_4,
            engine=Engine.SPECTRA,
        )
        return data.open(category="absorption_spectrum", id=dataset_id)

    def eval_sigma_a_mono(self, w: pint.Quantity) -> pint.Quantity:
        profile = self.thermoprops
        if self.has_absorption:
            if self.absorption_data_set is None:  # ! this is never tested
                data_set = self.default_absorption_data_set(wavelength=w)
            else:
                data_set = xr.open_dataset(self.absorption_data_set)

            # Compute scattering coefficient
            return compute_sigma_a(
                ds=data_set,
                wl=w,
                p=to_quantity(profile.p),
                n=to_quantity(profile.n),
                fill_values=dict(
                    pt=0.0
                ),  # us76_u86_4 dataset is limited to pressures above
                # 0.101325 Pa, but us76 thermophysical profile goes below that
                # value for altitudes larger than 93 km. At these altitudes, the
                # number density is so small compared to that at the sea level that
                # we assume it is negligible.
            )
        else:
            return ureg.Quantity(np.zeros(profile.z_layer.size), "km^-1")

    def eval_sigma_a_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        raise NotImplementedError(
            "CKD data sets are not yet available for the U.S. Standard Atmosphere 1976 atmopshere model."
        )

    def eval_sigma_s_mono(self, w: pint.Quantity) -> pint.Quantity:
        profile = self.thermoprops
        if self.has_scattering:
            return compute_sigma_s_air(
                wavelength=w,
                number_density=to_quantity(profile.n),
            )
        else:
            return ureg.Quantity(np.zeros(profile.z_layer.size), "km^-1")

    def eval_sigma_s_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        wavelengths = ureg.Quantity(
            np.array([bindex.bin.wcenter.m_as("nm") for bindex in bindexes]), "nm"
        )
        return self.eval_sigma_s_mono(w=wavelengths)

    def eval_albedo_mono(self, w: pint.Quantity) -> pint.Quantity:
        sigma_s = self.eval_sigma_s_mono(w)
        sigma_t = self.eval_sigma_t_mono(w)
        return np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        ).to("dimensionless")

    def eval_albedo_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        sigma_s = self.eval_sigma_s_ckd(*bindexes)
        sigma_t = self.eval_sigma_t_ckd(*bindexes)
        return np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        ).to("dimensionless")

    def eval_sigma_t_mono(self, w: pint.Quantity) -> pint.Quantity:
        return self.eval_sigma_a_mono(w) + self.eval_sigma_s_mono(w)

    def eval_sigma_t_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        return self.eval_sigma_a_ckd(bindexes) + self.eval_sigma_s_ckd(bindexes)

    def eval_dataset_mono(self, w: pint.Quantity) -> xr.Dataset:
        profile = self.thermoprops
        return make_dataset(
            wavelength=w,
            z_level=to_quantity(profile.z_level),
            z_layer=to_quantity(profile.z_layer),
            sigma_a=self.eval_sigma_a_mono(w),
            sigma_s=self.eval_sigma_s_mono(w),
        ).squeeze()

    def eval_dataset_ckd(self, *bindexes: Bindex) -> xr.Dataset:
        if len(bindexes) > 1:
            raise NotImplementedError
        else:
            return make_dataset(
                wavelength=bindexes[0].bin.wcenter,
                z_level=to_quantity(self.thermoprops.z_level),
                z_layer=to_quantity(self.thermoprops.z_layer),
                sigma_a=self.eval_sigma_a_ckd(*bindexes),
                sigma_s=self.eval_sigma_s_ckd(*bindexes),
            ).squeeze()


def _convert_thermoprops_afgl1986(
    value: t.Union[t.MutableMapping, xr.Dataset]
) -> xr.Dataset:
    if isinstance(value, dict):
        return afgl1986.make_profile(**value)
    else:
        return value


@rad_profile_factory.register(type_id="afgl1986")
@parse_docs
@attr.s
class AFGL1986RadProfile(RadProfile):
    """
    Radiative properties profile corresponding to the AFGL (1986) atmospheric
    thermophysical properties profiles
    :cite:`Anderson1986AtmosphericConstituentProfiles`.
    """

    _thermoprops: xr.Dataset = documented(
        attr.ib(
            factory=lambda: afgl1986.make_profile(),
            converter=_convert_thermoprops_afgl1986,
            validator=attr.validators.instance_of(xr.Dataset),
        ),
        doc="Thermophysical properties.",
        type=":class:`~xarray.Dataset`",
        default=":func:`~eradiate.thermoprops.afgl1986.make_profile`",
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

    absorption_data_sets: t.Optional[t.MutableMapping[str, str]] = documented(
        attr.ib(
            factory=dict,
            converter=attr.converters.optional(dict),
            validator=attr.validators.optional(attr.validators.instance_of(dict)),
        ),
        doc="Mapping of species and absorption data set files paths. If "
        "``None``, the default absorption data sets are used to compute "
        "the absorption coefficient. If not ``None``, the absorption data "
        "set files whose paths are provided in the mapping will be used to "
        "compute the absorption coefficient. If the mapping does not "
        "include all species from the AFGL (1986) atmospheric "
        "thermophysical profile, the default data sets will be used to "
        "compute the absorption coefficient of the corresponding species.",
        type="dict",
        default="{}",
    )

    @property
    def thermoprops(self) -> xr.Dataset:
        return self._thermoprops

    @property
    def levels(self) -> pint.Quantity:
        return to_quantity(self.thermoprops.z_level)

    @staticmethod
    def _auto_compute_sigma_a_absorber(
        wavelength: pint.Quantity,
        absorber: Absorber,
        n_absorber: pint.Quantity,
        p: pint.Quantity,
        t: pint.Quantity,
    ) -> pint.Quantity:
        """
        Compute absorption coefficient using the predefined absorption data set.
        """
        # ! This method is never tested because it requires large absorption
        # data sets to be downloaded
        wavenumber = 1.0 / wavelength
        try:
            dataset_id = find_dataset(
                wavenumber=wavenumber,
                absorber=absorber,
                engine=Engine.SPECTRA,
            )
            dataset = data.open(category="absorption_spectrum", id=dataset_id)
            sigma_a_absorber = compute_sigma_a(
                ds=dataset,
                wl=wavelength,
                p=p,
                t=t,
                n=n_absorber,
                fill_values=dict(
                    w=0.0, pt=0.0
                ),  # extrapolate to zero along wavenumber and pressure and temperature dimensions
            )
        except ValueError:  # no data at current wavelength/wavenumber
            sigma_a_absorber = ureg.Quantity(np.zeros(len(p)), "km^-1")

        return sigma_a_absorber

    @staticmethod
    def _compute_sigma_a_absorber_from_data_set(
        path: str,
        wavelength: pint.Quantity,
        n_absorber: pint.Quantity,
        p: pint.Quantity,
        t: pint.Quantity,
    ) -> pint.Quantity:
        """
        Compute the absorption coefficient using a custom absorption data set
        file.

        Parameters
        ----------
        path : (str):
            Path to data set file.

        wavelength : float
            Wavelength  [nm].

        n_absorber : float
            Absorber number density [m^-3].

        p : float
            Pressure [Pa].

        t : float
            Temperature [K].

        Returns
        -------
        quantity
            Absorption coefficient [km^-1].
        """
        dataset = xr.open_dataset(path)
        return compute_sigma_a(
            ds=dataset,
            wl=wavelength,
            p=p,
            t=t,
            n=n_absorber,
            fill_values=dict(
                w=0.0, pt=0.0
            ),  # extrapolate to zero along wavenumber and pressure and temperature dimensions
        )

    def eval_sigma_a_mono(self, w: pint.Quantity) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        Notes
        -----
        Extrapolate to zero when wavelength, pressure and/or temperature are out
        of bounds.
        """
        profile = self.thermoprops
        if self.has_absorption:
            wavelength = w

            p = to_quantity(profile.p)
            t = to_quantity(profile.t)
            n = to_quantity(profile.n)
            mr = profile.mr

            sigma_a = np.full(mr.shape, np.nan)
            absorbers = [
                Absorber.CH4,
                Absorber.CO,
                Absorber.CO2,
                Absorber.H2O,
                Absorber.N2O,
                Absorber.O2,
                Absorber.O3,
            ]

            for i, absorber in enumerate(absorbers):
                n_absorber = n * to_quantity(mr.sel(species=absorber.value))

                if absorber.value in self.absorption_data_sets:
                    sigma_a_absorber = self._compute_sigma_a_absorber_from_data_set(
                        path=self.absorption_data_sets[absorber.value],
                        wavelength=wavelength,
                        n_absorber=n_absorber,
                        p=p,
                        t=t,
                    )
                else:
                    sigma_a_absorber = self._auto_compute_sigma_a_absorber(
                        wavelength=wavelength,
                        absorber=absorber,
                        n_absorber=n_absorber,
                        p=p,
                        t=t,
                    )

                sigma_a[i, :] = sigma_a_absorber.m_as("km^-1")

            sigma_a = np.sum(sigma_a, axis=0)

            return ureg.Quantity(sigma_a, "km^-1")
        else:
            return ureg.Quantity(np.zeros(profile.z_layer.size), "km^-1")

    def eval_sigma_a_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        ds = eradiate.data.open(
            category="ckd_absorption", id="afgl_1986-us_standard-10nm"
        )
        # evaluate H2O and O3 concentrations
        h2o_concentration = compute_column_mass_density(
            ds=self.thermoprops, species="H2O"
        )
        o3_concentration = compute_column_number_density(
            ds=self.thermoprops, species="O3"
        )

        z = to_quantity(self.thermoprops.z_layer).m_as(ds.z.units)
        return ureg.Quantity(
            [
                ds.k.sel(bd=(bindex.bin.id, bindex.index))
                .interp(
                    z=z,
                    H2O=h2o_concentration.m_as(ds["H2O"].units),
                    O3=o3_concentration.m_as(ds["O3"].units),
                    kwargs=dict(fill_value=0.0),
                )
                .values
                for bindex in bindexes
            ],
            ds.k.units,
        )

    def eval_sigma_s_mono(self, w: pint.Quantity) -> pint.Quantity:
        thermoprops = self.thermoprops
        if self.has_scattering:
            return compute_sigma_s_air(
                wavelength=w,
                number_density=to_quantity(thermoprops.n),
            )
        else:
            return ureg.Quantity(np.zeros(thermoprops.z_layer.size), "km^-1")

    def eval_sigma_s_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        wavelengths = ureg.Quantity(
            np.array([bindex.bin.wcenter.m_as("nm") for bindex in bindexes]), "nm"
        )
        return self.eval_sigma_s_mono(w=wavelengths)

    def eval_albedo_mono(self, w: pint.Quantity) -> pint.Quantity:
        return (self.eval_sigma_s_mono(w) / self.eval_sigma_t_mono(w)).to(
            ureg.dimensionless
        )

    def eval_albedo_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        sigma_s = self.eval_sigma_s_ckd(*bindexes)
        sigma_t = self.eval_sigma_t_ckd(*bindexes)
        return np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        ).to("dimensionless")

    def eval_sigma_t_mono(self, w: pint.Quantity) -> pint.Quantity:
        return self.eval_sigma_s_mono(w) + self.eval_sigma_a_mono(w)

    def eval_sigma_t_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        return self.eval_sigma_a_ckd(*bindexes) + self.eval_sigma_s_ckd(*bindexes)

    def eval_dataset_mono(self, w: pint.Quantity) -> xr.Dataset:
        return make_dataset(
            wavelength=w,
            z_level=to_quantity(self.thermoprops.z_level),
            z_layer=to_quantity(self.thermoprops.z_layer),
            sigma_a=self.eval_sigma_a_mono(w),
            sigma_s=self.eval_sigma_s_mono(w),
        ).squeeze()

    def eval_dataset_ckd(self, *bindexes: Bindex) -> xr.Dataset:
        if len(bindexes) > 1:
            raise NotImplementedError
        else:
            return make_dataset(
                wavelength=bindexes[0].bin.wcenter,
                z_level=to_quantity(self.thermoprops.z_level),
                z_layer=to_quantity(self.thermoprops.z_layer),
                sigma_a=self.eval_sigma_a_ckd(*bindexes),
                sigma_s=self.eval_sigma_s_ckd(*bindexes),
            ).squeeze()
