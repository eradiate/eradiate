"""
Radiative property profiles.
"""
from __future__ import annotations

import datetime
import pathlib
from abc import ABC, abstractmethod
from typing import MutableMapping, Optional, Union

import attr
import numpy as np
import pint
import pinttr
import xarray as xr

import eradiate

from .absorption import compute_sigma_a
from .rayleigh import compute_sigma_s_air
from .. import data
from .._factory import Factory
from .._mode import ModeFlags
from .._presolver import path_resolver
from ..attrs import documented, parse_docs
from ..contexts import SpectralContext
from ..data.absorption_spectra import Absorber, Engine, find_dataset
from ..exceptions import UnsupportedModeError
from ..thermoprops import afgl1986, us76
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..validators import all_positive

rad_profile_factory = Factory()


@ureg.wraps(
    ret=None, args=("nm", "km", "km", "km^-1", "km^-1", "km^-1", ""), strict=False
)
def make_dataset(
    wavelength: Union[pint.Quantity, float],
    z_level: Union[pint.Quantity, float],
    z_layer: Optional[Union[pint.Quantity, float]] = None,
    sigma_a: Optional[Union[pint.Quantity, float]] = None,
    sigma_s: Optional[Union[pint.Quantity, float]] = None,
    sigma_t: Optional[Union[pint.Quantity, float]] = None,
    albedo: Optional[Union[pint.Quantity, float]] = None,
) -> xr.Dataset:
    """
    Makes an atmospheric radiative properties data set.

    Parameter ``wavelength`` (float):
        Wavelength [nm].

    Parameter ``z_level`` (array):
        Level altitudes [km].

    Parameter ``z_layer`` (array):
        Layer altitudes [km].

        If ``None``, the layer altitudes are computed automatically, so that
        they are halfway between the adjacent altitude levels.

    Parameter ``sigma_a`` (array):
        Absorption coefficient values [km^-1].

    Parameter ``sigma_s`` (array):
        Scattering coefficient values [km^-1].

    Parameter ``sigma_t`` (array):
        Extinction coefficient values [km^-1].

    Parameter ``albedo`` (array):
        Albedo values [/].

    Returns → :class:`~xarray.Dataset`:
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
    coefficients as Pint-wrapped 3D Numpy arrays.

    .. warning::

       Arrays returned by the :meth:`albedo`, :meth:`sigma_a`, :meth:`sigma_s`
       and :meth:`sigma_t` methods **must** be 3D. Should the profile
       be one-dimensional, invariant dimensions can be set to 1.

    .. seealso::

       :class:`.RadProfileFactory`

    """

    @abstractmethod
    def eval_albedo(
        self: RadProfile, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        """
        Return albedo.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`~pint.Quantity`:
            Profile albedo.
        """
        pass

    @abstractmethod
    def eval_sigma_t(
        self: RadProfile, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        """
        Return extinction coefficient.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`~pint.Quantity`:
            Profile extinction coefficient.
        """
        pass

    @abstractmethod
    def eval_sigma_a(
        self, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        """
        Return absorption coefficient.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`~pint.Quantity`:
            Profile absorption coefficient.
        """
        pass

    @abstractmethod
    def eval_sigma_s(
        self, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        """
        Return scattering coefficient.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`~pint.Quantity`:
            Profile scattering coefficient.
        """
        pass

    @abstractmethod
    def to_dataset(self, spectral_ctx: Optional[SpectralContext] = None) -> xr.Dataset:
        """
        Return a dataset that holds the radiative properties of the corresponding
        atmospheric profile.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`~xarray.Dataset`:
            Radiative properties dataset.
        """
        pass


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
        "Unit-enabled field (default: ucc[length]).",
        type="array",
    )

    albedo_values: pint.Quantity = documented(
        pinttr.ib(
            validator=all_positive,
            units=ureg.dimensionless,
        ),
        doc="An array specifying albedo values. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (dimensionless).",
        type="array",
    )

    sigma_t_values: pint.Quantity = documented(
        pinttr.ib(
            validator=all_positive,
            units=ucc.deferred("collision_coefficient"),
        ),
        doc="An array specifying extinction coefficient values. **Required, no "
        "default**.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]^-1).",
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

    def eval_albedo(
        self, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        return self.albedo_values

    def eval_sigma_t(
        self, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        return self.sigma_t_values

    def eval_sigma_a(
        self, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        return self.eval_sigma_t(spectral_ctx) * (1.0 - self.eval_albedo(spectral_ctx))

    def eval_sigma_s(
        self, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        return self.eval_sigma_t(spectral_ctx) * self.eval_albedo(spectral_ctx)

    @classmethod
    def from_dataset(
        cls: ArrayRadProfile, path: Union[str, pathlib.Path]
    ) -> ArrayRadProfile:
        ds = xr.open_dataset(path_resolver.resolve(path))
        z_level = to_quantity(ds.z_level)
        albedo = to_quantity(ds.albedo)
        sigma_t = to_quantity(ds.sigma_t)
        return cls(albedo_values=albedo, sigma_t_values=sigma_t, levels=z_level)

    def to_dataset(
        self: ArrayRadProfile, spectral_ctx: Optional[SpectralContext] = None
    ) -> xr.Dataset:
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return make_dataset(
                wavelength=spectral_ctx.wavelength,
                z_level=self.levels,
                sigma_t=self.eval_sigma_t(spectral_ctx=spectral_ctx),
                albedo=self.eval_albedo(spectral_ctx=spectral_ctx),
            ).squeeze()
        else:
            raise UnsupportedModeError(supported="monochromatic")


def _convert_thermoprops_us76_approx(
    value: Union[MutableMapping, xr.Dataset]
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

    absorption_data_set: Optional[str] = documented(
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
    def thermoprops(self: US76ApproxRadProfile) -> xr.Dataset:
        """
        Return thermophysical properties.
        """
        return self._thermoprops

    @property
    def levels(self: US76ApproxRadProfile) -> pint.Quantity:
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
        dataset_ids = find_dataset(
            wavenumber=wavenumber,
            absorber=Absorber.us76_u86_4,
            engine=Engine.SPECTRA,
        )
        if len(dataset_ids) == 1:
            return data.open(category="absorption_spectrum", id=dataset_ids[0])
        elif len(dataset_ids) == 2:
            ds1 = data.open(category="absorption_spectrum", id=dataset_ids[0])
            ds2 = data.open(category="absorption_spectrum", id=dataset_ids[1])
            return xr.concat([ds1.isel(w=slice(0,-1)), ds2], dim="w")


    def eval_sigma_a(
        self: US76ApproxRadProfile, spectral_ctx: SpectralContext
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient given spectral context.
        """
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            profile = self.thermoprops
            if self.has_absorption:
                wavelength = spectral_ctx.wavelength

                if self.absorption_data_set is None:  # ! this is never tested
                    data_set = self.default_absorption_data_set(wavelength=wavelength)
                else:
                    data_set = xr.open_dataset(self.absorption_data_set)

                # Compute scattering coefficient
                return compute_sigma_a(
                    ds=data_set,
                    wl=wavelength,
                    p=profile.p.values,
                    n=profile.n.values,
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

        else:
            raise UnsupportedModeError(supported="monochromatic")

    def eval_sigma_s(
        self: US76ApproxRadProfile, spectral_ctx: SpectralContext
    ) -> pint.Quantity:
        """
        Evaluate scattering coefficient given spectral context.
        """
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            profile = self.thermoprops
            if self.has_scattering:
                return compute_sigma_s_air(
                    wavelength=spectral_ctx.wavelength,
                    number_density=to_quantity(profile.n),
                )
            else:
                return ureg.Quantity(np.zeros(profile.z_layer.size), "km^-1")

        else:
            raise UnsupportedModeError(supported="monochromatic")

    def eval_albedo(
        self: US76ApproxRadProfile, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        """
        Evaluate albedo given spectral context.
        """
        return (self.eval_sigma_s(spectral_ctx) / self.eval_sigma_t(spectral_ctx)).to(
            ureg.dimensionless
        )

    def eval_sigma_t(
        self: US76ApproxRadProfile, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        """
        Evaluate extinction coefficient given spectral context.
        """
        return self.eval_sigma_a(spectral_ctx) + self.eval_sigma_s(spectral_ctx)

    def to_dataset(
        self: US76ApproxRadProfile, spectral_ctx: Optional[SpectralContext] = None
    ) -> xr.Dataset:
        """
        Return a dataset that holds the atmosphere radiative properties.

        Returns → :class:`xarray.Dataset`:
            Radiative properties dataset.
        """
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            profile = self.thermoprops
            return make_dataset(
                wavelength=spectral_ctx.wavelength,
                z_level=to_quantity(profile.z_level),
                z_layer=to_quantity(profile.z_layer),
                sigma_a=self.eval_sigma_a(spectral_ctx),
                sigma_s=self.eval_sigma_s(spectral_ctx),
            ).squeeze()
        else:
            raise UnsupportedModeError(supported="monochromatic")


def _convert_thermoprops_afgl1986(
    value: Union[MutableMapping, xr.Dataset]
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

    absorption_data_sets: Optional[MutableMapping[str, str]] = documented(
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
    def thermoprops(self: AFGL1986RadProfile) -> xr.Dataset:
        return self._thermoprops

    @property
    def levels(self: AFGL1986RadProfile) -> pint.Quantity:
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
            dataset_ids = find_dataset(
                wavenumber=wavenumber,
                absorber=absorber,
                engine=Engine.SPECTRA,
            )
            if len(dataset_ids) == 1:
                dataset = data.open(category="absorption_spectrum", id=dataset_ids[0])
            elif len(dataset_ids) == 2:
                ds1 = data.open(category="absorption_spectrum", id=dataset_ids[0])
                ds2 = data.open(category="absorption_spectrum", id=dataset_ids[1])
                dataset = xr.concat([ds1.isel(w=slice(0,-1)), ds2], dim="w")
            
            sigma_a_absorber = compute_sigma_a(
                ds=dataset,
                wl=wavelength,
                p=p.values,
                t=t.values,
                n=n_absorber.values,
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

        Parameter ``path`` (str):
            Path to data set file.

        Parameter ``wavelength`` (float):
            Wavelength  [nm].

        Parameter ``n_absorber`` (float):
            Absorber number density [m^-3].

        Parameter ``p`` (float):
            Pressure [Pa].

        Parameter ``t`` (float):
            Temperature [K].

        Returns → ():
            Absorption coefficient [km^-1].
        """
        dataset = xr.open_dataset(path)
        return compute_sigma_a(
            ds=dataset,
            wl=wavelength,
            p=p.values,
            t=t.values,
            n=n_absorber.values,
            fill_values=dict(
                w=0.0, pt=0.0
            ),  # extrapolate to zero along wavenumber and pressure and temperature dimensions
        )

    def eval_sigma_a(
        self: AFGL1986RadProfile, spectral_ctx: SpectralContext
    ) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        .. note:: Extrapolate to zero when wavelength, pressure and/or
           temperature are out of bounds.
        """
        profile = self.thermoprops
        if self.has_absorption:
            wavelength = spectral_ctx.wavelength

            p = profile.p
            t = profile.t
            n = profile.n
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
                n_absorber = n * mr.sel(species=absorber.value)

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

    def eval_sigma_s(
        self: AFGL1986RadProfile, spectral_ctx: SpectralContext
    ) -> pint.Quantity:
        """
        Evaluate scattering coefficient given a spectral context.
        """
        thermoprops = self.thermoprops
        if self.has_scattering:
            return compute_sigma_s_air(
                wavelength=spectral_ctx.wavelength,
                number_density=ureg.Quantity(to_quantity(thermoprops.n)),
            )
        else:
            return ureg.Quantity(np.zeros(thermoprops.z_layer.size), "km^-1")

    def eval_albedo(
        self: AFGL1986RadProfile, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        return (self.eval_sigma_s(spectral_ctx) / self.eval_sigma_t(spectral_ctx)).to(
            ureg.dimensionless
        )

    def eval_sigma_t(
        self: AFGL1986RadProfile, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        return self.eval_sigma_s(spectral_ctx=spectral_ctx) + self.eval_sigma_a(
            spectral_ctx=spectral_ctx
        )

    def to_dataset(
        self: AFGL1986RadProfile, spectral_ctx: Optional[SpectralContext] = None
    ) -> pint.Quantity:
        """
        Return a dataset that holds the atmosphere radiative properties.

        Returns → :class:`xarray.Dataset`:
            Radiative properties dataset.
        """
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return make_dataset(
                wavelength=spectral_ctx.wavelength,
                z_level=to_quantity(self.thermoprops.z_level),
                z_layer=to_quantity(self.thermoprops.z_layer),
                sigma_a=self.eval_sigma_a(spectral_ctx),
                sigma_s=self.eval_sigma_s(spectral_ctx),
            ).squeeze()
        else:
            raise UnsupportedModeError(supported="monochromatic")
