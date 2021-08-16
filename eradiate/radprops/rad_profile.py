"""
Radiative property profile definitions.
"""
import datetime
from abc import ABC, abstractmethod

import attr
import numpy as np
import pinttr
import xarray as xr

import eradiate
from eradiate import path_resolver

from .absorption import compute_sigma_a
from .rayleigh import compute_sigma_s_air
from .. import data
from .._factory import Factory
from .._mode import ModeFlags
from ..attrs import documented, parse_docs
from ..data.absorption_spectra import Absorber, Engine, find_dataset
from ..exceptions import UnsupportedModeError
from ..thermoprops import afgl1986, us76
from ..thermoprops.util import (
    compute_scaling_factors,
    interpolate,
    rescale_concentration,
)
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..validators import all_positive

rad_profile_factory = Factory()


@ureg.wraps(
    ret=None, args=("nm", "km", "km", "km^-1", "km^-1", "km^-1", ""), strict=False
)
def make_dataset(
    wavelength,
    z_level,
    z_layer=None,
    sigma_a=None,
    sigma_s=None,
    sigma_t=None,
    albedo=None,
):
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
        albedo = sigma_s / sigma_t
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
    def albedo(self, spectral_ctx=None):
        """
        Return albedo.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`pint.Quantity`:
            Profile albedo.
        """
        pass

    @abstractmethod
    def sigma_t(self, spectral_ctx=None):
        """
        Return extinction coefficient.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`pint.Quantity`:
            Profile extinction coefficient.
        """
        pass

    @abstractmethod
    def sigma_a(self, spectral_ctx=None):
        """
        Return absorption coefficient.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`pint.Quantity`:
            Profile absorption coefficient.
        """
        pass

    @abstractmethod
    def sigma_s(self, spectral_ctx=None):
        """
        Return scattering coefficient.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`pint.Quantity`:
            Profile scattering coefficient.
        """
        pass

    @abstractmethod
    def to_dataset(self, spectral_ctx=None):
        """
        Return a dataset that holds the radiative properties of the corresponding
        atmospheric profile.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext` or None):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).

        Returns → :class:`xarray.Dataset`:
            Radiative properties dataset.
        """
        pass


@rad_profile_factory.register(type_id="array")
@parse_docs
@attr.s
class ArrayRadProfile(RadProfile):
    """
    A flexible radiative properties profile whose level altitudes, albedo
    and extinction coefficient are specified as numpy arrays.

    .. warning::

       The ``albedo_values`` and ``sigma_t_values`` parameters must be 3D
       arrays (even though the profile is 1D) and have the same shape,
       the first axis being the x axis, the second the y axis and the third the
       z axis.
       The length of the ``albedo_values`` and ``sigma_t_values`` arrays
       along the z axis must be that of the ``levels`` array minus 1.

    .. admonition:: Example

       The following creates a radiative property profile with 3 layers between
       0 and 5 kilometers, corresponding to a purely scattering atmosphere
       (albedo = 1) with scattering coefficient values of :code:`9e-6`,
       :code:`5e-6` and :code:`1e-6` in units of :code:`ucc[length]^-1`:

        .. code:: python

            import numpy as np

            rad_profile = ArrayRadProfile(
                levels=ureg.Quantity(np.linspace(0, 5, 4), "km")
                sigma_t_values=np.array([9e-6, 5e-6, 1e-6]).reshape(1, 1, 3),
                albedo_values=np.ones((1, 1, 3))
            )

        Here the shape of the ``sigma_t_values`` and ``albedo_values``
        arrays is :code:`(1, 1, 3)`, where the last axis corresponds to the
        ``z`` axis.
    """

    levels = documented(
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

    albedo_values = documented(
        pinttr.ib(
            validator=all_positive,
            units=ureg.dimensionless,
        ),
        doc="An array specifying albedo values. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (dimensionless).",
        type="array",
    )

    sigma_t_values = documented(
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
        if value.ndim != 3:
            raise ValueError(
                f"while setting {attribute.name}: "
                f"must have 3 dimensions "
                f"(got shape {value.shape})"
            )

        if instance.albedo_values.shape != instance.sigma_t_values.shape:
            raise ValueError(
                f"while setting {attribute.name}: "
                f"'albedo_values' and 'sigma_t_values' must have "
                f"the same length"
            )

    def albedo(self, spectral_ctx=None):
        return self.albedo_values

    def sigma_t(self, spectral_ctx=None):
        return self.sigma_t_values

    def sigma_a(self, spectral_ctx=None):
        return self.sigma_t(spectral_ctx) * (1.0 - self.albedo(spectral_ctx))

    def sigma_s(self, spectral_ctx=None):
        return self.sigma_t(spectral_ctx) * self.albedo(spectral_ctx)

    @classmethod
    def from_dataset(cls, path):
        ds = xr.open_dataset(path_resolver.resolve(path))
        z_level = to_quantity(ds.z_level)
        n_layers = ds.z_level.size - 1
        albedo = to_quantity(ds.albedo).reshape(1, 1, n_layers)
        sigma_t = to_quantity(ds.sigma_t).reshape(1, 1, n_layers)
        return cls(albedo_values=albedo, sigma_t_values=sigma_t, levels=z_level)

    def to_dataset(self, spectral_ctx=None):
        if eradiate.mode().has_flags("ANY_MONO"):
            return make_dataset(
                wavelength=spectral_ctx.wavelength,
                z_level=self.levels,
                sigma_t=self.sigma_t().flatten(),
                albedo=self.albedo().flatten(),
            )

        else:
            raise UnsupportedModeError(supported="monochromatic")


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

    levels = documented(
        pinttr.ib(
            factory=lambda: np.arange(0.0, 86.01, 1.0) * ureg.km,
            units=ucc.deferred("length"),
        ),
        doc="Level altitudes. Default is regular mesh from 0 to 86 km with "
        "1 km layer size.\n"
        "\n"
        "Unit-enabled field (ucc[length]).",
        type="array",
        default="range(0, 87) km",
    )

    has_absorption = documented(
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

    has_scattering = documented(
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

    absorption_data_set = documented(
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

    @staticmethod
    def default_absorption_data_set(wavelength):
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

    def eval_thermoprops_profile(self):
        """
        Evaluate thermophysical properties.
        """
        return us76.make_profile(self.levels)

    def eval_sigma_a(self, spectral_ctx):
        """
        Evaluate absorption coefficient given spectral context.
        """
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            profile = self.eval_thermoprops_profile()
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

    def eval_sigma_s(self, spectral_ctx):
        """
        Evaluate scattering coefficient given spectral context.
        """
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            profile = self.eval_thermoprops_profile()
            if self.has_scattering:
                return compute_sigma_s_air(
                    wavelength=spectral_ctx.wavelength,
                    number_density=ureg.Quantity(profile.n.values, profile.n.units),
                )
            else:
                return ureg.Quantity(np.zeros(profile.z_layer.size), "km^-1")

        else:
            raise UnsupportedModeError(supported="monochromatic")

    def albedo(self, spectral_ctx=None):
        return (self.sigma_s(spectral_ctx) / self.sigma_t(spectral_ctx)).to(
            ureg.dimensionless
        )

    def sigma_a(self, spectral_ctx=None):
        # Add missing dimensions
        return self.eval_sigma_a(spectral_ctx)[np.newaxis, np.newaxis, ...]

    def sigma_s(self, spectral_ctx=None):
        # Add missing dimensions
        return self.eval_sigma_s(spectral_ctx)[np.newaxis, np.newaxis, ...]

    def sigma_t(self, spectral_ctx=None):
        return self.sigma_a(spectral_ctx) + self.sigma_s(spectral_ctx)

    def to_dataset(self, spectral_ctx=None):
        """
        Return a dataset that holds the atmosphere radiative properties.

        Returns → :class:`xarray.Dataset`:
            Radiative properties dataset.
        """
        profile = self.eval_thermoprops_profile()
        return make_dataset(
            wavelength=spectral_ctx.wavelength,
            z_level=to_quantity(profile.z_level),
            z_layer=to_quantity(profile.z_layer),
            sigma_a=self.sigma_a(spectral_ctx).flatten(),
            sigma_s=self.sigma_s(spectral_ctx).flatten(),
        )


_AFGL1986_MODELS = [
    "tropical",
    "midlatitude_summer",
    "midlatitude_winter",
    "subarctic_summer",
    "subarctic_winter",
    "us_standard",
]


@rad_profile_factory.register(type_id="afgl1986")
@parse_docs
@attr.s
class AFGL1986RadProfile(RadProfile):
    """
    Radiative properties profile corresponding to the AFGL (1986) atmospheric
    thermophysical properties profiles
    :cite:`Anderson1986AtmosphericConstituentProfiles`.

    :cite:`Anderson1986AtmosphericConstituentProfiles` defines six models,
    listed in the table below.

    .. list-table:: AFGL (1986) atmospheric thermophysical properties profiles models
       :widths: 2 4 4
       :header-rows: 1

       * - Model number
         - Model identifier
         - Model name
       * - 1
         - ``tropical``
         - Tropic (15N Annual Average)
       * - 2
         - ``midlatitude_summer``
         - Mid-Latitude Summer (45N July)
       * - 3
         - ``midlatitude_winter``
         - Mid-Latitude Winter (45N Jan)
       * - 4
         - ``subarctic_summer``
         - Sub-Arctic Summer (60N July)
       * - 5
         - ``subarctic_winter``
         - Sub-Arctic Winter (60N Jan)
       * - 6
         - ``us_standard``
         - U.S. Standard (1976)

    .. attention::
        The original altitude mesh specified by
        :cite:`Anderson1986AtmosphericConstituentProfiles` is a piece-wise
        regular altitude mesh with an altitude step of 1 km from 0 to 25 km,
        2.5 km from 25 km to 50 km and 5 km from 50 km to 120 km.
        Since the Eradiate kernel only supports regular altitude mesh, the
        original atmospheric thermophysical properties profiles were
        interpolated on the regular altitude mesh with an altitude step of 1 km
        from 0 to 120 km.

    Although the altitude meshes of the interpolated
    :cite:`Anderson1986AtmosphericConstituentProfiles` profiles is fixed,
    this class lets you define a custom altitude mesh (regular or irregular).

    .. admonition:: Example
        :class: example

        .. code:: python

            import numpy as np
            from eradiate import unit_registry as ureg

            AFGL1986RadProfile(
                levels=np.array([0., 5., 10., 25., 50., 100]) * ureg.km
            )

        In this example, the :cite:`Anderson1986AtmosphericConstituentProfiles`
        profile is truncated at the height of 100 km.

    All six models include the following six absorbing molecular species:
    H2O, CO2, O3, N2O, CO, CH4 and O2.
    The concentrations of these species in the atmosphere is fixed by
    :cite:`Anderson1986AtmosphericConstituentProfiles`.
    However, this class allows you to rescale the concentrations of each
    individual molecular species to custom concentration values.
    Custom concentrations can be provided in different units.

    .. admonition:: Example
        :class: example

        .. code:: python

            from eradiate import unit_registry as ureg

            AFGL1986RadProfile(
                concentrations={
                    "H2O": ureg.Quantity(15 , "kg/m^2"),
                    "CO2": 420 * ureg.dimensionless,
                    "O3": 350 * ureg.dobson_unit,
                }
            )
    """

    model = documented(
        attr.ib(
            default="us_standard",
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc=(
            "AFGL (1986) atmospheric thermophysical properties profile model "
            "identifier in [``'tropical'``, ``'midlatitude_summer'``, "
            "``'midlatitude_winter'``, ``'subarctic_summer'``, "
            "``'subarctic_winter'``, ``'us_standard'``.]"
        ),
        type="str",
    )

    @model.validator
    def _model_validator(self, attribute, value):
        if value not in _AFGL1986_MODELS:
            raise ValueError(
                f"{attribute} should be in {_AFGL1986_MODELS} " f"(got {value})."
            )

    levels = documented(
        pinttr.ib(
            factory=lambda: np.arange(0.0, 120.01, 1.0) * ureg.km,
            units=ucc.deferred("length"),
        ),
        doc="Level altitudes. Default is a regular mesh from 0 to 120 km with "
        "1 km layer size.\n"
        "\n"
        "Unit-enabled field (ucc[length]).",
        type="array",
        default="range(0, 121) km",
    )

    concentrations = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(dict),
            validator=attr.validators.optional(attr.validators.instance_of(dict)),
        ),
        doc="Mapping of species and concentration. "
        "For more information about rescaling process and the supported "
        "concentration units, refer to the documentation of "
        ":func:`~eradiate.thermoprops.util.compute_scaling_factors`.",
        type="dict",
    )

    has_absorption = documented(
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

    has_scattering = documented(
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

    absorption_data_sets = documented(
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

    def eval_thermoprops_profile(self):
        """
        Compute the atmosphere thermophysical properties.
        """
        thermoprops = afgl1986.make_profile(model_id=self.model)
        if self.levels is not None:
            thermoprops = interpolate(
                ds=thermoprops, z_level=self.levels, conserve_columns=True
            )

        if self.concentrations is not None:
            factors = compute_scaling_factors(
                ds=thermoprops, concentration=self.concentrations
            )
            thermoprops = rescale_concentration(ds=thermoprops, factors=factors)

        return thermoprops

    @staticmethod
    def _auto_compute_sigma_a_absorber(wavelength, absorber, n_absorber, p, t):
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
    def _compute_sigma_a_absorber_from_data_set(path, wavelength, n_absorber, p, t):
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

    def eval_sigma_a(self, spectral_ctx):
        """
        Evaluate absorption coefficient given a spectral context.

        .. note:: Extrapolate to zero when wavelength, pressure and/or
           temperature are out of bounds.
        """
        profile = self.eval_thermoprops_profile()
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

    def eval_sigma_s(self, spectral_ctx):
        """
        Evaluate scattering coefficient given a spectral context.
        """
        profile = self.eval_thermoprops_profile()
        if self.has_scattering:
            return compute_sigma_s_air(
                wavelength=spectral_ctx.wavelength,
                number_density=ureg.Quantity(profile.n.values, profile.n.units),
            )
        else:
            return ureg.Quantity(np.zeros(profile.z_layer.size), "km^-1")

    def albedo(self, spectral_ctx=None):
        return (self.sigma_s(spectral_ctx) / self.sigma_t(spectral_ctx)).to(
            ureg.dimensionless
        )

    def sigma_a(self, spectral_ctx=None):
        # Add missing dimensions
        return self.eval_sigma_a(spectral_ctx)[np.newaxis, np.newaxis, ...]

    def sigma_s(self, spectral_ctx=None):
        # Add missing dimensions
        return self.eval_sigma_s(spectral_ctx)[np.newaxis, np.newaxis, ...]

    def sigma_t(self, spectral_ctx=None):
        return self.sigma_a(spectral_ctx) + self.sigma_s(spectral_ctx)

    def to_dataset(self, spectral_ctx=None):
        """
        Return a dataset that holds the atmosphere radiative properties.

        Returns → :class:`xarray.Dataset`:
            Radiative properties dataset.
        """
        return make_dataset(
            wavelength=spectral_ctx.wavelength,
            z_level=to_quantity(self.eval_thermoprops_profile().z_level),
            z_layer=to_quantity(self.eval_thermoprops_profile().z_layer),
            sigma_a=self.sigma_a(spectral_ctx).flatten(),
            sigma_s=self.sigma_s(spectral_ctx).flatten(),
        )
