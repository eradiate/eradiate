"""
Molecular atmospheres.
"""

from __future__ import annotations

import attrs
import joseki
import numpy as np
import pint
import xarray as xr

import eradiate

from ._core import AbstractHeterogeneousAtmosphere
from ..core import traverse
from ..phase import PhaseFunction, RayleighPhaseFunction
from ...attrs import define, documented
from ...contexts import KernelContext
from ...converters import convert_thermoprops
from ...exceptions import UnsupportedModeError
from ...radprops import (
    AbsorptionDatabase,
    AtmosphereRadProfile,
    ErrorHandlingConfiguration,
    RadProfile,
    ZGrid,
)
from ...spectral.index import SpectralIndex
from ...units import unit_registry as ureg
from ...util.misc import summary_repr


def _default_absorption_data():
    if eradiate.mode().is_mono:
        return "komodo"
    elif eradiate.mode().is_ckd:
        return "monotropa"
    else:
        raise UnsupportedModeError(unsupported=["mono", "ckd"])


@define(eq=False, slots=False)
class MolecularAtmosphere(AbstractHeterogeneousAtmosphere):
    """
    Molecular atmosphere scene element [``molecular``].

    See Also
    --------
    :class:`~eradiate.scenes.atmosphere.ParticleLayer`,
    :class:`~eradiate.scenes.atmosphere.HeterogeneousAtmosphere`.

    Notes
    -----
    This is commonly referred to as a clear-sky
    atmosphere, namely the atmosphere is free of clouds, aerosols or any other
    type of liquid or solid particles in suspension.

    It describes a gaseous mixture (air) whose thermophysical properties,
    namely pressure, temperature and constituent mole fractions, are allowed to
    vary with altitude (see the ``thermoprops`` attribute).

    The corresponding radiative properties are computed with this
    thermophysical profile as input.

    Special care must be taken that the absorption data is able to accomodate
    the specified thermophysical profile, especially along the constituent mole
    fractions axes (see also ``error_handler_config``).
    Note that the absoprtion data is able to support one spectral mode at a
    time. As a result, the Eradiate mode must be selected before instantiating
    this class, and the relevant absorption data must be provided.

    The Rayleigh scattering theory is used to compute the air volume
    scattering coefficient.

    The scattering phase function defaults to the Rayleigh scattering phase
    function but can be set to other values.
    """

    _absorption_data: AbsorptionDatabase = documented(
        attrs.field(
            kw_only=True,
            factory=AbsorptionDatabase.default,
            converter=AbsorptionDatabase.convert,
            validator=attrs.validators.instance_of(AbsorptionDatabase),
        ),
        doc="Absorption coefficient data. The passed value is pre-processed by "
        ":meth:`.AbsorptionDatabase.convert`.",
        type="AbsorptionDatabase",
        init_type="str or path-like or dict or .AbsorptionDatabase",
        default=":meth:`AbsorptionDatabase.default() <.AbsorptionDatabase.default>`",
    )

    _thermoprops: xr.Dataset | None = documented(
        attrs.field(
            kw_only=True,
            factory=lambda: joseki.make(
                identifier="afgl_1986-us_standard",
                z=np.linspace(0.0, 120.0, 121) * ureg.km,
                additional_molecules=False,
            ),
            converter=attrs.converters.optional(convert_thermoprops),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(xr.Dataset)
            ),
            repr=summary_repr,
        ),
        doc="Thermophysical property dataset. If a path is passed, Eradiate will "
        "look it up and load it. If a dictionary is passed, it will be passed "
        "as keyword argument to ``joseki.make()``. The default is "
        '``joseki.make(identifier="afgl_1986-us_standard",  z=np.linspace(0.0, 120.0, 121) * ureg.km)``. '
        "See the `Joseki documentation <https://joseki.readthedocs.io/en/latest/reference/joseki/index.html#joseki.make>`__ "
        "for details.",
        type="Dataset or None",
        init_type="Dataset or path-like or dict or None",
    )

    has_absorption: bool = documented(
        attrs.field(kw_only=True, default=True, converter=bool),
        doc="Absorption switch. If ``True``, the absorption coefficient is "
        "computed. Else, the absorption coefficient is set to zero.",
        type="bool",
        default="True",
    )

    has_scattering: bool = documented(
        attrs.field(kw_only=True, default=True, converter=bool),
        doc="Scattering switch. If ``True``, the scattering coefficient is "
        "computed. Else, the scattering coefficient is set to zero.",
        type="bool",
        default="True",
    )

    rayleigh_depolarization: np.ndarray | str = documented(
        attrs.field(
            converter=lambda x: x
            if isinstance(x, str)
            else np.array(x, dtype=np.float64),
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
    )

    @has_absorption.validator
    @has_scattering.validator
    def _switch_validator(self, attribute, value):
        if not self.has_absorption and not self.has_scattering:
            raise ValueError(
                f"while validating {attribute.name}: at least one of "
                "'has_absorption' and 'has_scattering' must be True"
            )

    _radprops_profile: RadProfile | None = documented(
        attrs.field(
            kw_only=True,
            default=None,
            validator=attrs.validators.optional(
                attrs.validators.instance_of(RadProfile)
            ),
        ),
        doc="Radiative property profile. "
        "If ``thermoprops`` is not ``None``, this field is automatically "
        "overridden with an :class:`.AtmosphereRadProfile` during initialization. "
        "Note that at least ``thermoprops`` or ``radprops_profile`` should not "
        "be ``None``.",
        type=".RadProfile or None",
        init_type=".RadProfile or None",
        default="None",
    )

    error_handler_config: ErrorHandlingConfiguration | None = documented(
        attrs.field(
            kw_only=True,
            default=None,
            converter=ErrorHandlingConfiguration.convert,
            validator=attrs.validators.optional(
                attrs.validators.instance_of(ErrorHandlingConfiguration)
            ),
        ),
        doc="Error handler configuration for absorption data interpolation. If "
        "unset, the global configuration specified in the "
        "``absorption_database.error_handling`` section is used. ",
        type=".ErrorHandlingConfiguration or None",
        init_type="dict or .ErrorHandlingConfiguration, optional",
    )

    def update(self) -> None:
        # Inherit docstring
        self.phase.id = self.phase_id

        if self.thermoprops is not None:
            self._radprops_profile = AtmosphereRadProfile(
                thermoprops=self.thermoprops,
                has_scattering=self.has_scattering,
                has_absorption=self.has_absorption,
                absorption_data=self.absorption_data,
                rayleigh_depolarization=self.rayleigh_depolarization,
            )

    # --------------------------------------------------------------------------
    #              Spatial extension and thermophysical properties
    # --------------------------------------------------------------------------

    @property
    def thermoprops(self) -> xr.Dataset:
        # Inherit docstring
        return self._thermoprops

    def eval_mfp(self, ctx: KernelContext) -> pint.Quantity:
        # Inherit docstring
        min_sigma_s = self.radprops_profile.eval_sigma_s(ctx.si).min()
        return np.divide(
            1.0,
            min_sigma_s,
            where=min_sigma_s != 0.0,
            out=np.array([np.inf]),
        )

    # --------------------------------------------------------------------------
    #                             Radiative properties
    # --------------------------------------------------------------------------

    @property
    def absorption_data(self) -> AbsorptionDatabase:
        # Inherit docstring
        return self._absorption_data

    @property
    def phase(self) -> PhaseFunction:
        # Inherit docstring

        def eval_depolarization_factor(si: SpectralIndex) -> np.ndarray:
            return self.eval_depolarization_factor(si).m_as("dimensionless")

        # pass callable for depolarization to phase function for InitParams and UpdateParams.
        return RayleighPhaseFunction(
            depolarization=eval_depolarization_factor, geometry=self.geometry
        )

    @property
    def radprops_profile(self) -> RadProfile:
        # Inherit docstring
        return self._radprops_profile

    def eval_albedo(
        self, si: SpectralIndex, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self.radprops_profile.eval_albedo(
            si,
            zgrid=self.geometry.zgrid if zgrid is None else zgrid,
        )

    def eval_sigma_t(
        self, si: SpectralIndex, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self.radprops_profile.eval_sigma_t(
            si,
            zgrid=self.geometry.zgrid if zgrid is None else zgrid,
        )

    def eval_sigma_a(
        self, si: SpectralIndex, zgrid: ZGrid | None = None, **kwargs
    ) -> pint.Quantity:
        # Inherit docstring
        return self.radprops_profile.eval_sigma_a(
            si,
            zgrid=self.geometry.zgrid if zgrid is None else zgrid,
        )

    def eval_sigma_s(
        self, si: SpectralIndex, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        # Inherit docstring
        return self.radprops_profile.eval_sigma_s(
            si,
            zgrid=self.geometry.zgrid if zgrid is None else zgrid,
        )

    def eval_depolarization_factor(
        self, si: SpectralIndex, zgrid: ZGrid | None = None
    ) -> pint.Quantity:
        return self.radprops_profile.eval_depolarization_factor(
            si,
            zgrid=self.geometry.zgrid if zgrid is None else zgrid,
        )

    # --------------------------------------------------------------------------
    #                             Kernel dictionary
    # --------------------------------------------------------------------------

    @property
    def _template_phase(self) -> dict:
        # Inherit docstring
        result, _ = traverse(self.phase)
        return result.data

    @property
    def _params_phase(self) -> dict:
        # Inherit docstring
        _, result = traverse(self.phase)
        return result.data
