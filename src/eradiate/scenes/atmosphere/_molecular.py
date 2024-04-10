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
from ..phase import PhaseFunction, RayleighPhaseFunction, phase_function_factory
from ...attrs import documented, parse_docs
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
from ...spectral.ckd import BinSet, QuadSpec
from ...spectral.index import SpectralIndex
from ...spectral.mono import WavelengthSet
from ...spectral.spectral_set import SpectralSet
from ...units import unit_registry as ureg
from ...util.misc import summary_repr


def _default_absorption_data():
    if eradiate.mode().is_mono:
        return "komodo"
    elif eradiate.mode().is_ckd:
        return "monotropa"
    else:
        raise UnsupportedModeError(unsupported=["mono", "ckd"])


@parse_docs
@attrs.define(eq=False, slots=False)
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

    absorption_data: AbsorptionDatabase = documented(
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

    _thermoprops: xr.Dataset = documented(
        attrs.field(
            kw_only=True,
            factory=lambda: joseki.make(
                identifier="afgl_1986-us_standard",
                z=np.linspace(0.0, 120.0, 121) * ureg.km,
                additional_molecules=False,
            ),
            converter=convert_thermoprops,
            validator=attrs.validators.instance_of(xr.Dataset),
            repr=summary_repr,
        ),
        doc="Thermophysical property dataset. If a path is passed, Eradiate will "
        "look it up and load it. If a dictionary is passed, it will be passed "
        "as keyword argument to ``joseki.make()``. The default is "
        '``joseki.make(identifier="afgl_1986-us_standard",  z=np.linspace(0.0, 120.0, 121) * ureg.km)``. '
        "See `the Joseki docs <https://rayference.github.io/joseki/latest/reference/#src.joseki.core.make>`_ "
        "for details.",
        type="Dataset",
        init_type="Dataset or path-like or dict",
    )

    _phase: PhaseFunction = documented(
        attrs.field(
            kw_only=True,
            factory=lambda: RayleighPhaseFunction(),
            converter=phase_function_factory.convert,
            validator=attrs.validators.instance_of(PhaseFunction),
        ),
        doc="Phase function.",
        type=":class:`.PhaseFunction`",
        init_type=":class:`.PhaseFunction` or dict",
        default=":class:`RayleighPhaseFunction() <.RayleighPhaseFunction>`",
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

    @has_absorption.validator
    @has_scattering.validator
    def _switch_validator(self, attribute, value):
        if not self.has_absorption and not self.has_scattering:
            raise ValueError(
                f"while validating {attribute.name}: at least one of "
                "'has_absorption' and 'has_scattering' must be True"
            )

    _radprops_profile: AtmosphereRadProfile | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(AtmosphereRadProfile)
        ),
        init=False,
        repr=False,
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

        self._radprops_profile = AtmosphereRadProfile(
            thermoprops=self.thermoprops,
            has_scattering=self.has_scattering,
            has_absorption=self.has_absorption,
            absorption_data=self.absorption_data,
        )

    def spectral_set(self, quad_spec: QuadSpec | None = None) -> None | SpectralSet:
        if self.has_absorption:
            if eradiate.mode().is_mono:
                return WavelengthSet.from_absorption_database(self.absorption_data)

            elif eradiate.mode().is_ckd:
                if quad_spec is None:
                    quad_spec = QuadSpec.default()  # default
                return BinSet.from_absorption_database(self.absorption_data, quad_spec)

            else:
                raise NotImplementedError
        else:
            return None

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
    def phase(self) -> PhaseFunction:
        # Inherit docstring
        return self._phase

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
