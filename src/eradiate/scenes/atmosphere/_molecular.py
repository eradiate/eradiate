"""
Molecular atmospheres.
"""

from __future__ import annotations

from copy import deepcopy

import attrs
import joseki
import numpy as np
import pint
import portion as P
import xarray as xr

import eradiate

from ._core import AbstractHeterogeneousAtmosphere
from ..core import traverse
from ..phase import PhaseFunction, RayleighPhaseFunction, phase_function_factory
from ...attrs import documented, parse_docs
from ...contexts import KernelContext
from ...converters import convert_absorption_data, convert_thermoprops
from ...exceptions import UnsupportedModeError
from ...radprops import (
    AtmosphereRadProfile,
    RadProfile,
    ZGrid,
)
from ...radprops._atmosphere import _absorption_data_repr
from ...radprops.absorption import DEFAULT_HANDLER_CONFIG
from ...spectral.ckd import BinSet, QuadratureSpecifications
from ...spectral.index import SpectralIndex
from ...spectral.mono import WavelengthSet
from ...units import unit_registry as ureg
from ...util.misc import summary_repr
from ...validators import validate_absorption_data


def default_absorption_data() -> tuple:
    """Default absorption data based on active spectral mode.

    Returns
    -------
    tuple
        Absorption data specifications.

    Raises
    ------
    UnsupportedModeError:
        When the spectral mode is neither 'mono' or 'ckd'.

    Notes
    -----
    The correct Eradiate mode must be set before this function is called.
    In monochromatic mode, the returned absorption data specifications are
    suitable for working in the 250 nm to 3125 nm wavelength range.
    In CKD mode, the returned absorption data specifications are suitable for
    working with the wavenumber band [18100, 18200] cm^-1 (equivalent to the
    wavelength range [549.45, 552.48] nm).
    The corresponding absorption datasets will be downloaded from the Eradiate
    online data store, if they have not already been downloaded.
    """
    wavelength_range = [549.5, 550.5] * ureg.nm
    if eradiate.mode().is_mono:
        dataset_codename = "komodo"
    elif eradiate.mode().is_ckd:
        dataset_codename = "monotropa"
    else:
        raise UnsupportedModeError(supported=["mono", "ckd"])
    return (dataset_codename, wavelength_range)


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

    absorption_data: dict[P.Interval, xr.Dataset] = documented(
        attrs.field(
            kw_only=True,
            factory=default_absorption_data,
            converter=convert_absorption_data,
            validator=validate_absorption_data,
            repr=_absorption_data_repr,
        ),
        doc="Absorption coefficient data. "
        "If a file path, the absorption coefficient is loaded from the "
        "specified file (must be a NetCDF file)."
        "If a tuple, the first element is the dataset codename and the"
        "second is the desired working wavelength range.",
        type="Dict of portion.Interval and xarray.Dataset",
        init_type=(
            "Dataset or list of Datasets or str or :class:`.PathLike` or "
            "tuple[str, pint.Quantity]"
        ),
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
        doc="Thermophysical properties.",
        type="Dataset",
        default=":meth:`joseki.make() <joseki.make>` with "
        "``identifier'=afgl_1986-us_standard'`` and "
        "``z=np.linspace(0.0, 120.0, 121) * ureg.km``",
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
        attrs.field(
            kw_only=True,
            default=True,
            converter=bool,
        ),
        doc="Absorption switch. If ``True``, the absorption coefficient is "
        "computed. Else, the absorption coefficient is set to zero.",
        type="bool",
        default="True",
    )

    has_scattering: bool = documented(
        attrs.field(
            kw_only=True,
            default=True,
            converter=bool,
        ),
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

    error_handler_config: dict[str, dict[str, str]] = documented(
        attrs.field(
            kw_only=True,
            factory=lambda: deepcopy(DEFAULT_HANDLER_CONFIG),
            validator=attrs.validators.deep_mapping(
                key_validator=attrs.validators.instance_of(str),
                value_validator=attrs.validators.deep_mapping(
                    key_validator=attrs.validators.instance_of(str),
                    value_validator=attrs.validators.instance_of(str),
                ),
            ),
        ),
        doc="Error handler configuration for absorption data interpolation.",
        type="dict",
        default=DEFAULT_HANDLER_CONFIG,
    )

    def update(self) -> None:
        # Inherit docstring
        self.phase.id = self.phase_id

        self.error_handler_config = {
            **DEFAULT_HANDLER_CONFIG,
            **self.error_handler_config,
        }

        self._radprops_profile = AtmosphereRadProfile(
            thermoprops=self.thermoprops,
            has_scattering=self.has_scattering,
            has_absorption=self.has_absorption,
            absorption_data=self.absorption_data,
            error_handler_config=self.error_handler_config,
        )

    def spectral_set(
        self, quad_spec: QuadratureSpecifications = QuadratureSpecifications()
    ) -> None | BinSet | WavelengthSet:
        if self.has_absorption:
            absorption_data = list(self.absorption_data.values())
            if len(absorption_data) == 1:
                absorption_data = absorption_data[0]
            if eradiate.mode().is_mono:
                return WavelengthSet.from_absorption_dataset(absorption_data)
            elif eradiate.mode().is_ckd:
                return BinSet.from_absorption_data(absorption_data, quad_spec)
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
