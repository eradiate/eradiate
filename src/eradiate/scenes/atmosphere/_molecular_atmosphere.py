"""
Molecular atmospheres.
"""

from __future__ import annotations

import typing as t

import attrs
import numpy as np
import pint
import pinttr
import xarray as xr

import eradiate

from ._core import AbstractHeterogeneousAtmosphere
from ..core import traverse
from ..phase import PhaseFunction, RayleighPhaseFunction, phase_function_factory
from ... import converters
from ...attrs import documented, parse_docs
from ...contexts import KernelContext
from ...quad import Quad
from ...radprops import AFGL1986RadProfile, RadProfile, US76ApproxRadProfile, ZGrid
from ...spectral.ckd import BinSet
from ...spectral.index import SpectralIndex
from ...spectral.mono import WavelengthSet
from ...thermoprops import afgl_1986, us76
from ...thermoprops.util import (
    compute_scaling_factors,
    interpolate,
    rescale_concentration,
)
from ...units import unit_registry as ureg
from ...util.misc import summary_repr


@parse_docs
@attrs.define(eq=False, slots=False)
class MolecularAtmosphere(AbstractHeterogeneousAtmosphere):
    """
    Molecular atmosphere scene element [``molecular``].

    .. admonition:: Class method constructors

       .. autosummary::

          afgl_1986
          ussa_1976
    """

    _thermoprops: xr.Dataset = documented(
        attrs.field(
            factory=us76.make_profile,
            validator=attrs.validators.instance_of(xr.Dataset),
            repr=summary_repr,
        ),
        doc="Thermophysical properties.",
        type="Dataset",
        default=":meth:`us76.make_profile() <eradiate.thermoprops.us76.make_profile>`",
    )

    _phase: PhaseFunction = documented(
        attrs.field(
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
            default=True,
            converter=bool,
        ),
        doc="Absorption switch. If ``True``, the absorption coefficient is "
        "computed. Else, the absorption coefficient is not computed and "
        "instead set to zero.",
        type="bool",
        default="True",
    )

    has_scattering: bool = documented(
        attrs.field(
            default=True,
            converter=bool,
        ),
        doc="Scattering switch. If ``True``, the scattering coefficient is "
        "computed. Else, the scattering coefficient is not computed and "
        "instead set to zero.",
        type="bool",
        default="True",
    )

    @has_absorption.validator
    @has_scattering.validator
    def _switch_validator(self, attribute, value):
        if not self.has_absorption and not self.has_scattering:
            raise ValueError(
                f"while validating {attribute.name}: at least one of 'has_absorption' "
                "and 'has_scattering' must be True"
            )

    absorption_dataset: xr.Dataset | None = documented(
        attrs.field(
            default=None,
            converter=attrs.converters.optional(
                converters.to_dataset(load_from_id=None)
            ),
            validator=attrs.validators.optional(
                attrs.validators.instance_of(xr.Dataset)
            ),
        ),
        doc="Absorption coefficient dataset. If ``None``, the absorption "
        "coefficient is set to zero.",
        type="Dataset",
        init_type="PathLike or Dataset",
        default="None",
    )

    _radprops_profile: RadProfile | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.instance_of(RadProfile)),
        init=False,
        repr=False,
    )

    @property
    def spectral_set(self) -> None | BinSet | WavelengthSet:
        if self.has_absorption:
            if eradiate.mode().is_mono:
                return WavelengthSet.from_absorption_dataset(
                    dataset=self.absorption_dataset
                )
            elif eradiate.mode().is_ckd:
                return BinSet.from_absorption_dataset(
                    dataset=self.absorption_dataset,
                    quad=Quad.gauss_legendre(16),  # TODO: PR#311 hack
                )
            else:
                raise NotImplementedError
        else:
            return None

    def update(self) -> None:
        # Inherit docstring

        self.phase.id = self.phase_id

        if self.thermoprops.title == "U.S. Standard Atmosphere 1976":
            self._radprops_profile = US76ApproxRadProfile(
                thermoprops=self.thermoprops,
                has_scattering=self.has_scattering,
                has_absorption=self.has_absorption,
                absorption_dataset=self.absorption_dataset,
            )
        elif "AFGL (1986)" in self.thermoprops.title:
            self._radprops_profile = AFGL1986RadProfile(
                thermoprops=self.thermoprops,
                has_scattering=self.has_scattering,
                has_absorption=self.has_absorption,
                absorption_dataset=self.absorption_dataset,
            )
        else:
            raise NotImplementedError("Unsupported thermophysical property data set.")

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
        self, si: SpectralIndex, zgrid: ZGrid | None = None
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

    # --------------------------------------------------------------------------
    #                               Constructors
    # --------------------------------------------------------------------------

    @classmethod
    def afgl_1986(
        cls,
        model: str = "us_standard",
        levels: pint.Quantity | None = None,
        concentrations: dict[str, str | pint.Quantity] | None = None,
        binset: str = "10nm",
        **kwargs: dict[str],
    ) -> MolecularAtmosphere:
        """
        Molecular atmosphere based on the AFGL (1986) atmospheric
        thermophysical properties profiles
        :cite:`Anderson1986AtmosphericConstituentProfiles` (CKD mode only).

        Parameters
        ----------
        model : {"us_standard", "tropical", "midlatitude_summer", \
            "midlatitude_winter", "subarctic_summer", "subarctic_winter"}, \
            optional, default: "us_standard"
            AFGL (1986) model identifier.

        levels : quantity
            Altitude levels.

        concentrations : dict
            Molecular concentrations as a ``{str: quantity}`` mapping.
            This dictionary is interpreted by :func:`pinttr.util.ensure_units`,
            which allows for passing units as strings.

        binset: str
            Wavelength bin set identifier. Either ``"10nm"`` or ``"1nm"``.

        **kwargs
            Keyword arguments passed to the :class:`.MolecularAtmosphere`
            constructor.

        Returns
        -------
        :class:`MolecularAtmosphere`
            AFGL (1986) molecular atmosphere.

        Notes
        -----
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
           :cite:`Anderson1986AtmosphericConstituentProfiles` is a piecewise
           regular altitude mesh with an altitude step of 1 km from 0 to 25 km,
           2.5 km from 25 km to 50 km and 5 km from 50 km to 120 km.
           Since the Eradiate kernel only supports regular altitude mesh, the
           original atmospheric thermophysical properties profiles were
           interpolated on the regular altitude mesh with an altitude step of 1 km
           from 0 to 120 km.

        Although the altitude meshes of the interpolated
        :cite:`Anderson1986AtmosphericConstituentProfiles` profiles is fixed,
        this class lets you define a custom altitude mesh (regular or
        irregular).

        All six models include the following six absorbing molecular species:
        H2O, CO2, O3, N2O, CO, CH4 and O2.
        The concentrations of these species in the atmosphere is fixed by
        :cite:`Anderson1986AtmosphericConstituentProfiles`.
        However, this class allows you to rescale the concentrations of each
        individual molecular species to custom concentration values.
        Custom concentrations can be provided in different units.
        """
        if "absorption_dataset" in kwargs:
            raise TypeError(
                "Cannot pass 'absorption_dataset' keyword argument. The "
                "'afgl_1986' sets the absorption dataset automatically."
            )

        thermoprops = afgl_1986.make_profile(model_id=model)

        path = "ckd/absorption"
        if kwargs.get("has_absorption", True):
            if binset == "10nm":
                absorption_dataset = f"{path}/10nm/afgl_1986-{model}-10nm-v3.nc"
            elif binset == "1nm":
                absorption_dataset = f"{path}/1nm/afgl_1986-{model}-1nm-v3.nc"
            else:
                raise ValueError(f"Invalid binset: {binset}")
        else:
            absorption_dataset = None

        if concentrations is not None:
            factors = compute_scaling_factors(
                ds=thermoprops,
                concentration=pinttr.interpret_units(concentrations, ureg=ureg),
            )
            thermoprops = rescale_concentration(ds=thermoprops, factors=factors)

        if levels is not None:
            thermoprops = interpolate(
                ds=thermoprops,
                z_level=levels,
                conserve_columns=True,
            )

        return cls(
            thermoprops=thermoprops,
            absorption_dataset=absorption_dataset,
            **kwargs,
        )

    @classmethod
    def ussa_1976(
        cls,
        levels: pint.Quantity | None = None,
        concentrations: dict[str, pint.Quantity] | None = None,
        **kwargs: dict[str, t.Any],
    ) -> MolecularAtmosphere:
        """
        Molecular atmosphere based on the US Standard Atmosphere (1976) model
        :cite:`NASA1976USStandardAtmosphere` (monochromatic mode only).

        Parameters
        ----------
        levels : quantity, optional
            Altitude levels. If ``None``, defaults to [0, 1, ..., 99, 100] km.

        concentrations : dict
            Molecules concentrations as a ``{str: quantity}`` mapping.

        **kwargs
            Keyword arguments passed to the :class:`.MolecularAtmosphere`
            constructor.

        Returns
        -------
        :class:`.MolecularAtmosphere`
            U.S. Standard Atmosphere (1976) molecular atmosphere object.
        """
        if levels is None:
            levels = np.linspace(0, 100, 101) * ureg.km

        ds = us76.make_profile(levels=levels)

        if concentrations is not None:
            factors = compute_scaling_factors(ds=ds, concentration=concentrations)
            ds = rescale_concentration(ds=ds, factors=factors)

        return cls(thermoprops=ds, **kwargs)
