"""
Molecular atmospheres.
"""

from __future__ import annotations

import typing as t

import attr
import numpy as np
import pint
import xarray as xr

from ._core import AbstractHeterogeneousAtmosphere, atmosphere_factory
from ..core import KernelDict
from ..phase import PhaseFunction, RayleighPhaseFunction, phase_function_factory
from ...attrs import AUTO, documented, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...radprops.rad_profile import AFGL1986RadProfile, RadProfile, US76ApproxRadProfile
from ...thermoprops import afgl_1986, us76
from ...thermoprops.util import (
    compute_scaling_factors,
    interpolate,
    rescale_concentration,
)
from ...units import to_quantity
from ...units import unit_registry as ureg


@atmosphere_factory.register(type_id="molecular", dict_constructor="afgl_1986")
@parse_docs
@attr.s
class MolecularAtmosphere(AbstractHeterogeneousAtmosphere):
    """
    Molecular atmosphere scene element [``molecular``].

    .. admonition:: Class method constructors

       .. autosummary::

          afgl_1986
          ussa1976
    """

    _thermoprops: xr.Dataset = documented(
        attr.ib(
            factory=us76.make_profile,
            validator=attr.validators.instance_of(xr.Dataset),
        ),
        doc="Thermophysical properties.",
        type="Dataset",
        default=":meth:`us76.make_profile() <eradiate.thermoprops.us76.make_profile>`",
    )

    phase: PhaseFunction = documented(
        attr.ib(
            factory=lambda: RayleighPhaseFunction(),
            converter=phase_function_factory.convert,
            validator=attr.validators.instance_of(PhaseFunction),
        )
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

    absorption_data_sets_mono: t.Dict[str, str] = documented(
        attr.ib(
            factory=dict,
            converter=dict,
            validator=attr.validators.instance_of(dict),
        ),
        doc="Mapping of species and absorption data set files paths. For "
        "species not listed in the mapping, the default absorption data sets "
        "are used."
        "In CKD mode, this parameter is irrelevant and will be ignored.",
        type="dict",
    )

    def update(self) -> None:
        super(MolecularAtmosphere, self).update()
        self.phase.id = self.id_phase

    # --------------------------------------------------------------------------
    #              Spatial extension and thermophysical properties
    # --------------------------------------------------------------------------

    @property
    def bottom(self) -> pint.Quantity:
        return to_quantity(self.thermoprops.z_level).min()

    @property
    def top(self) -> pint.Quantity:
        return to_quantity(self.thermoprops.z_level).max()

    @property
    def thermoprops(self) -> xr.Dataset:
        return self._thermoprops

    def eval_width(self, ctx: KernelDictContext) -> pint.Quantity:
        if ctx is not None and ctx.override_scene_width is not None:
            return ctx.override_scene_width
        else:
            if self.width is AUTO:
                min_sigma_s = self.radprops_profile.eval_sigma_s(ctx.spectral_ctx).min()
                width = np.divide(
                    10.0,
                    min_sigma_s,
                    where=min_sigma_s != 0.0,
                    out=np.array([np.inf]),
                )

                return min(width, ureg.Quantity(1e3, "km"))
            else:
                return self.width

    # --------------------------------------------------------------------------
    #                             Radiative properties
    # --------------------------------------------------------------------------

    @property
    def radprops_profile(self) -> RadProfile:
        if self.thermoprops.title == "U.S. Standard Atmosphere 1976":
            absorption_data_set = (
                None
                if self.absorption_data_sets_mono == {}
                else self.absorption_data_sets_mono["us76_u86_4"]
            )
            return US76ApproxRadProfile(
                thermoprops=self.thermoprops,
                has_scattering=self.has_scattering,
                has_absorption=self.has_absorption,
                absorption_data_set=absorption_data_set,
            )
        elif "AFGL (1986)" in self.thermoprops.title:
            return AFGL1986RadProfile(
                thermoprops=self.thermoprops,
                has_scattering=self.has_scattering,
                has_absorption=self.has_absorption,
                absorption_data_sets=self.absorption_data_sets_mono,
            )
        else:
            raise NotImplementedError("Unsupported thermophysical properties data set.")

    def eval_radprops(self, spectral_ctx: SpectralContext) -> xr.Dataset:
        return self.radprops_profile.eval_dataset(spectral_ctx=spectral_ctx)

    # --------------------------------------------------------------------------
    #                             Kernel dictionary
    # --------------------------------------------------------------------------

    def kernel_phase(self, ctx: KernelDictContext) -> KernelDict:
        return self.phase.kernel_dict(ctx=ctx)

    # --------------------------------------------------------------------------
    #                               Constructors
    # --------------------------------------------------------------------------

    @classmethod
    def afgl_1986(
        cls,
        model: str = "us_standard",
        levels: t.Optional[pint.Quantity] = None,
        concentrations: t.Optional[t.MutableMapping[str, pint.Quantity]] = None,
        **kwargs: t.MutableMapping[str],
    ) -> MolecularAtmosphere:
        """
        Molecular atmosphere based on the AFGL (1986) atmospheric
        thermophysical properties profiles
        :cite:`Anderson1986AtmosphericConstituentProfiles`.

        Parameters
        ----------
        model : {"us_standard", "tropical", "midlatitude_summer", "midlatitude_winter", "subarctic_summer", "subarctic_winter"}, default: "us_standard"
            AFGL (1986) model identifier.

        levels : quantity
            Altitude levels.

        concentrations : dict
            Molecules concentrations as a ``{str: quantity}`` mapping.

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
        ds = afgl_1986.make_profile(model_id=model)

        if levels is not None:
            ds = interpolate(ds=ds, z_level=levels, conserve_columns=True)

        if concentrations is not None:
            factors = compute_scaling_factors(ds=ds, concentration=concentrations)
            ds = rescale_concentration(ds=ds, factors=factors)

        return MolecularAtmosphere(thermoprops=ds, **kwargs)

    @classmethod
    def ussa1976(
        cls,
        levels: t.Optional[pint.Quantity] = None,
        concentrations: t.Optional[t.MutableMapping[str, pint.Quantity]] = None,
        **kwargs: t.MutableMapping[str, t.Any],
    ) -> MolecularAtmosphere:
        """
        Molecular atmosphere based on the US Standard Atmosphere (1976) model
        :cite:`NASA1976USStandardAtmosphere`.

        Parameters
        ----------
        levels : quantity, optional
            Altitude levels. If ``None``, defaults to [0, 1, ..., 99, 100] km.

        concentrations : dict
            Molecules concentrations as a ``{str: quantity}`` mapping.

        **kwargs
            Keyword arguments passed to the :class:`MolecularAtmosphere`
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

        return MolecularAtmosphere(thermoprops=ds, **kwargs)
