"""
Molecular atmospheres.
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import Dict, MutableMapping, Optional

import attr
import numpy as np
import pint
import xarray as xr

from ._core import Atmosphere, atmosphere_factory, write_binary_grid3d
from ..phase import PhaseFunction, RayleighPhaseFunction, phase_function_factory
from ..._util import onedict_value
from ...attrs import AUTO, documented, parse_docs
from ...contexts import KernelDictContext, SpectralContext
from ...kernel.transform import map_cube, map_unit_cube
from ...radprops.rad_profile import AFGL1986RadProfile, RadProfile, US76ApproxRadProfile
from ...thermoprops import afgl1986, us76
from ...thermoprops.util import (
    compute_scaling_factors,
    interpolate,
    rescale_concentration,
)
from ...units import to_quantity
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


@atmosphere_factory.register(
    type_id="molecular_atmosphere", dict_constructor="afgl1986"
)
@parse_docs
@attr.s
class MolecularAtmosphere(Atmosphere):
    """
    Molecular atmosphere.

    .. admonition:: Class method constructors

       .. autosummary::

          afgl1986
          ussa1976
    """

    _thermoprops: xr.Dataset = documented(
        attr.ib(
            factory=us76.make_profile,
            validator=attr.validators.instance_of(xr.Dataset),
        ),
        doc="Thermophysical properties.",
        type=":class:`~xarray.Dataset`",
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

    absorption_data_sets: Optional[MutableMapping[str, str]] = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(dict),
            validator=attr.validators.optional(attr.validators.instance_of(dict)),
        ),
        doc="Mapping of species and absorption data set files paths. If "
        "``None``, the default absorption data sets are used to compute "
        "the absorption coefficient. If not ``None``, the absorption data "
        "set files whose paths are provided in the mapping will be used to "
        "compute the absorption coefficient. If the mapping does not "
        "include all species from the atmospheric "
        "thermophysical profile, the default data sets will be used to "
        "compute the absorption coefficient of the corresponding species.",
        type="dict",
    )

    albedo_filename: str = documented(
        attr.ib(
            default="albedo.vol",
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Name of the albedo volume data file.",
        type="str",
        default='"albedo.vol"',
    )

    sigma_t_filename: str = documented(
        attr.ib(
            default="sigma_t.vol",
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Name of the extinction coefficient volume data file.",
        type="str",
        default='"sigma_t.vol"',
    )

    cache_dir: pathlib.Path = documented(
        attr.ib(
            default=pathlib.Path(tempfile.mkdtemp()),
            converter=pathlib.Path,
            validator=attr.validators.instance_of(pathlib.Path),
        ),
        doc="Path to a cache directory where volume data files will be created.",
        type="path-like",
        default="Temporary directory",
    )

    def __attrs_post_init__(self) -> None:
        # Prepare cache directory in case we'd need it
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.update()

    def update(self) -> None:
        self.phase.id = f"phase_{self.id}"

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
        if self.width is AUTO:
            spectral_ctx = ctx.spectral_ctx if ctx is not None else None
            min_sigma_s = self.radprops_profile.eval_sigma_s(spectral_ctx).min()

            if min_sigma_s <= 0.0:
                raise ValueError(
                    "cannot compute width automatically when scattering "
                    "coefficient reaches zero"
                )

            return min(10.0 / min_sigma_s, ureg.Quantity(1e3, "km"))
        else:
            return self.width

    # --------------------------------------------------------------------------
    #                             Radiative properties
    # --------------------------------------------------------------------------

    @property
    def radprops_profile(self) -> RadProfile:
        if self.thermoprops.title == "U.S. Standard Atmosphere 1976":
            if self.absorption_data_sets is not None:
                absorption_data_set = self.absorption_data_sets["us76_u86_4"]
            else:
                absorption_data_set = None
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
                absorption_data_sets=self.absorption_data_sets,
            )
        else:
            raise NotImplementedError("Unsupported thermophysical properties data set.")

    def eval_radprops(self, spectral_ctx: SpectralContext) -> xr.Dataset:
        """
        Evaluates the molecular atmosphere's radiative properties.

        Parameter ``spectral_ctx`` (:class:`SpectralContext`):
            Spectral context.

        Returns → :class:`~xarray.Dataset`:
            Radiative properties dataset.
        """
        return self.radprops_profile.to_dataset(spectral_ctx=spectral_ctx)

    # --------------------------------------------------------------------------
    #                             Kernel dictionary
    # --------------------------------------------------------------------------

    @property
    def albedo_file(self) -> pathlib.Path:
        return self.cache_dir / self.albedo_filename

    @property
    def sigma_t_file(self) -> pathlib.Path:
        return self.cache_dir / self.sigma_t_filename

    def kernel_phase(self, ctx: KernelDictContext) -> Dict:
        return self.phase.kernel_dict(ctx=ctx).data

    def kernel_media(self, ctx: KernelDictContext) -> Dict:
        length_units = uck.get("length")
        width = self.kernel_width(ctx).m_as(length_units)
        top = self.top.m_as(length_units)
        bottom = self.bottom.m_as(length_units)
        trafo = map_unit_cube(
            xmin=-width / 2.0,
            xmax=width / 2.0,
            ymin=-width / 2.0,
            ymax=width / 2.0,
            zmin=bottom,
            zmax=top,
        )

        radprops = self.radprops_profile.to_dataset(spectral_ctx=ctx.spectral_ctx)
        albedo = to_quantity(radprops.albedo).m_as(uck.get("albedo"))
        sigma_t = to_quantity(radprops.sigma_t).m_as(uck.get("collision_coefficient"))
        write_binary_grid3d(
            filename=str(self.albedo_file), values=albedo[np.newaxis, np.newaxis, ...]
        )
        write_binary_grid3d(
            filename=str(self.sigma_t_file), values=sigma_t[np.newaxis, np.newaxis, ...]
        )
        if ctx.ref:
            phase = {"type": "ref", "id": f"phase_{self.id}"}
        else:
            phase = onedict_value(self.kernel_phase(ctx=ctx))
        return {
            f"medium_{self.id}": {
                "type": "heterogeneous",
                "phase": phase,
                "albedo": {
                    "type": "gridvolume",
                    "filename": str(self.albedo_file),
                    "to_world": trafo,
                },
                "sigma_t": {
                    "type": "gridvolume",
                    "filename": str(self.sigma_t_file),
                    "to_world": trafo,
                },
            }
        }

    def kernel_shapes(self, ctx: KernelDictContext) -> Dict:
        if ctx.ref:
            medium = {"type": "ref", "id": f"medium_{self.id}"}
        else:
            medium = self.kernel_media(ctx)[f"medium_{self.id}"]

        length_units = uck.get("length")
        width = self.kernel_width(ctx).m_as(length_units)
        bottom = self.bottom.m_as(length_units)
        top = self.top.m_as(length_units)
        offset = self.kernel_offset(ctx).m_as(length_units)
        trafo = map_cube(
            xmin=-width / 2.0,
            xmax=width / 2.0,
            ymin=-width / 2.0,
            ymax=width / 2.0,
            zmin=bottom - offset,
            zmax=top,
        )

        return {
            f"shape_{self.id}": {
                "type": "cube",
                "to_world": trafo,
                "bsdf": {"type": "null"},
                "interior": medium,
            }
        }

    # --------------------------------------------------------------------------
    #                               Constructors
    # --------------------------------------------------------------------------

    @classmethod
    def afgl1986(
        cls: MolecularAtmosphere,
        model: str = "us_standard",
        levels: Optional[pint.Quantity] = None,
        concentrations: Optional[MutableMapping[str, pint.Quantity]] = None,
        **kwargs: MutableMapping[str],
    ) -> MolecularAtmosphere:
        """
        Molecular atmosphere based on the AFGL (1986) atmospheric
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
        this class lets you define a custom altitude mesh (regular or
        irregular).

        All six models include the following six absorbing molecular species:
        H2O, CO2, O3, N2O, CO, CH4 and O2.
        The concentrations of these species in the atmosphere is fixed by
        :cite:`Anderson1986AtmosphericConstituentProfiles`.
        However, this class allows you to rescale the concentrations of each
        individual molecular species to custom concentration values.
        Custom concentrations can be provided in different units.

        Parameter ``model`` (str):
            AFGL (1986) model identifier.

        Parameter ``levels`` (:class:`pint.Quantity`):
            Altitude levels.

        Parameter ``concentrations`` (dict[str, :class:`pint.Quantity`]):
            Molecules concentrations.

        Parameter ``**kwargs`` (dict[str]):
            Keyword arguments passed to :class:`MolecularAtmosphere`'s
            constructor.

        Returns → :class:`MolecularAtmosphere`:
            AFGL (1986) molecular atmosphere.
        """
        ds = afgl1986.make_profile(model_id=model)

        if levels is not None:
            ds = interpolate(ds=ds, z_level=levels, conserve_columns=True)

        if concentrations is not None:
            factors = compute_scaling_factors(ds=ds, concentration=concentrations)
            ds = rescale_concentration(ds=ds, factors=factors)

        return MolecularAtmosphere(thermoprops=ds, **kwargs)

    @classmethod
    def ussa1976(
        cls: MolecularAtmosphere,
        levels: Optional[pint.Quantity] = np.linspace(0, 100, 101) * ureg.km,
        concentrations: Optional[MutableMapping[str, pint.Quantity]] = None,
        **kwargs: MutableMapping[str],
    ) -> MolecularAtmosphere:
        """
        Molecular atmosphere based on the US Standard Atmosphere (1976) model
        :cite:`NASA1976USStandardAtmosphere`.

        Parameter ``levels`` (:class:`pint.Quantity`):
            Altitude levels.

        Parameter ``concentrations`` (dict[str, :class:`pint.Quantity`])
            Molecules concentrations.

        Parameter ``**kwargs`` (dict[str]):
            Keyword arguments passed to :class:`MolecularAtmosphere`'s
            constructor.

        Returns → :class:`MolecularAtmosphere`:
            U.S. Standard Atmosphere (1976) molecular atmosphere object.
        """
        ds = us76.make_profile(levels=levels)

        if concentrations is not None:
            factors = compute_scaling_factors(ds=ds, concentration=concentrations)
            ds = rescale_concentration(ds=ds, factors=factors)

        return MolecularAtmosphere(thermoprops=ds, **kwargs)
