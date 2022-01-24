"""
AFGL (1986) radiative profile definition.
"""

from __future__ import annotations

import typing as t

import attr
import numpy as np
import pint
import xarray as xr

from . import _util_ckd
from ._core import RadProfile, make_dataset, rad_profile_factory
from .absorption import compute_sigma_a
from .rayleigh import compute_sigma_s_air
from ..attrs import documented, parse_docs
from ..ckd import Bindex
from ..thermoprops import afgl_1986
from ..thermoprops.util import (
    compute_column_mass_density,
    compute_column_number_density,
)
from ..units import to_quantity
from ..units import unit_registry as ureg


def _convert_thermoprops_afgl_1986(
    value: t.Union[t.MutableMapping, xr.Dataset]
) -> xr.Dataset:
    if isinstance(value, dict):
        return afgl_1986.make_profile(**value)
    else:
        return value


@rad_profile_factory.register(type_id="afgl_1986")
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
            factory=lambda: afgl_1986.make_profile(),
            converter=_convert_thermoprops_afgl_1986,
            validator=attr.validators.instance_of(xr.Dataset),
        ),
        doc="Thermophysical properties.",
        type=":class:`~xarray.Dataset`",
        default=":func:`~eradiate.thermoprops.afgl_1986.make_profile`",
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
        absorber: str,
        n_absorber: pint.Quantity,
        p: pint.Quantity,
        t: pint.Quantity,
    ) -> pint.Quantity:
        """
        Compute absorption coefficient using the predefined absorption data set.
        """
        raise NotImplementedError("To be refactored")
        # ! This method is never tested because it requires large absorption
        # data sets to be downloaded
        wavenumber = 1.0 / wavelength
        try:
            dataset_id = find_dataset(
                wavenumber=wavenumber,
                absorber=absorber,
                engine=Engine.SPECTRA,
            )
            with _util_ckd.open_dataset(
                category="absorption_spectrum", id=dataset_id
            ) as dataset:
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
        with xr.open_dataset(path) as dataset:
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

        Parameters
        ----------
        w : quantity
            Wavelength array (may be scalar as well).

        Returns
        -------
        quantity
            Computed absorption coefficient array with the same shape as `w`.

        Notes
        -----
        Extrapolate to zero when wavelength, pressure and/or temperature are out
        of bounds.
        """
        profile = self.thermoprops

        if not self.has_absorption:
            return ureg.Quantity(np.zeros(profile.z_layer.size), "km^-1")

        # else:
        wavelength = w

        p = to_quantity(profile.p)
        t = to_quantity(profile.t)
        n = to_quantity(profile.n)
        mr = profile.mr

        sigma_a = np.full(mr.shape, np.nan)
        absorbers = ["CH4", "CO", "CO2", "H2O", "N2O", "O2", "O3"]

        for i, absorber in enumerate(absorbers):
            n_absorber = n * to_quantity(mr.sel(species=absorber))

            if absorber in self.absorption_data_sets:
                sigma_a_absorber = self._compute_sigma_a_absorber_from_data_set(
                    path=self.absorption_data_sets[absorber],
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

    def eval_sigma_a_ckd(self, *bindexes: Bindex, bin_set_id: str) -> pint.Quantity:
        if bin_set_id is None:
            raise ValueError("argument 'bin_set_id' is required")

        with _util_ckd.open_dataset(f"afgl_1986-us_standard-{bin_set_id}") as ds:
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
            return ureg.Quantity(np.zeros((1, thermoprops.z_layer.size)), "km^-1")

    def eval_sigma_s_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        wavelengths = ureg.Quantity(
            np.array([bindex.bin.wcenter.m_as("nm") for bindex in bindexes]), "nm"
        )
        return self.eval_sigma_s_mono(w=wavelengths)

    def eval_albedo_mono(self, w: pint.Quantity) -> pint.Quantity:
        return (self.eval_sigma_s_mono(w) / self.eval_sigma_t_mono(w)).to(
            ureg.dimensionless
        )

    def eval_albedo_ckd(self, *bindexes: Bindex, bin_set_id: str) -> pint.Quantity:
        sigma_s = self.eval_sigma_s_ckd(*bindexes)
        sigma_t = self.eval_sigma_t_ckd(*bindexes, bin_set_id=bin_set_id)
        return np.divide(
            sigma_s, sigma_t, where=sigma_t != 0.0, out=np.zeros_like(sigma_s)
        ).to("dimensionless")

    def eval_sigma_t_mono(self, w: pint.Quantity) -> pint.Quantity:
        return self.eval_sigma_s_mono(w) + self.eval_sigma_a_mono(w)

    def eval_sigma_t_ckd(self, *bindexes: Bindex, bin_set_id: str) -> pint.Quantity:
        return self.eval_sigma_a_ckd(
            *bindexes, bin_set_id=bin_set_id
        ) + self.eval_sigma_s_ckd(*bindexes)

    def eval_dataset_mono(self, w: pint.Quantity) -> xr.Dataset:
        return make_dataset(
            wavelength=w,
            z_level=to_quantity(self.thermoprops.z_level),
            z_layer=to_quantity(self.thermoprops.z_layer),
            sigma_a=self.eval_sigma_a_mono(w),
            sigma_s=self.eval_sigma_s_mono(w),
        ).squeeze()

    def eval_dataset_ckd(self, *bindexes: Bindex, bin_set_id: str) -> xr.Dataset:
        if len(bindexes) > 1:
            raise NotImplementedError
        else:
            return make_dataset(
                wavelength=bindexes[0].bin.wcenter,
                z_level=to_quantity(self.thermoprops.z_level),
                z_layer=to_quantity(self.thermoprops.z_layer),
                sigma_a=self.eval_sigma_a_ckd(*bindexes, bin_set_id=bin_set_id),
                sigma_s=self.eval_sigma_s_ckd(*bindexes),
            ).squeeze()
