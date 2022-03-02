import attr
import numpy as np
import pint
import pinttr
import xarray as xr

from eradiate.units import to_quantity

from ..data import load_dataset
from ._core import RadProfile, make_dataset
from ..attrs import documented
from ..ckd import Bindex
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..validators import is_positive
from eradiate import validators


@attr.s
class ParticleRadProfile(RadProfile):
    """
    Particle radiative properties profile.
    """
    fractions: np.ndarray = documented(
        attr.ib(
            converter=np.ndarray,
            validators=attr.validators.instance_of(np.ndarray)
        ),
        doc="Particle number fractions at cell centers [dimensionless].",
        type="array",
    )

    dataset: xr.Dataset = documented(
        attr.ib(
            converter=xr.Dataset,
            validator=attr.validators.instance_of(xr.Dataset),
        ),
        doc="dataset",
    )

    z_level: pint.Quantity = pinttr.ib(
        units=ucc.deferred("length"),
        doc="Level altitudes. **Required, no default**.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity or array",
    )

    tau_550: pint.Quantity = documented(
        pinttr.ib(
            units=ucc.deferred("dimensionless"),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Extinction optical thickness at the wavelength of 550 nm.\n"
        "\n"
        "Unit-enabled field (default: ucc[dimensionless]).",
        type="quantity",
        init_type="quantity or float",
    )

    def eval_albedo_mono(self, w: pint.Quantity) -> pint.Quantity:
        w_units = self.dataset.w["units"]
        albedo = to_quantity(self.dataset.interp(w=w.m_as(w_units)))
        n_layers = len(self.z_level) - 1
        return albedo * np.ones(n_layers)
    
    def eval_albedo_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        w_units = ureg.nm
        w = [bindex.bin.wcenter.m_as(w_units) for bindex in bindexes] * w_units
        return self.eval_albedo_mono(w)
    
    def eval_sigma_t_mono(self, w: pint.Quantity) -> pint.Quantity:
        w_units = self.dataset.w["units"]
        xs_t = to_quantity(self.dataset.sigma_t.interp(w=w.m_as(w_units)))
        xs_t_550 = to_quantity(
            self.dataset.sigma_t.interp(w=ureg.convert(550.0, ureg.nm, w_units))
        )
        ki = xs_t_550 * self.fractions
        dz = (self.top - self.bottom) / self.n_layers
        sigma_t_normalized = self._normalize_to_tau(
            ki=ki.magnitude,
            dz=dz,
            tau=self.tau_550,
        )
        return sigma_t_normalized * xs_t / xs_t_550
    
    def eval_sigma_t_ckd(self, *bindexes: Bindex) -> pint.Quantity:
        w_units = ureg.nm
        w = [bindex.bin.wcenter.m_as(w_units) for bindex in bindexes] * w_units
        return self.eval_sigma_t_mono(w)
    
    def eval_dataset_mono(self, w: pint.Quantity) -> xr.Dataset:
        return make_dataset(
            wavelength=w,
            z_level=self.levels,
            sigma_t=self.eval_sigma_t_mono(w),
            albedo=self.eval_albedo_mono(w),
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
    
    @staticmethod
    @ureg.wraps(ret="km^-1", args=("", "km", ""), strict=False)
    def _normalize_to_tau(ki: np.ndarray, dz: np.ndarray, tau: float) -> pint.Quantity:
        r"""
        Normalise extinction coefficient values :math:`k_i` so that:

        .. math::
           \sum_i k_i \Delta z = \tau_{550}

        where :math:`\tau` is the particle layer optical thickness.

        Parameters
        ----------
        ki : quantity or ndarray
            Dimensionless extinction coefficients values [].

        dz : quantity or ndarray
            Layer divisions thickness [km].

        tau : float
            Layer optical thickness (dimensionless).

        Returns
        -------
        quantity
            Normalised extinction coefficients.
        """
        return ki * tau / (np.sum(ki) * dz)
