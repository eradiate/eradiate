import attrs
import numpy as np
import pint
import pinttrs
import xarray as xr

from ..attrs import define, documented
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg


@define
class ParticleProperties:
    data: xr.Dataset = documented(attrs.field())
    _w: pint.Quantity | None = pinttrs.field(
        default=None,
        units=ucc.deferred("wavelength"),
        init=False,
        repr=False,
    )
    _phase: xr.DataArray | None = attrs.field(default=None, init=False, repr=False)
    _ext: pint.Quantity | None = attrs.field(default=None, init=False, repr=False)
    _ssa: pint.Quantity | None = attrs.field(default=None, init=False, repr=False)

    @property
    def w(self) -> pint.Quantity:
        if self._w is None:
            self._w = to_quantity(self.data["w"]).to(ucc.get("wavelength"))
        return self._w

    @property
    def ext(self) -> pint.Quantity:
        if self._ext is None:
            self._ext = to_quantity(self.data["ext"]).to(
                ucc.get("collision_coefficient")
            )
        return self._ext

    @property
    def ssa(self) -> pint.Quantity:
        if self._ssa is None:
            self._ssa = self.data["ssa"].values * ureg.dimensionless
        return self._ssa

    @property
    def phase(self) -> xr.DataArray:
        if self._phase is None:
            self._phase = self.data["phase"].transpose("phamat", "iangle", "w")
        return self._phase

    def eval_ext(self, w: pint.Quantity) -> pint.Quantity:
        return np.interp(w, self.w, self.ext)

    def eval_ssa(self, w: pint.Quantity) -> pint.Quantity:
        return np.interp(w, self.w, self.ssa)

    def eval_phase(self, w: pint.Quantity) -> pint.Quantity:
        from axsdb.math import interp1d

        w = np.atleast_1d(w.m)
        return interp1d(self.w.m, self.phase.values, w)
