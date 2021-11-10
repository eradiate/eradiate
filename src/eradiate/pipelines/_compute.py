import typing as t

import attr
import numpy as np
import pint
import xarray as xr
from pinttr.util import always_iterable

from ._core import PipelineStep
from ..attrs import documented, parse_docs
from ..scenes.measure import Measure
from ..scenes.spectra import InterpolatedSpectrum
from ..units import symbol, to_quantity
from ..units import unit_registry as ureg


@parse_docs
@attr.s
class ApplySpectralResponseFunction(PipelineStep):
    """
    Apply spectral response function to specified variables.

    This post-processing pipeline step applies the spectral response function
    to specified variables. It creates new corresponding data variables with no
    dependency against the wavelength dimension.

    Notes
    -----
    The processed dataset is expected to have a ``bin`` coordinate, associated
    with bounds ``wmin`` and ``wmax``. If not, it becomes a no-op. In practice,
    this means that nothing will happen in monochromatic modes.
    """

    measure: Measure = documented(
        attr.ib(
            validator=attr.validators.instance_of(Measure),
            repr=lambda self: f"{self.__class__.__name__}(id='{self.id}', ...)",
        ),
        doc="A :class:`.Measure` instance from which the processed data originates.",
        type=":class:`.Measure`",
    )

    vars: t.List[str] = documented(
        attr.ib(
            factory=list,
            converter=lambda x: list(always_iterable(x)),
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(str)
            ),
        ),
        doc="List of variables to which the spectral response function is to be "
        "applied.",
        type="list of str",
        init_type="str or list of str",
        default="[]",
    )

    def transform(self, x: t.Any) -> t.Any:
        with xr.set_options(keep_attrs=True):
            result = x.copy(deep=False)
        measure = self.measure

        # Evaluate integral of spectral response function within selected interval
        wmin = to_quantity(result.bin_wmin).min()
        wmax = to_quantity(result.bin_wmax).max()
        srf = measure.spectral_cfg.srf
        srf_int = srf.integral(wmin, wmax)

        for var in self.vars:
            # Evaluate integral of product of variable and SRF within selected interval
            # Note: Spectral grid is the finest between data and SRF grids
            data_w = to_quantity(result.w)
            srf_w = srf.wavelengths
            w_units = data_w.units

            w_m = np.array(sorted(set(data_w.m_as(w_units)) | set(srf_w.m_as(w_units))))
            var_values = result[var].interp(
                w=w_m, method="nearest", kwargs={"fill_value": "extrapolate"}
            )

            srf_values = srf.eval_mono(w_m * w_units).reshape(
                [-1 if dim == "w" else 1 for dim in var_values.dims]
            )
            var_srf_int = (var_values * srf_values).integrate("w")

            # Apply SRF to variable and store result
            dims = list(result[var].dims)
            dims.remove("w")
            result[f"{var}_srf"] = (dims, var_srf_int.values / srf_int.m_as(w_units))
            attrs = result[var].attrs.copy()

            if "standard_name" in attrs:
                attrs["standard_name"] += "_srf"
            if "long_name" in attrs:
                attrs["long_name"] += " (SRF applied)"
            result[f"{var}_srf"].attrs = attrs

        return result


@parse_docs
@attr.s
class ComputeReflectance(PipelineStep):
    """
    Derive reflectance from radiance and irradiance values.
    """

    radiance_var: str = documented(
        attr.ib(default="radiance", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing leaving radiance values.",
        type="str",
        default='"radiance"',
    )

    irradiance_var: str = documented(
        attr.ib(default="irradiance", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing irradiance (incoming flux) values.",
        type="str",
        default='"irradiance"',
    )

    brdf_var: str = documented(
        attr.ib(default="brdf", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing BRDF values.",
        type="str",
        default='"brdf"',
    )

    brf_var: str = documented(
        attr.ib(default="brf", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing BRF values.",
        type="str",
        default='"brf"',
    )

    def transform(self, x: t.Any) -> t.Any:
        # Compute BRDF and BRF
        result = x.copy(deep=False)

        # We assume that all quantities are stored in kernel units
        result[self.brdf_var] = result[self.radiance_var] / result[self.irradiance_var]
        result[self.brdf_var].attrs = {
            "standard_name": "brdf",
            "long_name": "bi-directional reflection distribution function",
            "units": symbol("1/sr"),
        }

        result[self.brf_var] = result[self.brdf_var] * np.pi
        result[self.brf_var].attrs = {
            "standard_name": "brf",
            "long_name": "bi-directional reflectance factor",
            "units": symbol("dimensionless"),
        }

        return result


@parse_docs
@attr.s
class ComputeAlbedo(PipelineStep):
    """
    Derive the albedo from radiosity and irradiance fields.
    """

    radiosity_var: str = documented(
        attr.ib(default="radiosity", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing the radiosity (leaving flux) value.",
        type="str",
        default='"radiosity"',
    )

    irradiance_var: str = documented(
        attr.ib(default="irradiance", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing the irradiance (incoming flux) value.",
        type="str",
        default='"irradiance"',
    )

    albedo_var: str = documented(
        attr.ib(default="albedo", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing the albedo value.",
        type="str",
        default='"albedo"',
    )

    def transform(self, x: t.Any) -> t.Any:
        # Compute albedo
        result = x.copy(deep=False)

        # We assume that all quantities are stored in kernel units
        result[self.albedo_var] = (
            result[self.radiosity_var] / result[self.irradiance_var]
        )
        result[self.albedo_var].attrs = {
            "standard_name": "albedo",
            "long_name": "surface albedo",
            "units": "",
        }

        return result
