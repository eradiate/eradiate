from __future__ import annotations

import logging
import typing as t

import attrs
import numpy as np
from pinttr.util import always_iterable

import eradiate

from ._core import PipelineStep
from ..attrs import documented, parse_docs
from ..scenes.measure import Measure
from ..scenes.spectra import InterpolatedSpectrum, UniformSpectrum
from ..units import symbol, to_quantity
from ..units import unit_registry as ureg

logger = logging.getLogger(__name__)


@parse_docs
@attrs.define
class ApplySpectralResponseFunction(PipelineStep):
    r"""
    Apply spectral response function to specified variables.

    This post-processing pipeline step applies the spectral response function
    to specified variables. It creates new corresponding data variables with no
    dependency against the wavelength dimension.

    Notes
    -----
    The processed dataset is expected to have a ``bin`` coordinate, associated
    with bounds ``bin_wmin`` and ``bin_wmax``. If not, :meth:`transform` will
    raise an exception.

    A common nomenclature practice refers to this process as "convolution"
    :cite:`Burggraaff2020BiasesIncorrectReflectance`,
    but we prefer to avoid this term as we consider it to refer to
    another mathematical operation defined by:

    .. math::

        (f \ast g) \, (t) := \int_{-\infty}^{+\infty} f(\tau) \, g(t - \tau) \, d\tau

    Instead, we adopt the term "spectral response weighted average" since the
    operation is defined by:

    .. math::

        \overline{x} = \frac{\int x(\lambda) \, w(\lambda) \, d\lambda}
                    {\int w(\lambda) \, d\lambda}

    where

    * :math:`x` is the variable to be weighted,
    * :math:`w` is the spectral response function,

    and which effectively translates into a weighted average in numerical form,
    with the weights being the spectral response function values.
    """

    measure: Measure = documented(
        attrs.field(
            validator=attrs.validators.instance_of(Measure),
            repr=lambda self: f"{self.__class__.__name__}(id='{self.id}', ...)",
        ),
        doc="A :class:`.Measure` instance from which the processed data originates.",
        type=":class:`.Measure`",
    )

    vars: list[str] = documented(
        attrs.field(
            factory=list,
            converter=lambda x: list(always_iterable(x)),
            validator=attrs.validators.deep_iterable(
                member_validator=attrs.validators.instance_of(str)
            ),
        ),
        doc="List of variables to which the spectral response function is to be "
        "applied.",
        type="list of str",
        init_type="str or list of str",
        default="[]",
    )

    def transform(self, x: t.Any) -> t.Any:
        if eradiate.mode().is_ckd:
            return self._transform_ckd(x)
        elif eradiate.mode().is_mono:
            return self._transform_mono(x)
        else:
            raise NotImplementedError

    def _transform_ckd(self, x: t.Any):
        result = x.copy(deep=False)

        if not "w" in result.coords:
            raise ValueError("input data is missing a 'w' coordinate")
        if not "wbounds" in result.data_vars:
            raise ValueError("input data is missing a 'wbounds' data variable")

        measure = self.measure

        # Evaluate integral of spectral response function within selected interval
        data_w = to_quantity(result.w)
        w_units = data_w.units
        wbounds = to_quantity(result.wbounds)
        wmin, wmax = wbounds.min(), wbounds.max()
        srf = measure.srf
        srf_int = srf.integral(wmin, wmax)

        if isinstance(srf, InterpolatedSpectrum):
            srf_w = srf.wavelengths
        else:
            raise TypeError(f"unhandled SRF type '{srf.__class__.__name__}'")

        for var in self.vars:
            # Evaluate integral of product of variable and SRF within selected interval

            # Spectral grid is the finest between data and SRF grids
            w_m = np.array(sorted(set(data_w.m_as(w_units)) | set(srf_w.m_as(w_units))))

            # If data var has length 1 on spectral dimension, directly select
            # the value instead of using interpolation (it's a known scipy issue)
            if result[var].w.size == 1:
                # Note: The tricky thing is to recreate and extend the 'w'
                # dimension with the same axis index as in the original data
                var_values = (
                    result[var]
                    .isel(w=0, drop=True)
                    .expand_dims(w=w_m, axis=result[var].get_axis_num("w"))
                )

            # Otherwise, use nearest neighbour interpolation (we assume that var
            # is constant over each spectral bin)
            else:
                var_values = result[var].interp(
                    w=w_m, method="nearest", kwargs={"fill_value": "extrapolate"}
                )

            srf_values = (
                srf.eval_mono(w_m * w_units)
                .reshape([-1 if dim == "w" else 1 for dim in var_values.dims])
                .magnitude
            )
            assert isinstance(srf_values, np.ndarray)  # Check for leftover bugs
            var_srf_int = (var_values * srf_values).integrate("w")

            # Apply SRF to variable and store result
            dims = list(result[var].dims)
            dims.remove("w")
            result[f"{var}_srf"] = (dims, var_srf_int.values / srf_int.m_as(w_units))
            attrs = result[var].attrs.copy()

            if "standard_name" in attrs:
                attrs["standard_name"] += "_srf"
            if "long_name" in attrs:
                attrs["long_name"] += " (SRF-weighted)"
            result[f"{var}_srf"].attrs = attrs

        logger.debug("ApplySpectralResponseFunction pipeline step: end")

        return result

    def _transform_mono(self, x: t.Any) -> t.Any:
        result = x.copy(deep=False)

        if not {"w"}.issubset(set(result.coords.keys())):
            raise ValueError("input data is missing a 'w' coordinate")

        measure = self.measure

        # Evaluate integral of spectral response function within selected interval
        data_w = to_quantity(result.w)
        w_units = data_w.units
        wmin, wmax = data_w.min(), data_w.max()
        srf = measure.srf

        if isinstance(srf, InterpolatedSpectrum):
            srf_w = srf.wavelengths
        else:
            raise TypeError(f"unhandled SRF type '{srf.__class__.__name__}'")

        for var in self.vars:
            # Evaluate integral of product of variable and SRF within selected
            # interval

            # Spectral grid is the finest between data and SRF grids
            w_m = np.array(sorted(set(data_w.m_as(w_units)) | set(srf_w.m_as(w_units))))

            # If data var has length 1 on spectral dimension, directly select
            # the value instead of using interpolation (it's a known scipy
            # issue)
            if result[var].w.size == 1:
                # Note: The tricky thing is to recreate and extend the 'w'
                # dimension with the same axis index as in the original data
                var_values = (
                    result[var]
                    .isel(w=0, drop=True)
                    .expand_dims(w=w_m, axis=result[var].get_axis_num("w"))
                )

            # Otherwise, use nearest neighbour interpolation (we assume that var
            # is constant over each spectral bin)
            else:
                var_values = result[var].interp(w=w_m, method="linear")

            srf_values = (
                srf.eval_mono(w_m * w_units)
                .reshape([-1 if dim == "w" else 1 for dim in var_values.dims])
                .magnitude
            )
            assert isinstance(srf_values, np.ndarray)  # Check for leftover bugs
            var_srf_int = (var_values * srf_values).integrate("w")

            # Apply SRF to variable and store result
            dims = list(result[var].dims)
            dims.remove("w")
            srf_int = srf.integral(wmin, wmax)

            result[f"{var}_srf"] = (dims, var_srf_int.values / srf_int.m_as(w_units))
            attrs = result[var].attrs.copy()

            if "standard_name" in attrs:
                attrs["standard_name"] += "_srf"
            if "long_name" in attrs:
                attrs["long_name"] += " (SRF-weighted)"
            result[f"{var}_srf"].attrs = attrs

        logger.debug("ApplySpectralResponseFunction pipeline step: end")

        return result


@parse_docs
@attrs.define
class ComputeReflectance(PipelineStep):
    """
    Derive reflectance from radiance and irradiance values.
    """

    radiance_var: str = documented(
        attrs.field(default="radiance", validator=attrs.validators.instance_of(str)),
        doc="Name of the variable storing leaving radiance values.",
        type="str",
        default='"radiance"',
    )

    irradiance_var: str = documented(
        attrs.field(default="irradiance", validator=attrs.validators.instance_of(str)),
        doc="Name of the variable storing irradiance (incoming flux) values.",
        type="str",
        default='"irradiance"',
    )

    brdf_var: str = documented(
        attrs.field(default="brdf", validator=attrs.validators.instance_of(str)),
        doc="Name of the variable storing BRDF values.",
        type="str",
        default='"brdf"',
    )

    brf_var: str = documented(
        attrs.field(default="brf", validator=attrs.validators.instance_of(str)),
        doc="Name of the variable storing BRF values.",
        type="str",
        default='"brf"',
    )

    def transform(self, x: t.Any) -> t.Any:
        logger.debug("ComputeReflectance pipeline step: begin")
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
        logger.debug("ComputeReflectance pipeline step: end")

        return result


@parse_docs
@attrs.define
class ComputeAlbedo(PipelineStep):
    """
    Derive the albedo from radiosity and irradiance fields.
    """

    radiosity_var: str = documented(
        attrs.field(default="radiosity", validator=attrs.validators.instance_of(str)),
        doc="Name of the variable storing the radiosity (leaving flux) value.",
        type="str",
        default='"radiosity"',
    )

    irradiance_var: str = documented(
        attrs.field(default="irradiance", validator=attrs.validators.instance_of(str)),
        doc="Name of the variable storing the irradiance (incoming flux) value.",
        type="str",
        default='"irradiance"',
    )

    albedo_var: str = documented(
        attrs.field(default="albedo", validator=attrs.validators.instance_of(str)),
        doc="Name of the variable storing the albedo value.",
        type="str",
        default='"albedo"',
    )

    def transform(self, x: t.Any) -> t.Any:
        logger.debug("ComputeAlbedo pipeline step: begin")
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
        logger.debug("ComputeAlbedo pipeline step: end")

        return result
