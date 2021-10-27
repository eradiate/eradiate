import itertools
import typing as t
from collections import OrderedDict

import attr
import numpy as np
import xarray as xr

import eradiate

from ._core import PipelineStep
from .._mode import ModeFlags
from ..attrs import documented, parse_docs
from ..scenes.measure import Measure
from ..units import symbol
from ..units import unit_context_config as ucc
from ..units import unit_context_kernel as uck


@parse_docs
@attr.s
class AggregateSampleCount(PipelineStep):
    """
    Aggregate sample count.

    This post-processing pipeline step aggregates sample counts:

    * it computes the average of sensor values weighted by the sample count;
    * it sums the ``spp`` dimension.

    The ``spp_index`` dimension is dropped during this step and the ``spp``
    variable ends up with no dimension.
    """

    def transform(self, x: t.Any) -> t.Any:
        with xr.set_options(keep_attrs=True):
            result = x.weighted(x.spp).mean(dim="spp_index")
            result["spp"] = x.spp.sum()

        return result


@parse_docs
@attr.s
class AggregateCKDQuad(PipelineStep):
    """
    Compute CKD quadrature.
    """

    measure: Measure = documented(
        attr.ib(
            validator=attr.validators.instance_of(Measure),
            repr=lambda self: f"{self.__class__.__name__}(id='{self.id}', ...)",
        ),
        doc="Measure from which the processed data was obtained.",
        type=":class:`.Measure`",
    )

    var: str = documented(
        attr.ib(default="img", validator=attr.validators.instance_of(str)),
        doc="Name of the variable for which CKD quadrature computation "
        "is to be performed.",
        type="str",
        default='"img"',
    )

    def transform(self, x: t.Any) -> t.Any:
        # If not in CKD mode, this step is a no-op
        if not eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return x

        # Otherwise, compute quadrature spectrum-indexed variables and turn spp
        # into a per-bin average

        # Deduplicate bin list preserving order
        bins = list(OrderedDict.fromkeys(x.bin.to_index()))
        n_bins = len(bins)

        # Collect quadrature data
        quad = self.measure.spectral_cfg.bin_set.quad

        # Collect wavelengths associated with each bin
        wavelength_units = ucc.get("wavelength")
        wavelengths = [
            bin.wcenter.m_as(wavelength_units)
            for bin in self.measure.spectral_cfg.bin_set.select_bins(
                ("ids", {"ids": bins})
            )
        ]

        result = x
        var = self.var

        # Get dimensions of current variable
        img = x.data_vars[var]
        dims = OrderedDict((y, len(img.coords[y])) for y in img.dims)

        if "bin" not in dims:
            raise ValueError(f"variable '{var}' is missing dimension 'bin'")

        if "index" not in dims:
            raise ValueError(f"variable '{var}' is missing dimension 'index'")

        # Init storage
        del dims["bin"]
        del dims["index"]

        aggregated = xr.DataArray(
            np.zeros([n_bins] + list(dims.values())),
            coords={"bin": img.bin, **{dim: img.coords[dim] for dim in dims}},
        )

        # For each bin and each pixel, compute quadrature and store the result
        for i_bin, bin in enumerate(bins):
            values_at_nodes = img.sel(bin=bin).values

            # Rationale: Avoid using xarray's indexing in this loop for
            # performance reasons (wrong data indexing method will result in
            # 10x+ speed reduction)
            for indexes in itertools.product(*[list(range(n)) for n in dims.values()]):
                aggregated.values[(i_bin, *indexes)] = quad.integrate(
                    values_at_nodes[(slice(None), *indexes)],
                    interval=np.array([0.0, 1.0]),
                )

        result = result.assign({var: aggregated})
        result[var].attrs = x[var].attrs

        # Add spectral coordinate
        result = result.assign_coords(
            {
                "w": (
                    "bin",
                    wavelengths,
                    {
                        "standard_name": "wavelength",
                        "long_description": "wavelength",
                        "units": symbol(wavelength_units),
                    },
                )
            }
        )

        # Final output has a spectral coordinate but retains bin ID coordinate
        return result.drop_dims("index")


@parse_docs
@attr.s
class AggregateRadiosity(PipelineStep):
    """
    Aggregate flux density field.
    """

    sector_radiosity_var: str = documented(
        attr.ib(default="sector_radiosity", validator=attr.validators.instance_of(str)),
        doc="Name of the variable containing radiosity values for the "
        "hemisphere sector corresponding to each film pixel. This quantity is "
        "expressed in flux units (typically W/m²) and, when summed over the "
        "entire film, aggregates into a radiosity.",
        type="str",
        default='"sector_radiosity"',
    )

    radiosity_var: str = documented(
        attr.ib(default="radiosity", validator=attr.validators.instance_of(str)),
        doc="Name of the variable storing the computed radiosity "
        "(leaving flux) value.",
        type="str",
        default='"radiosity"',
    )

    def transform(self, x: t.Any) -> t.Any:
        result = x.copy(deep=False)

        result[self.radiosity_var] = result[self.var].sum(dim=("x", "y"))
        result[self.radiosity_var].attrs = {
            "standard_name": "toa_outgoing_flux_density_per_unit_wavelength",
            "long_name": "top-of-atmosphere outgoing spectral flux density",
            "units": symbol(uck.get("irradiance")),
        }

        return result
