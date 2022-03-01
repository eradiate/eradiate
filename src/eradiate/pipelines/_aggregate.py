import itertools
import typing as t
from collections import OrderedDict

import attr
import numpy as np
import xarray as xr

import eradiate

from ._core import PipelineStep
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
            result["spp"] = x.spp.sum(dim="spp_index")

        return result


@parse_docs
@attr.s
class AggregateCKDQuad(PipelineStep):
    """
    Compute CKD quadrature.

    In CKD modes, this pipeline step aggregates spectral data and computes
    evaluates the selected quadrature rule. The following updates to the input
    data are expected:

    * the ``index`` dimension is dropped;
    * the ``bin`` dimension is renamed ``w``;
    * the ``bin`` coordinate persists and is indexed by ``w``;
    * a ``w`` coordinate is created and contains the central wavelength of each
      bin;
    * a ``bin_wmin`` (resp. ``bin_wmax``) coordinate is created and contains the
      lower (resp. upper) spectral bound of each bin.


    Notes
    -----
    * The ``spp`` variable is averaged on the ``index`` dimension.
    * In non-CKD modes, this step is a no-op.
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
        if not eradiate.mode().is_ckd:
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
        wavelengths = []
        bin_wmins = []
        bin_wmaxs = []

        for bin in self.measure.spectral_cfg.bin_set.select_bins(
            ("ids", {"ids": bins})
        ):
            wavelengths.append(bin.wcenter.m_as(wavelength_units))
            bin_wmins.append(bin.wmin.m_as(wavelength_units))
            bin_wmaxs.append(bin.wmax.m_as(wavelength_units))

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

        # Average the 'spp' variable over the 'index' dimension
        with xr.set_options(keep_attrs=True):
            result["spp"] = x.spp.mean(dim="index")

        # Add spectral coordinates
        result = result.assign_coords(
            {
                "w": (
                    "bin",
                    wavelengths,
                    {
                        "standard_name": "wavelength",
                        "long_name": "wavelength",
                        "units": symbol(wavelength_units),
                    },
                ),
                "bin_wmin": (
                    "bin",
                    bin_wmins,
                    {
                        "standard_name": "bin_wmin",
                        "long_name": "spectral bin lower bound",
                        "units": symbol(wavelength_units),
                    },
                ),
                "bin_wmax": (
                    "bin",
                    bin_wmaxs,
                    {
                        "standard_name": "bin_wmax",
                        "long_name": "spectral bin upper bound",
                        "units": symbol(wavelength_units),
                    },
                ),
            }
        )

        # Swap the 'bin' and 'w' dimensions
        result = result.swap_dims({"bin": "w"})

        # Remove the 'index' dimension
        result = result.drop_dims("index")

        return result


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

        result[self.radiosity_var] = result[self.sector_radiosity_var].sum(
            dim=("x_index", "y_index")
        )
        result[self.radiosity_var].attrs = {
            "standard_name": "toa_outgoing_flux_density_per_unit_wavelength",
            "long_name": "top-of-atmosphere outgoing spectral flux density",
            "units": symbol(uck.get("irradiance")),
        }

        return result
