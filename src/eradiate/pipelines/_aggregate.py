import itertools
import logging
import typing as t
from collections import OrderedDict

import attrs
import numpy as np
import xarray as xr

from ._core import PipelineStep
from ..attrs import documented, parse_docs
from ..scenes.measure import Measure
from ..spectral.ckd import BinSet
from ..units import symbol
from ..units import unit_context_config as ucc
from ..units import unit_context_kernel as uck

logger = logging.getLogger(__name__)


@parse_docs
@attrs.define
class AggregateCKDQuad(PipelineStep):
    """
    Compute CKD quadrature.

    In CKD modes, this pipeline step aggregates spectral data and
    evaluates the selected quadrature rule. The following updates to the input
    data are expected:

    * the ``g`` dimension is dropped;
    * the ``bin`` dimension is renamed ``w``;
    * the ``bin`` coordinate persists and is indexed by ``w``;
    * a ``w`` coordinate is created and contains the central wavelength of each
      bin;
    * a ``bin_wmin`` (resp. ``bin_wmax``) coordinate is created and contains the
      lower (resp. upper) spectral bound of each bin;
    * the dataset is reordered by ascending ``w`` values.

    Notes
    -----
    * The ``spp`` variable is averaged on the ``g`` dimension.
    * In non-CKD modes, this step is a no-op.
    """

    measure: Measure = documented(
        attrs.field(
            validator=attrs.validators.instance_of(Measure),
            repr=lambda self: f"{self.__class__.__name__}(id='{self.id}', ...)",
        ),
        doc="Measure from which the processed data was obtained.",
        type=":class:`.Measure`",
    )

    binset: BinSet = documented(
        attrs.field(validator=attrs.validators.instance_of(BinSet)),
        doc="Bin set.",
        type=":class:`.BinSet`",
    )

    var: str = documented(
        attrs.field(default="img", validator=attrs.validators.instance_of(str)),
        doc="Name of the variable for which CKD quadrature computation "
        "is to be performed.",
        type="str",
        default='"img"',
    )

    def transform(self, x: t.Any) -> t.Any:
        logger.debug("aggregate_ckd_quad: begin")
        # Compute quadrature spectrum-indexed variables and turn spp
        # into a per-bin average
        var = self.var

        # Get dimensions of current variable
        img = x.data_vars[var]

        dims = OrderedDict((y, len(img.coords[y])) for y in img.dims)

        if "w" not in dims:
            raise ValueError(f"variable '{var}' is missing dimension 'w'")

        if "g" not in dims:
            raise ValueError(f"variable '{var}' is missing dimension 'g'")

        # Init storage
        del dims["w"]
        del dims["g"]

        # Collect wavelengths associated with each bin
        wunit = ucc.get("wavelength")
        wavelengths = x.w.values
        bin_wmins = []
        bin_wmaxs = []
        for bin in self.binset.bins:
            wmin, wmax = bin.wmin, bin.wmax
            bin_wmins.append(wmin.m_as(wunit))
            bin_wmaxs.append(wmax.m_as(wunit))

        result = x

        aggregated = xr.DataArray(
            np.zeros([wavelengths.size] + list(dims.values())),
            coords={"w": img.w, **{dim: img.coords[dim] for dim in dims}},
        )

        # For each bin and each pixel, compute quadrature and store the result
        for i in range(wavelengths.size):

            bin_i = self.binset.bins[i]

            values_at_nodes = img.isel(w=i).values

            # Rationale: Avoid using xarray's indexing in this loop for
            # performance reasons (wrong data indexing method will result in
            # 10x+ speed reduction)
            indexes_product = itertools.product(
                *[list(range(n)) for n in dims.values()]
            )
            for indexes in indexes_product:
                aggregated.values[(i, *indexes)] = bin_i.quad.integrate(
                    values_at_nodes[(slice(None), *indexes)],
                    interval=np.array([0.0, 1.0]),
                )

        result = result.assign({var: aggregated})
        result[var].attrs = x[var].attrs

        # Average the 'spp' variable over the 'index' dimension
        with xr.set_options(keep_attrs=True):
            result["spp"] = x.spp.mean(dim="g")

        # Add spectral coordinates
        result = result.assign_coords(
            {
                "w": (
                    "w",
                    wavelengths,
                    {
                        "standard_name": "radiation_wavelength",
                        "long_name": "wavelength",
                        "units": symbol(wunit),
                    },
                ),
            }
        )

        # Remove the 'g' dimension
        result = result.drop_dims("g")

        # Reorder by ascending "w"
        result = result.sortby("w")

        logger.debug("aggregate_ckd_quad: end")

        return result


@parse_docs
@attrs.define
class AggregateRadiosity(PipelineStep):
    """
    Aggregate flux density field.
    """

    sector_radiosity_var: str = documented(
        attrs.field(
            default="sector_radiosity", validator=attrs.validators.instance_of(str)
        ),
        doc="Name of the variable containing radiosity values for the "
        "hemisphere sector corresponding to each film pixel. This quantity is "
        "expressed in flux units (typically W/m²) and, when summed over the "
        "entire film, aggregates into a radiosity.",
        type="str",
        default='"sector_radiosity"',
    )

    radiosity_var: str = documented(
        attrs.field(default="radiosity", validator=attrs.validators.instance_of(str)),
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
