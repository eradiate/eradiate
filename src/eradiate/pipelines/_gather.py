from __future__ import annotations

import itertools
import logging

import attrs
import numpy as np
import xarray as xr
from pinttr.util import always_iterable

import eradiate

from ._core import PipelineStep
from ..attrs import documented, parse_docs
from ..cfconventions import ATTRIBUTES
from ..exceptions import UnsupportedModeError
from ..kernel import bitmap_to_dataset
from ..spectral.ckd import BinSet
from ..units import symbol
from ..units import unit_context_config as ucc

logger = logging.getLogger(__name__)


def _spectral_dims():
    if eradiate.mode().is_mono:
        return (("w", ATTRIBUTES["radiation_wavelength"]),)
    elif eradiate.mode().is_ckd:
        return (
            ("w", ATTRIBUTES["radiation_wavelength"]),
            ("g", ATTRIBUTES["quantile"]),
        )
    else:
        raise UnsupportedModeError


@parse_docs
@attrs.define
class GatherMono(PipelineStep):
    """
    Gather raw kernel results (output as nested dictionaries) into an xarray
    dataset.

    This pipeline step takes a nested dictionary produced by the parametric loop
    of an :class:`.Experiment` and repackages it as a :class:`~xarray.Dataset`.
    The top-level spectral index is mapped to mode-dependent spectral
    coordinates. Film dimensions are left unmodified and retain their
    metadata.

    An ``img`` variable holds sensor values. An ``spp`` variable holds the
    sample count.
    """

    var: str | tuple[str, dict] = documented(
        attrs.field(default="img"),
        default='"img"',
        type="str or tuple[str, dict]",
        init_type="str or tuple[str, dict], optional",
        doc="Name of the variable containing sensor data. Optionally, a "
        "(name, metadata) pair can be passed.",
    )

    def transform(self, x: dict) -> xr.Dataset:

        # Basic preparation
        spectral_dims = []
        spectral_dim_metadata = {}

        for y in _spectral_dims():
            if isinstance(y, str):
                spectral_dims.append(y)
                spectral_dim_metadata[y] = {}
            else:
                spectral_dims.append(y[0])
                spectral_dim_metadata[y[0]] = y[1]

        sensor_datasets = []

        # Loop on spectral indexes
        for siah, result_dict in x.items():
            if eradiate.mode().is_mono:
                spectral_index = siah
            elif eradiate.mode().is_ckd:
                spectral_index = siah

            ds = bitmap_to_dataset(result_dict["bitmap"])
            spp = result_dict["spp"]

            # Set spectral coordinates
            all_coords = {
                spectral_dim: [spectral_coord]
                for spectral_dim, spectral_coord in zip(
                    spectral_dims, always_iterable(spectral_index)
                )
            }

            # Add spectral and sensor dimensions to img array
            ds["img"] = ds.img.expand_dims(dim=all_coords)

            # Package spp in a data array
            all_dims = list(all_coords.keys())
            ds["spp"] = (all_dims, np.reshape(spp, [1 for _ in all_dims]))

            sensor_datasets.append(ds)

        # Combine all the data
        with xr.set_options(keep_attrs=True):
            result = xr.combine_by_coords(sensor_datasets)

        # Drop "channel" dimension when using a monochromatic Mitsuba variant
        if eradiate.mode().check(mi_color_mode="mono"):
            result = result.squeeze("channel", drop=True)

        for spectral_dim in spectral_dims:
            result[spectral_dim].attrs = spectral_dim_metadata[spectral_dim]

        # Apply metadata to data variables
        if isinstance(self.var, str):
            var = self.var
            var_metadata = {}
        else:
            var = self.var[0]
            var_metadata = self.var[1]

        result = result.rename({"img": var})
        result[var].attrs.update(var_metadata)

        return result


@parse_docs
@attrs.define
class GatherCKD(PipelineStep):
    """
    Gather raw kernel results into an xarray dataset.
    """

    binset: BinSet = documented(
        attrs.field(validator=attrs.validators.instance_of(BinSet)),
        doc="Bin set.",
        type=":class:`.BinSet`",
    )

    var: tuple[str, dict] = documented(
        attrs.field(default="img"),
        default='"img"',
        type="tuple[str, dict]",
        init_type="tuple[str, dict], optional",
        doc="Variable name containing sensor data and metadata.",
    )

    def transform(self, x: dict) -> xr.Dataset:

        # transform 'x' into a list of 'xr.Dataset' where each dataset
        # corresponds to a spectral index

        logger.debug("gather_ckd: begin")

        datasets = []

        # x is a dictionary whose keys are spectral indexes as hashable tuples
        # and whose values are dictionaries with keys "bitmap" and "spp"

        ix = 0
        bins = self.binset.bins
        # sort bins by wavelength
        bins = sorted(bins, key=lambda b: b.wcenter)
        items = list(x.items())

        for i, _bin in enumerate(bins):  # bin (w) loop
            ng = _bin.quad.weights.size
            _datasets = []

            for _ in range(ng):  # g loop
                item = items[ix]
                siah, result_dict = item
                w, g = siah  # wavelength, quantile pair
                bitmap = result_dict["bitmap"]
                spp = result_dict["spp"]
                dataset = bitmap_to_dataset(bitmap)

                # expand dimensions of 'img' data variable to include 'w' and 'g'
                dataset["img"] = dataset.img.expand_dims(dim={"w": [w], "g": [g]})

                # Drop "channel" dimension when using a monochromatic Mitsuba variant
                if eradiate.mode().check(mi_color_mode="mono"):
                    dataset = dataset.squeeze("channel", drop=True)

                # self.var is a tuple (name, metadata)
                name, metadata = self.var
                dataset = dataset.rename({"img": name})
                dataset[name].attrs.update(metadata)

                _datasets.append(dataset)
                ix += 1

            # concatenate along 'g'
            ds = xr.concat(_datasets, dim="g")

            # compute quadrature
            # this is a weighted sum array reduction:
            # https://docs.xarray.dev/en/stable/user-guide/computation.html#weighted-array-reductions

            # normalise weights to the [0, 1] g-interval
            weights_values = 0.5 * _bin.quad.weights
            weights = xr.DataArray(weights_values, dims=["g"], coords={"g": ds.g})
            with xr.set_options(keep_attrs=True):
                weighted = ds[name].weighted(weights)
                weighted_sum = weighted.sum(dim="g")

            #  create dataset for current bin (w)
            dataset = xr.Dataset()
            dataset[name] = weighted_sum

            # add 'spp' data variable
            dims = dataset.dims
            dataset["spp"] = (dims, spp * np.ones(tuple(dims.values())))

            # add 'wbounds' data variable, and 'wbv' coordinate
            wbounds = np.stack([_bin.wmin, _bin.wmax])
            wunits = ucc.get("wavelength")
            dataset["wbounds"] = (
                ["wbv", "w"],
                wbounds.m_as(wunits).reshape((2, 1)),
                {
                    "standard_name": "radiation_wavelength_bound",
                    "long_name": "wavelength bound",
                    "units": symbol(wunits),
                },
            )
            dataset["wbv"] = (["wbv"], ["lower", "upper"])

            datasets.append(dataset)

        with xr.set_options(keep_attrs=True):
            result = xr.concat(datasets, dim="w")

            # add metadata for 'w'
            result["w"].attrs.update(ATTRIBUTES["radiation_wavelength"])

        logger.debug("gather: end")
        return result
