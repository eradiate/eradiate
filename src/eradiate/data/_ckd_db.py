from __future__ import annotations

import enum
import json
import os
import re
import textwrap
import warnings
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import attrs
import numpy as np
import pandas as pd
import pint
import xarray as xr

from ..config._settings import SETTINGS
from ..exceptions import InterpolationError


class ErrorHandlingAction(enum.Enum):
    IGNORE = "ignore"
    RAISE = "raise"
    WARN = "warn"


@attrs.define
class ErrorHandlingPolicy:
    missing: ErrorHandlingAction
    scalar: ErrorHandlingAction
    bounds: ErrorHandlingAction

    @classmethod
    def convert(cls, value):
        if isinstance(value, Mapping):
            kwargs = {k: ErrorHandlingAction(v) for k, v in value.items()}
            return cls(**kwargs)
        else:
            return value


@attrs.define
class ErrorHandlingConfiguration:
    x: ErrorHandlingPolicy = attrs.field(converter=ErrorHandlingPolicy.convert)
    p: ErrorHandlingPolicy = attrs.field(converter=ErrorHandlingPolicy.convert)
    t: ErrorHandlingPolicy = attrs.field(converter=ErrorHandlingPolicy.convert)

    @classmethod
    def convert(cls, value):
        if isinstance(value, dict):
            return cls(**value)
        else:
            return value


def handle_error(error: InterpolationError, action: ErrorHandlingAction):
    if action is ErrorHandlingAction.IGNORE:
        return

    if action is ErrorHandlingAction.WARN:
        warnings.warn(str(error), UserWarning)
        return

    if action is ErrorHandlingAction.RAISE:
        raise error

    raise NotImplementedError


ERROR_HANDLING_CONFIG_DEFAULT = ErrorHandlingConfiguration.convert(
    SETTINGS["absorption_database_error_handling"]
)


ureg = pint.get_application_registry()


@attrs.define(repr=False, eq=False)
class CKDAbsorptionDataBase:
    _dirpath: Path
    _index: pd.DataFrame
    _metadata: dict = attrs.field(factory=dict)
    _chunks: dict[str, np.ndarray] = attrs.field(factory=dict)

    def __attrs_post_init__(self):
        # Build bin set for all coordinates

        # Parse field names and units
        regex = re.compile(r"(?P<coord>.*)\_(?P<minmax>min|max) \[(?P<units>.*)\]")
        quantities = {}
        for colname in self._index.columns:
            if colname == "filename":
                continue

            m = regex.match(colname)
            units = m.group("units")
            magnitude = self._index[colname].values
            quantities[f"{m.group('coord')}_{m.group('minmax')}"] = ureg.Quantity(
                magnitude, units
            )

        # Populate bin bounds
        self._chunks["wl"] = np.concatenate(
            (quantities["wl_min"], [quantities["wl_max"][-1]])
        )
        self._chunks["wn"] = np.concatenate(
            (quantities["wn_max"], [quantities["wn_min"][-1]])
        )

    def __repr__(self) -> str:
        with pd.option_context("display.max_columns", 4):
            result = (
                f"<{type(self).__name__}> {self._dirpath}\n"
                "Index:\n"
                f"{textwrap.indent(repr(self._index), '    ')}"
            )
        return result

    @classmethod
    def from_directory(cls, dirpath) -> CKDAbsorptionDataBase:
        dirpath = Path(dirpath).resolve()

        index = (
            pd.read_csv(os.path.join(dirpath, "index.csv"))
            .sort_values(by="wl_min [nm]")
            .reset_index(drop=True)
        )

        try:
            with open(os.path.join(dirpath, "metadata.json")) as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {}

        return cls(dirpath, index, metadata)

    @lru_cache
    def load_dataset(self, path) -> xr.Dataset:
        print(f"Loading {path}")
        return xr.load_dataset(path)

    def lookup_filenames(self, /, **kwargs) -> list[str]:
        if len(kwargs) != 1:
            raise ValueError
        mode, values = next(iter(kwargs.items()))
        values = np.atleast_1d(values)
        bins = self._chunks[mode]
        out_bound = (values < bins.min()) | (values > bins.max())
        if np.any(out_bound):
            # TODO: handle this error
            raise RuntimeError

        indexes = np.digitize(values.m_as(bins.u), bins=bins.m) - 1
        return list(self._index["filename"].iloc[indexes])

    def lookup_datasets(self, /, **kwargs) -> list[xr.Dataset]:
        filenames = self.lookup_filenames(**kwargs)
        return [self.load_dataset(self._dirpath / filename) for filename in filenames]

    def eval_sigma_a_ckd(
        self,
        w,
        g,
        thermoprops: xr.Dataset,
        error_handling_config: ErrorHandlingConfiguration | None = None,
    ):
        if error_handling_config is None:
            error_handling_config = ERROR_HANDLING_CONFIG_DEFAULT
        # Lookup dataset
        ds = self.lookup_datasets(wl=w)[0]

        # Select bin
        w_u = ds.w.units
        w_m = w.m_as(w_u)
        result = ds.sigma_a.sel(w=w_m, method="nearest")

        # Interpolate along g
        result = result.interp(g=g).drop_vars("g")

        # Interpolate on temperature
        bounds_error = error_handling_config.t.bounds is ErrorHandlingAction.RAISE
        fill_value = None if bounds_error else 0.0  # TODO: use 2-element tuple?
        result = result.interp(
            t=thermoprops.t,
            kwargs={"bounds_error": bounds_error, "fill_value": fill_value},
        )

        # Interpolate on pressure
        bounds_error = error_handling_config.p.bounds is ErrorHandlingAction.RAISE
        fill_value = None if bounds_error else 0.0  # TODO: use 2-element tuple?
        result = result.interp(
            p=thermoprops.p,
            kwargs={"bounds_error": bounds_error, "fill_value": fill_value},
        )

        # Interpolate on concentrations

        # -- List species requested species concentrations
        x_ds = [coord for coord in ds.coords if coord.startswith("x_")]
        x_ds_scalar = [coord for coord in x_ds if ds[coord].size == 1]
        x_ds_array = set(x_ds) - set(x_ds_scalar)

        # -- Select on scalar coordinates
        result = result.isel(**{x: 0 for x in x_ds_scalar})

        # -- Interpolate on array coordinates
        bounds_error = error_handling_config.x.bounds is ErrorHandlingAction.RAISE
        fill_value = None if bounds_error else 0.0  # TODO: use 2-element tuple?
        result = result.interp(
            thermoprops[x_ds_array],
            kwargs={"bounds_error": bounds_error, "fill_value": fill_value},
        )

        # Ensure a wavelength dimension and drop all the rest
        result = result.drop_vars(["p", "t", *x_ds]).expand_dims("w")

        return result
