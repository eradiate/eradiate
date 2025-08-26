from __future__ import annotations

import enum
import errno
import glob
import json
import logging
import os
import re
import textwrap
import warnings
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any, Callable

import attrs
import numpy as np
import pandas as pd
import pint
import xarray as xr
from cachetools import LRUCache, cachedmethod
from pinttrs.util import ensure_units

import eradiate

from .. import config
from .._mode import ModeFlag, SubtypeDispatcher
from ..attrs import define, documented
from ..data import fresolver
from ..exceptions import (
    DataError,
    InterpolationError,
    UnsetModeError,
    UnsupportedModeError,
)
from ..typing import PathLike
from ..units import to_quantity
from ..units import unit_registry as ureg

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#                           Error handling components
# ------------------------------------------------------------------------------


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
    config.settings.ABSORPTION_DATABASE.ERROR_HANDLING
)


# ------------------------------------------------------------------------------
#                           Database type definitions
# ------------------------------------------------------------------------------


@define(repr=False, eq=False)
class AbsorptionDatabase:
    """
    Common parent type for absorption coefficient databases.

    This class implements most of the data indexing logic common to all
    absorption coefficient databases.
    A database is composed of a set of NetCDF files compliant with the
    absorption coefficient database format specification and placed in the
    same directory. A database instance is initialized by specifying the path
    to the directory where the files are stored.

    If it exists, a ``metadata.json`` file is loaded into the :attr:`metadata`
    attribute.

    Databases are usually not initialized using the constructor, but rather
    using the class method constructors :meth:`from_directory`,
    :meth:`from_name` and :meth:`from_dict`.

    Notes
    -----
    A file index, stored as the :attr:`_index` private attribute, associates
    to each file the spectral region it covers. The index is preferably loaded
    from a CSV file that contains all this information; if it is not found, the
    table is built upon database initialization and saved to the database
    directory. The indexing step requires to access all files and may take a
    while. The file index table is used during queries to select efficiently the
    file where data will be read. For convenience, information about bounds
    contained in the index is assembled into a spectral mesh suitable for
    query using :func:`numpy.digitize` and stored in the :attr:`_chunks`
    dictionary.

    A spectral coverage table, stored as the :attr:`_spectral_coverage` private
    attribute, merges the spectral coordinates of all files into a consistent
    index. This table is used to provide spectral coverage information to
    higher-level components that drive the simulation. Table contents are
    preferably loaded from a CSV file; if it is not found, the table is build
    upon database initialization and saved to the database directory. This
    indexing step also requires to access all files and may take a while.

    Database access and memory usage can be controlled through two parameters:

    * File queries are stored in an LRU cache. The initial size is set to a low
      value (8) and should be appropriate for most situations. If more cache
      control is needed, the :meth:`cache_clear`,  :meth:`cache_close` and
      :meth:`cache_reset` methods can be used.
    * Datasets can be open with an eager or lazy approach. This behaviour is
      controlled using the ``lazy`` constructor parameter. In eager mode, the
      entire file used for a query is loaded into memory. This can bring
      significant access overhead when using large files. If desired, datasets
      can instead be open lazily, triggering disk access only for the specific
      data that are used.
    """

    subtypes = SubtypeDispatcher("AbsorptionDatabase")

    _dir_path: Path = documented(
        attrs.field(converter=lambda x: Path(x).absolute().resolve()),
        doc="Path to database root directory.",
        type="pathlib.Path",
        init_type="path-like",
    )

    @_dir_path.validator
    def _dir_path_validator(self, attribute, value):
        if not value.is_dir():
            raise ValueError(
                f"while validating '{attribute.name}': path '{value}' is not a "
                "directory"
            )

    _index: pd.DataFrame = documented(
        attrs.field(repr=False),
        doc="File index, assumed sorted by ascending wavelengths.",
        type="DataFrame",
    )

    @_index.validator
    def _index_validator(self, attribute, value):
        if value.empty:
            raise ValueError(f"while validating '{attribute.name}': index is empty")

        wavelengths = value["wl_min [nm]"].values
        if not np.all(wavelengths[:-1] < wavelengths[1:]):
            raise ValueError(
                f"while validating '{attribute.name}': index must be sorted by "
                "ascending wavelength values"
            )

    _spectral_coverage: pd.DataFrame = documented(
        attrs.field(repr=False),
        doc="Dataframe that unrolls the spectral information contained in "
        "all data files in the database.",
        type="DataFrame",
    )

    _metadata: dict = documented(
        attrs.field(factory=dict, repr=False),
        doc="Dictionary that contains the database metadata.",
        type="dict",
    )

    _chunks: dict[str, np.ndarray] = documented(
        attrs.field(factory=dict, repr=False, init=False),
        doc="Dictionary mapping spectral lookup mode keys ('wl' or 'wn') to arrays "
        "containing the nodes of the spectral chunk mesh, which is used to perform "
        "spectral coordinate-based file lookup.",
        type="dict[str, ndarray]",
    )

    lazy: bool = documented(
        attrs.field(default=False, repr=False),
        doc="Access mode switch: if True, load data lazily; else, load data eagerly.",
        type="bool",
    )

    _cache: LRUCache = documented(
        attrs.field(factory=lambda: LRUCache(8), repr=False),
        doc="A mapping that implements an LRU caching policy.",
        type="cachetools.LRUCache",
    )

    def __attrs_post_init__(self):
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

        # Populate spectral mesh (nodes) for both wavelength and wavenumber
        # lookup modes
        self._chunks["wl"] = np.concatenate(
            (quantities["wl_min"], [quantities["wl_max"][-1]])
        )
        self._chunks["wn"] = np.concatenate(
            (quantities["wn_max"], [quantities["wn_min"][-1]])
        )

    def __repr__(self) -> str:
        with pd.option_context("display.max_columns", 4):
            result = (
                f"<{type(self).__name__}> {self._dir_path}\n"
                f"Access mode: {'lazy' if self.lazy else 'eager'}\n"
                "Index:\n"
                f"{textwrap.indent(repr(self._index), '    ')}"
            )
        return result

    @staticmethod
    def _make_index(filenames) -> pd.DataFrame:
        # Implementation is concrete class-specific
        raise NotImplementedError

    @staticmethod
    def _make_spectral_coverage(filenames) -> pd.DataFrame:
        with xr.open_dataset(filenames[0]) as ds:
            dims = set(ds.dims)
            db_type = None
            if "w" in dims:
                db_type = "mono"
                if "g" in dims:
                    db_type = "ckd"

            if db_type is None:
                raise ValueError

            wavenumber_spectral_lookup_mode = ureg(ds.w.units).check("[length]^-1")

        index = []
        headers = ["wbound_lower [nm]", "wbound_upper [nm]"]
        rows = None

        for filename in filenames:
            filename = Path(filename)
            with xr.open_dataset(filename) as ds:
                w = to_quantity(ds.w)

                if wavenumber_spectral_lookup_mode:  # Convert to wavelength
                    w = 1.0 / w
                w = w.m_as("nm")

                if db_type == "mono":
                    wbounds_lower = np.full((len(w),), np.nan)
                    wbounds_upper = np.full((len(w),), np.nan)
                else:
                    wbounds_lower = to_quantity(ds.wbounds.sel(wbv="lower"))
                    wbounds_upper = to_quantity(ds.wbounds.sel(wbv="upper"))
                    if wavenumber_spectral_lookup_mode:  # Convert to wavelength
                        wbounds_lower = 1.0 / wbounds_lower
                        wbounds_upper = 1.0 / wbounds_upper
                    wbounds_lower = wbounds_lower.m_as("nm")
                    wbounds_upper = wbounds_upper.m_as("nm")

            index.extend([(filename.name, x) for x in w])

            if rows is None:
                rows = np.stack((wbounds_lower, wbounds_upper), axis=1)
            else:
                rows = np.concatenate(
                    (
                        rows,
                        np.stack((wbounds_lower, wbounds_upper), axis=1),
                    ),
                    axis=0,
                )

        index = pd.MultiIndex.from_tuples(index, names=["filename", "wavelength [nm]"])
        # Sort index by wavelength
        result = pd.DataFrame(rows, index=index, columns=headers).sort_index(level=1)
        return result

    @classmethod
    def from_directory(
        cls, dir_path: PathLike, lazy: bool = False, fix: bool = True
    ) -> AbsorptionDatabase:
        """
        Initialize a CKD database from a directory that contains one or several
        datasets.

        Parameters
        ----------
        dir_path : path-like
            Path where the CKD database is located.

        lazy : bool
            Access mode switch: if True, load data lazily; else, load data
            eagerly.

        fix : bool
            If ``True``, attempt generating missing index files upon
            initialization. Otherwise, raise if they are missing.

        Returns
        -------
        AbsorptionDatabase

        Raises
        ------
        FileNotFoundError
            If an index file is missing and ``fix`` is ``False``.
        """
        dir_path = Path(dir_path).resolve()

        try:
            with open(os.path.join(dir_path, "metadata.json")) as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = {}

        filenames = glob.glob(os.path.join(dir_path, "*.nc"))

        def load_index(
            index_filename: PathLike,
            read_csv: Callable[[Path], pd.DataFrame],
            make_index: Callable[[list[PathLike]], pd.DataFrame],
            to_csv: Callable[[pd.DataFrame, Path], None],
        ):
            if index_filename.is_file():
                try:
                    df = read_csv(index_filename)
                except pd.errors.EmptyDataError as e:
                    raise DataError(
                        f"Error loading index file '{index_filename}'"
                    ) from e

            elif fix:
                logger.warning(
                    f"Could not find index file '{index_filename}', building it"
                )
                df = make_index(filenames)
                to_csv(df, index_filename)

            else:
                logger.critical(f"Could not find index file '{index_filename}'")
                raise FileNotFoundError(
                    errno.ENOENT, "Missing index file", index_filename
                )

            if df.empty:
                raise DataError(f"Index loaded from '{index_filename}' is empty")

            return df

        index_path = dir_path / "index.csv"
        logger.debug(f"Loading index from '{index_path}'")
        index = load_index(
            index_filename=index_path,
            read_csv=pd.read_csv,
            make_index=cls._make_index,
            to_csv=lambda df, filename: df.to_csv(filename, index=False),
        )
        index = index.sort_values(by="wl_min [nm]").reset_index(drop=True)

        spectral_coverage_path = dir_path / "spectral.csv"
        logger.debug(f"Loading spectral coverage table from '{spectral_coverage_path}'")
        spectral_coverage = load_index(
            index_filename=spectral_coverage_path,
            read_csv=lambda df: pd.read_csv(df, index_col=(0, 1)),
            make_index=cls._make_spectral_coverage,
            to_csv=lambda df, filename: df.to_csv(filename),
        )

        return cls(dir_path, index, spectral_coverage, metadata=metadata, lazy=lazy)

    @staticmethod
    def from_name(name: str, **kwargs) -> AbsorptionDatabase:
        """
        Initialize a database from a name.

        Parameters
        ----------
        name : path-like
            Name of the requested CKD database.

        kwargs
            Additional keyword arguments are forwarded to the :meth:`.from_directory`
            constructor.

        Returns
        -------
        AbsorptionDatabase
        """

        try:
            db = KNOWN_DATABASES[name]
        except KeyError as e:
            raise ValueError(f"Unknown absorption coefficient database '{name}'") from e

        # Not great, but a.t.m it's the nicest we can do
        try:
            path_metadata = fresolver.resolve(
                f"{db['path']}/metadata.json", strict=True
            )
        except FileNotFoundError as e:
            raise DataError(
                f"While opening absorption database '{name}': could not resolve the "
                "expected metadata file. That means either that the "
                "data is located at a location that cannot be discovered by the "
                "file resolver, or that the data has not been downloaded."
            ) from e
        path = path_metadata.parent
        cls = db["cls"]
        kwargs = {**db.get("kwargs", {}), **kwargs}

        # If an operational mode is selected, we check if the user is instantiating
        # a DB type that is relevant to that mode
        try:
            cls_mode = AbsorptionDatabase.subtypes.resolve()

            if cls is not cls_mode:
                warnings.warn(
                    f"Loading '{name}' database as a {cls.__name__}, but the current "
                    f"active mode requires to use a {cls_mode.__name__}. You might want "
                    "to consider loading a different database."
                )
        except UnsetModeError:
            pass

        return cls.from_directory(path, **kwargs)

    @classmethod
    def from_dict(cls, value: dict) -> AbsorptionDatabase:
        """
        Construct from a dictionary. The dictionary has a required entry ``"construct"``
        that specifies the constructor that will be used to instantiate the
        database. Additional entries are keyword arguments passed to the selected
        constructor.

        Parameters
        ----------
        value : dict
            Converted value.

        Returns
        -------
        AbsorptionDatabase
        """

        raise NotImplementedError

    @staticmethod
    def default() -> AbsorptionDatabase:
        """
        Return a default database, depending on the active mode.

        Defaults are as follows:

        * Monochromatic: ``"komodo"``
        * CKD: ``"monotropa"``

        Returns
        -------
        AbsorptionDatabase
        """
        if eradiate.mode().is_mono:
            return AbsorptionDatabase.from_name("komodo")
        elif eradiate.mode().is_ckd:
            return AbsorptionDatabase.from_name("monotropa")
        else:
            raise UnsupportedModeError(supported=["mono", "ckd"])

    @staticmethod
    def convert(value: Any) -> AbsorptionDatabase:
        """
        Attempt conversion of a value to an absorption database.

        Parameters
        ----------
        value
            The value for which conversion is attempted.

        Returns
        -------
        MonoAbsorptionDatabase or CKDAbsorptionDatabase

        Notes
        -----
        Conversion rules are as follows:

        * If ``value`` is a string, try converting using the :meth:`.from_name`
          constructor. Do not raise if this fails.
        * If ``value`` is a string or a path, try converting using the
          :meth:`.from_directory` constructor. The returned type is consistent
          with the active mode.
        * If ``value`` is a dict, try converting using the :meth:`.from_dict`
          constructor. The returned type is consistent with the active mode.
        * Otherwise, do not convert.
        """
        if isinstance(value, str):
            try:
                return AbsorptionDatabase.from_name(value)
            except ValueError:
                pass

        if isinstance(value, (str, Path, dict)):
            if eradiate.mode().is_mono:
                cls = MonoAbsorptionDatabase
            elif eradiate.mode().is_ckd:
                cls = CKDAbsorptionDatabase
            else:
                raise UnsupportedModeError(supported=["mono", "ckd"])

            if isinstance(value, (str, Path)):
                return cls.from_directory(value)

            if isinstance(value, dict):
                return cls.from_dict(value)

        # Legacy behaviour
        if isinstance(value, (tuple, list)) and isinstance(value[0], str):
            warnings.warn(
                "Initializing an atmospheric molecular absorption database "
                "from a tuple is deprecated: specifying the spectral range is "
                "no longer necessary. Passing the database name is now enough. "
                "Support for this syntax will be removed in a future version.",
                DeprecationWarning,
            )
            return AbsorptionDatabase.from_name(value[0])

        return value

    @property
    def dir_path(self) -> Path:
        """
        Returns
        -------
        Path
            Database root path.
        """
        return self._dir_path

    @property
    def metadata(self) -> dict:
        """
        Returns
        -------
        dict
            Database metadata.
        """
        return self._metadata

    @property
    def spectral_coverage(self) -> pd.DataFrame:
        """
        Returns
        -------
        DataFrame
            Spectral coverage table.
        """
        return self._spectral_coverage

    @cachedmethod(lambda self: self._cache)
    def load_dataset(self, fname: str) -> xr.Dataset:
        """
        Convenience method to load a dataset. This method is decorated with
        :func:`functools.lru_cache` with ``maxsize=1``, which limits the number
        of reload events when repeatedly querying the same file.

        The behaviour of this method is also affected by the ``lazy`` parameter:
        if ``lazy`` is ``False``, files are loaded eagerly with
        :func:`xarray.load_dataset`; if ``lazy`` is ``True``, files are loaded
        lazily with :func:`xarray.open_dataset`.

        Parameters
        ----------
        fname : str
            Name of the file that is to be loaded.

        Returns
        -------
        Dataset
        """
        path = self._dir_path / fname

        if self.lazy:
            logger.debug("Opening '%s'" % path)
            return xr.open_dataset(path)
        else:
            logger.debug("Loading '%s'" % path)
            return xr.load_dataset(path)

    def cache_clear(self) -> None:
        """
        Clear the cache.
        """
        self._cache.clear()

    def cache_close(self) -> None:
        """
        Close all cached datasets.
        """
        for value in self._cache.values():
            value.close()

    def cache_reset(self, maxsize: int) -> None:
        """
        Reset the cache with the specified maximum size.
        """
        self._cache.clear()
        self._cache = LRUCache(maxsize=maxsize)

    def lookup_filenames(self, /, **kwargs) -> list[str]:
        """
        Look up a filename in the index table from the coordinate values passed
        as keyword arguments.

        Parameters
        ----------
        wl : quantity or array-like, optional
            Wavelength (scalar or array, quantity or unitless). If passed as a
            unitless value, it is interpreted using the units of the wavelength
            chunk bounds.

        wn : quantity or array-like, optional
            Wavenumber (scalar or array, quantity or unitless). If passed as a
            unitless value, it is interpreted using the units of the wavenumber
            chunk bounds.

        Returns
        -------
        filenames : list of str
            Names of the successfully looked up files, relative to the database
            root directory.

        Raises
        ------
        ValueError
            If the requested spectral coordinate is out of bounds.

        Notes
        -----
        Depending on the specified keyword argument (``wl`` or ``wn``), the
        lookup will be performed in wavelength or wavenumber mode. Both are
        equivalent.
        """
        if len(kwargs) != 1:
            raise ValueError(
                "only one of the 'wl' and 'wn' keyword arguments is allowed"
            )
        lookup_mode, values = next(iter(kwargs.items()))
        chunks = self._chunks[lookup_mode]

        # Make sure that 'values' has the right units
        values = ensure_units(np.atleast_1d(values), chunks.units)

        # Perform bound check
        out_bound = (values < chunks.min()) | (values > chunks.max())
        if np.any(out_bound):
            # TODO: handle this error better?
            raise ValueError("out-of-bound spectral coordinate value")

        indexes = np.digitize(values.m_as(chunks.units), bins=chunks.magnitude) - 1
        return list(self._index["filename"].iloc[indexes])

    def lookup_datasets(self, /, **kwargs) -> list[xr.Dataset]:
        """
        Perform a dataset lookup based on the requested spectral coordinate.
        See :meth:`lookup_filenames` for the accepted arguments.
        """
        filenames = self.lookup_filenames(**kwargs)
        return [self.load_dataset(filename) for filename in filenames]

    def eval_sigma_a_mono(
        self,
        w: pint.Quantity,
        thermoprops: xr.Dataset,
        error_handling_config: ErrorHandlingConfiguration | None = None,
    ) -> xr.DataArray:
        """
        Compute the absorption coefficient given spectral coordinates and a
        thermophysical profile (mono variant).

        Parameters
        ----------
        w : quantity
            The wavelength for which the absorption coefficient is evaluated.

        thermoprops : Dataset
            The thermophysical profile for which the absorption coefficient is
            evaluated.

        error_handling_config : .ErrorHandlingConfiguration, optional
            The error handling policy applied if corrdinates are missing, do not
            have the appropriate dimension or are out of the dataset's bounds.
            If unset, the default policy specified by the
            ``absorption_dataset.error_handling`` setting is applied.

        Returns
        -------
        DataArray
            A data array containing the evaluated absorption coefficient as a
            function of the spectral coordinate and altitude.

        See Also
        --------
        :mod:`eradiate.config`
        """
        raise NotImplementedError

    def eval_sigma_a_ckd(
        self,
        w: pint.Quantity,
        g: float,
        thermoprops: xr.Dataset,
        error_handling_config: ErrorHandlingConfiguration | None = None,
    ) -> xr.DataArray:
        """
        Compute the absorption coefficient given spectral coordinates and a
        thermophysical profile (CKD variant).

        Parameters
        ----------
        w : quantity
            The wavelength for which the absorption coefficient is evaluated.

        g : float
            The g-point for which the absorption coefficient is evaluated.

        thermoprops : Dataset
            The thermophysical profile for which the absorption coefficient is
            evaluated.

        error_handling_config : .ErrorHandlingConfiguration, optional
            The error handling policy applied if corrdinates are missing, do not
            have the appropriate dimension or are out of the dataset's bounds.
            If unset, the default policy specified by the
            ``absorption_dataset.error_handling`` setting is applied.

        Returns
        -------
        DataArray
            A data array containing the evaluated absorption coefficient as a
            function of the spectral coordinate and altitude.

        See Also
        --------
        :mod:`eradiate.config`
        """
        raise NotImplementedError

    @staticmethod
    def _interp_thermophysical(
        ds: xr.Dataset,
        da: xr.DataArray,
        thermoprops: xr.Dataset,
        error_handling_config: ErrorHandlingConfiguration,
    ) -> tuple[xr.DataArray, list[Hashable]]:
        # Interpolate on temperature
        bounds_error = error_handling_config.t.bounds is ErrorHandlingAction.RAISE
        fill_value = None if bounds_error else 0.0  # TODO: use 2-element tuple?
        result = da.interp(
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

        # -- List requested species concentrations
        x_ds = [coord for coord in ds.coords if coord.startswith("x_")]
        x_ds_scalar = [coord for coord in x_ds if ds[coord].size == 1]
        x_ds_array = set(x_ds) - set(x_ds_scalar)

        x_thermoprops = [dv for dv in thermoprops.data_vars if dv.startswith("x_")]
        x_missing = set(x_ds_array) - set(x_thermoprops)
        x_ds_array = x_ds_array - x_missing

        # -- Select on scalar coordinates
        result = result.isel(**{x: 0 for x in x_ds_scalar + list(x_missing)})

        # -- Interpolate on array coordinates
        bounds_error = error_handling_config.x.bounds is ErrorHandlingAction.RAISE
        fill_value = None if bounds_error else 0.0  # TODO: use 2-element tuple?
        result = result.interp(
            thermoprops[x_ds_array],
            kwargs={"bounds_error": bounds_error, "fill_value": fill_value},
        )

        return result, x_ds


@AbsorptionDatabase.subtypes.register(ModeFlag.SPECTRAL_MODE_MONO)
@attrs.define(repr=False, eq=False)
class MonoAbsorptionDatabase(AbsorptionDatabase):
    """
    Absorption coefficient database (monochromatic variant).

    See Also
    --------
    AbsorptionDatabase
    """

    @staticmethod
    def _make_index(filenames) -> pd.DataFrame:
        headers = [
            "filename",
            "wn_min [cm^-1]",
            "wn_max [cm^-1]",
            "wl_min [nm]",
            "wl_max [nm]",
        ]
        rows = []

        for filename in filenames:
            filename = Path(filename)
            with xr.open_dataset(filename) as ds:
                w_u = ureg(ds.w.units)

                if w_u.check("[length]^-1"):  # wavenumber mode
                    wn_min = float(ds.w.min()) * w_u
                    wn_max = float(ds.w.max()) * w_u
                    wl_min = 1.0 / wn_max
                    wl_max = 1.0 / wn_min
                elif w_u.check("[length]"):  # wavelength mode
                    wl_min = float(ds.w.min()) * w_u
                    wl_max = float(ds.w.max()) * w_u
                    wn_min = 1.0 / wl_max
                    wn_max = 1.0 / wl_min
                else:
                    raise ValueError(f"Cannot interpret units '{w_u}'")

                rows.append(
                    [
                        filename.name,
                        wn_min.m_as("1/cm"),
                        wn_max.m_as("1/cm"),
                        wl_min.m_as("nm"),
                        wl_max.m_as("nm"),
                    ]
                )

        return pd.DataFrame(rows, columns=headers).sort_values("wl_min [nm]")

    @classmethod
    def from_dict(cls, value: dict) -> MonoAbsorptionDatabase:
        # Inherit docstring
        value = value.copy()
        constructor = getattr(cls, value.pop("construct"))
        return constructor(**value)

    def eval_sigma_a_mono(
        self,
        w: pint.Quantity,
        thermoprops: xr.Dataset,
        error_handling_config: ErrorHandlingConfiguration | None = None,
    ) -> xr.DataArray:
        # Inherit docstring

        if error_handling_config is None:
            error_handling_config = ERROR_HANDLING_CONFIG_DEFAULT

        # Lookup dataset
        ds = self.lookup_datasets(wl=w)[0]

        # Interpolate on spectral dimension
        # TODO: Optimize
        w_u = ureg(ds.w.units)
        # Note: Support for wavenumber spectral lookup mode is suboptimal
        w_m = (1.0 / w).m_as(w_u) if w_u.check("[length]^-1") else w.m_as(w_u)
        result = ds.sigma_a.interp(w=w_m, method="linear")

        # Interpolate on thermophysical dimensions
        result, x_ds = self._interp_thermophysical(
            ds, result, thermoprops, error_handling_config
        )

        # Drop thermophysical coordinates, ensure spectral dimension
        result = result.drop_vars(["p", "t", *x_ds])
        if "w" not in result.dims:
            result = result.expand_dims("w")

        return result.transpose("w", "z")


@AbsorptionDatabase.subtypes.register(ModeFlag.SPECTRAL_MODE_CKD)
@attrs.define(repr=False, eq=False)
class CKDAbsorptionDatabase(AbsorptionDatabase):
    """
    Absorption coefficient database (CKD variant).

    See Also
    --------
    AbsorptionDatabase
    """

    @staticmethod
    def _make_index(filenames) -> pd.DataFrame:
        headers = [
            "filename",
            "wn_min [cm^-1]",
            "wn_max [cm^-1]",
            "wl_min [nm]",
            "wl_max [nm]",
        ]
        rows = []

        for filename in filenames:
            filename = Path(filename)
            with xr.open_dataset(filename) as ds:
                w_u = ureg(ds.w.units)

                if w_u.check("[length]^-1"):  # wavenumber mode
                    wn_min = float(ds.wbounds.sel(wbv="lower").min()) * w_u
                    wn_max = float(ds.wbounds.sel(wbv="upper").max()) * w_u
                    wl_min = 1.0 / wn_max
                    wl_max = 1.0 / wn_min
                elif w_u.check("[length]"):  # wavelength mode
                    wl_min = float(ds.wbounds.sel(wbv="lower").min()) * w_u
                    wl_max = float(ds.wbounds.sel(wbv="upper").max()) * w_u
                    wn_min = 1.0 / wl_max
                    wn_max = 1.0 / wl_min
                else:
                    raise ValueError(f"Cannot interpret units '{w_u}'")

                rows.append(
                    [
                        filename.name,
                        wn_min.m_as("1/cm"),
                        wn_max.m_as("1/cm"),
                        wl_min.m_as("nm"),
                        wl_max.m_as("nm"),
                    ]
                )

        return pd.DataFrame(rows, columns=headers).sort_values("wl_min [nm]")

    @classmethod
    def from_dict(cls, value: dict) -> CKDAbsorptionDatabase:
        # Inherit docstring
        value = value.copy()
        constructor = getattr(cls, value.pop("construct"))
        return constructor(**value)

    def eval_sigma_a_ckd(
        self,
        w: pint.Quantity,
        g: float,
        thermoprops: xr.Dataset,
        error_handling_config: ErrorHandlingConfiguration | None = None,
    ) -> xr.DataArray:
        # Inherit docstring

        # TODO: Implement new bounds error handling policy. This policy is as
        #  follows:
        #  * Interpolation is done for an altitude range such that the pressure
        #    is higher than the lower bound of the pressure variable in the
        #    CKD table. This is implemented at a higher level (not here).
        #  * The default bound error handling policy for the pressure and
        #    temperature variables is 'extrapolate'.
        #  * Above the cut-off altitude, the profile is filled with zeros.
        #  Cut-off detection is implemented with pressure-based masking.

        # TODO: Use the 'assume_sorted' parameter of DataArray.interp()

        if error_handling_config is None:
            error_handling_config = ERROR_HANDLING_CONFIG_DEFAULT

        # Lookup dataset
        ds = self.lookup_datasets(wl=w)[0]

        # Select bin
        # TODO: Optimize
        w_u = ureg(ds.w.units)
        w_m = w.m_as(w_u)
        result = ds.sigma_a.sel(w=w_m, method="nearest")

        # Interpolate along g
        result = result.interp(g=g).drop_vars("g")

        # Interpolate on thermophysical dimensions
        result, x_ds = self._interp_thermophysical(
            ds, result, thermoprops, error_handling_config
        )

        # Drop thermophysical coordinates, ensure spectral dimension
        result = result.drop_vars(["p", "t", *x_ds])
        if "w" not in result.dims:
            result = result.expand_dims("w")

        return result.transpose("w", "z")


# ------------------------------------------------------------------------------
#                               Static definitions
# ------------------------------------------------------------------------------

KNOWN_DATABASES = {
    "gecko": {
        "cls": MonoAbsorptionDatabase,
        "path": "absorption_mono/gecko",
        "kwargs": {"lazy": True},
    },
    "komodo": {
        "cls": MonoAbsorptionDatabase,
        "path": "absorption_mono/komodo",
        "kwargs": {"lazy": True},
    },
    "monotropa": {
        "cls": CKDAbsorptionDatabase,
        "path": "absorption_ckd/monotropa",
    },
    "mycena": {
        "cls": CKDAbsorptionDatabase,
        "path": "absorption_ckd/mycena",
    },
    "panellus": {
        "cls": CKDAbsorptionDatabase,
        "path": "absorption_ckd/panellus",
    },
    "tuber": {
        "cls": CKDAbsorptionDatabase,
        "path": "absorption_ckd/tuber",
    },
}
