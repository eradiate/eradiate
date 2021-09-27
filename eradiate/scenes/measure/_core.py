from __future__ import annotations

import itertools
import typing as t
from abc import ABC, abstractmethod
from collections import OrderedDict, abc

import attr
import numpy as np
import pandas as pd
import pint
import pinttr
import xarray as xr

import eradiate

from ..core import KernelDict, SceneElement
from ... import ckd, converters, validators
from ..._factory import Factory
from ..._mode import ModeFlags
from ..._util import deduplicate, natsort_alphanum_key
from ...attrs import AUTO, AutoType, documented, get_doc, parse_docs
from ...ckd import Bin
from ...contexts import (
    CKDSpectralContext,
    KernelDictContext,
    MonoSpectralContext,
    SpectralContext,
)
from ...exceptions import ModeError, UnsupportedModeError
from ...units import interpret_quantities, symbol
from ...units import unit_context_config as ucc
from ...units import unit_registry as ureg

measure_factory = Factory()


@parse_docs
@attr.s(frozen=True)
class SensorInfo:
    """
    Data type to store information about a sensor associated with a measure.
    Instances are immutable.
    """

    id: str = documented(
        attr.ib(),
        doc="Sensor unique identifier.",
        type="str",
    )

    spp: int = documented(
        attr.ib(),
        doc="Sensor sample count.",
        type="int",
    )


# ------------------------------------------------------------------------------
#                   Spectral configuration data structures
# ------------------------------------------------------------------------------


class MeasureSpectralConfig(ABC):
    """
    Data structure specifying the spectral configuration of a :class:`.Measure`.

    While this class is abstract, it should however be the main entry point
    to create :class:`.MeasureSpectralConfig` child class objects through the
    :meth:`.MeasureSpectralConfig.new` class method constructor.
    """

    @abstractmethod
    def spectral_ctxs(self) -> t.List[SpectralContext]:
        """
        Return a list of :class:`.SpectralContext` objects based on the
        stored spectral configuration. These data structures can be used to
        drive the evaluation of spectrally dependent components during a
        spectral loop.

        Returns
        -------
        list of :class:`.SpectralContext`
            List of generated spectral contexts. The concrete class
            (:class:`.MonoSpectralContext`, :class:`.CKDSpectralContext`, etc.)
            depends on the active mode.
        """
        pass

    @staticmethod
    def new(**kwargs) -> MeasureSpectralConfig:
        """
        Create a new instance of one of the :class:`.SpectralContext` child
        classes. *The instantiated class is defined based on the currently active
        mode.* Keyword arguments are passed to the instantiated class's
        constructor.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments depending on the currently active mode (see below
            for a list of actual keyword arguments).

        wavelengths : quantity, default: [550] nm
            **Monochromatic modes** [:class:`.MonoMeasureSpectralConfig`].
            List of wavelengths (automatically converted to a Numpy array).
            *Unit-enabled field (default: ucc[wavelength]).*

        bin_set : :class:`.BinSet` or str, default: "10nm"
            **CKD modes** [:class:`.CKDMeasureSpectralConfig`].
            CKD bin set definition. If a string is passed, the data
            repository is queried for the corresponding identifier using
            :meth:`.BinSet.from_db`.

        bins : list of (str or tuple or dict or callable)
            **CKD modes** [:class:`.CKDSpectralContext`].
            List of CKD bins on which to perform the spectral loop. If unset,
            all the bins defined by the selected bin set will be covered.

        See Also
        --------
        :func:`eradiate.mode`, :func:`eradiate.set_mode`
        """
        mode = eradiate.mode()

        if mode is None:
            raise ModeError(
                "instantiating MeasureSpectralConfig requires a mode to be selected"
            )

        if mode.has_flags(ModeFlags.ANY_MONO):
            return MonoMeasureSpectralConfig(**kwargs)

        elif mode.has_flags(ModeFlags.ANY_CKD):
            return CKDMeasureSpectralConfig(**kwargs)

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    @staticmethod
    def from_dict(d: t.Dict) -> MeasureSpectralConfig:
        """
        Create from a dictionary. This class method will additionally pre-process
        the passed dictionary to merge any field with an associated ``"_units"``
        field into a :class:`pint.Quantity` container.

        Parameters
        ----------
        d : dict
            Configuration dictionary used for initialisation.

        Returns
        -------
        :class:`.MeasureSpectralConfig`
            Created object.
        """

        # Pre-process dict: apply units to unit-enabled fields
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Perform object creation
        return MeasureSpectralConfig.new(**d_copy)

    @staticmethod
    def convert(value: t.Any) -> t.Any:
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`from_dict` to
        create a :class:`.SpectralContext`.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return MeasureSpectralConfig.from_dict(value)

        return value


@parse_docs
@attr.s
class MonoMeasureSpectralConfig(MeasureSpectralConfig):
    """
    A data structure specifying the spectral configuration of a :class:`.Measure`
    in monochromatic modes.
    """

    _wavelengths: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity([550.0], ureg.nm),
            units=ucc.deferred("wavelength"),
            converter=lambda x: converters.on_quantity(np.atleast_1d)(
                pinttr.converters.ensure_units(x, ucc.get("wavelength"))
            ),
        ),
        doc="List of wavelengths on which to perform the monochromatic spectral "
        "loop.\n\nUnit-enabled field (default: ucc['wavelength']).",
        type="quantity",
        init_type="quantity or array-like",
        default="[550.0] nm",
    )

    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value

    def spectral_ctxs(self) -> t.List[MonoSpectralContext]:
        return [
            MonoSpectralContext(wavelength=wavelength)
            for wavelength in self._wavelengths
        ]


def _ckd_measure_spectral_config_bins_converter(value):
    # Converter for CKDMeasureSpectralConfig.bins
    #

    if isinstance(value, str):
        value = [value]

    bin_set_specs = []
    for bin_set_spec in value:
        # Strings and callables are passed without change
        if isinstance(bin_set_spec, str) or callable(bin_set_spec):
            bin_set_specs.append(bin_set_spec)
            continue

        # In sequences and dicts, we interpret units
        if isinstance(bin_set_spec, abc.Sequence):
            if bin_set_spec[0] == "interval":
                bin_set_specs.append(
                    (
                        "interval",
                        interpret_quantities(
                            bin_set_spec[1],
                            {"wmin": "wavelength", "wmax": "wavelength"},
                            ucc,
                        ),
                    )
                )
            else:
                bin_set_specs.append(bin_set_spec)

            continue

        if isinstance(bin_set_spec, abc.Mapping):
            if bin_set_spec["type"] == "interval":
                bin_set_specs.append(
                    {
                        "type": "interval",
                        "filter_kwargs": interpret_quantities(
                            bin_set_spec["filter_kwargs"],
                            {"wmin": "wavelength", "wmax": "wavelength"},
                            ucc,
                        ),
                    }
                )

            else:
                bin_set_specs.append(bin_set_spec)

            continue

        raise ValueError(f"unhandled CKD bin specification {bin_set_spec}")

    return bin_set_specs


@parse_docs
@attr.s(frozen=True)
class CKDMeasureSpectralConfig(MeasureSpectralConfig):
    """
    A data structure specifying the spectral configuration of a
    :class:`.Measure` in CKD modes.
    """

    # TODO: replace manual bin selection with automation based on sensor spectral
    #  response (with a system to easily design arbitrary SSRs for cases where an
    #  instrument is not simulated)

    bin_set: ckd.BinSet = documented(
        attr.ib(
            default="10nm",
            converter=ckd.BinSet.convert,
            validator=attr.validators.instance_of(ckd.BinSet),
        ),
        doc="CKD bin set definition. If a string is passed, the data "
        "repository is queried for the corresponding identifier using "
        ":meth:`.BinSet.from_db`.",
        type=":class:`~.ckd.BinSet`",
        init_type=":class:`~.ckd.BinSet` or str",
        default='"10nm"',
    )

    _bins: t.Union[t.List[str], AutoType] = documented(
        attr.ib(
            default=AUTO,
            converter=converters.auto_or(_ckd_measure_spectral_config_bins_converter),
        ),
        doc="List of CKD bins on which to perform the spectral loop. If set to "
        "``AUTO``, all the bins relevant to the selected spectral response will be "
        "covered.",
        type="list of str or AUTO",
        init_type="list of (str or tuple or dict or callable) or AUTO",
        default="AUTO",
    )

    @_bins.validator
    def _bins_validator(self, attribute, value):
        if value is AUTO:
            return

        for bin_spec in value:
            if not (
                isinstance(bin_spec, (str, list, tuple, dict)) or callable(bin_spec)
            ):
                raise ValueError(
                    f"while validating {attribute.name}: unsupported CKD bin "
                    f"specification {bin_spec}; expected str, sequence, mapping "
                    f"or callable"
                )

    @property
    def bins(self) -> t.Tuple[Bin]:
        """
        Returns
        -------
        tuple of :class:`.Bin`
            List of selected bins.
        """
        if self._bins is not AUTO:
            bin_selectors = self._bins
        else:
            bin_selectors = [lambda x: True]

        return self.bin_set.select_bins(*bin_selectors)

    def spectral_ctxs(self) -> t.List[CKDSpectralContext]:
        ctxs = []

        for bin in self.bins:
            for bindex in bin.bindexes:
                ctxs.append(CKDSpectralContext(bindex))

        return ctxs


# ------------------------------------------------------------------------------
#                     Raw result storage and basic processing
# ------------------------------------------------------------------------------


def _str_summary_raw(x):
    if not x:
        return "{}"

    keys = list(x.keys())
    return f"dict<{len(keys)} items>({{{keys[0]}: {{...}} , ... }})"


@parse_docs
@attr.s
class MeasureResults:
    """
    Data structure storing simulation results corresponding to a measure before
    post-processing. A ``raw`` field stores raw sensor results as nested
    dictionaries (see corresponding field documentation for further detail).
    The :meth:`~.MeasureResults.to_dataset` repacks the data as a
    :class:`~xarray.Dataset`, which is the data structure operated on by the
    :class:`.Measure.postprocess` pipeline.
    """

    raw: t.Dict = documented(
        attr.ib(
            factory=dict,
            validator=attr.validators.optional(attr.validators.instance_of(dict)),
            repr=_str_summary_raw,
        ),
        doc="Raw results stored as nested dictionaries with the following structure:\n"
        "\n"
        ".. code:: python\n\n"
        "   {\n"
        "       spectral_key_0: {\n"
        '           "values": {\n'
        '               "sensor_0": data_0,\n'
        '               "sensor_1": data_1,\n'
        "               ...\n"
        "           },\n"
        '           "spp": {\n'
        '               "sensor_0": sample_count_0,\n'
        '               "sensor_1": sample_count_1,\n'
        "               ...\n"
        "           },\n"
        "       },\n"
        "       spectral_key_1: {\n"
        '           "values": {\n'
        '               "sensor_0": data_0,\n'
        '               "sensor_1": data_1,\n'
        "               ...\n"
        "           },\n"
        '           "spp": {\n'
        '               "sensor_0": sample_count_0,\n'
        '               "sensor_1": sample_count_1,\n'
        "               ...\n"
        "           },\n"
        "       },\n"
        "       ...\n"
        "   }\n",
        type="dict",
        default="{}",
    )

    def to_dataset(self, aggregate_spps: bool = False) -> xr.Dataset:
        """
        Repack raw results as a :class:`xarray.Dataset`. Dimension coordinates are
        as follows:

        * spectral coordinate (varies vs active mode)
        * ``sensor_id``: kernel sensor identified (for SPP split, dropped if
          ``aggregate_spps`` is ``True``)
        * ``y``: film height
        * ``x``: film width

        .. important:: The spectral coordinate is sorted.

        Parameters
        ----------
        aggregate_spps : bool
            If ``True``, perform split SPP aggregation (*i.e.* sum results using
            SPP values as weights). This will result in the ``sensor_id``
            dimension being dropped.

        Returns
        -------
        Dataset
            Raw sensor data repacked as a :class:`~xarray.Dataset`.
        """

        if not self.raw:
            raise ValueError("no raw results to convert to xarray.Dataset")

        # Collect spectral coordinate label
        spectral_coord_label = eradiate.mode().spectral_coord_label

        # Collect spectral and sensor coordinate values
        spectral_coords, sensor_ids, film_size = self._to_dataset_helper_coord_values(
            self.raw
        )
        # Collect radiance values
        data = self._to_dataset_helper_data_values(
            self.raw, spectral_coords, sensor_ids, film_size
        )

        # Collect sample counts
        spps = self._to_dataset_helper_spp_values(self.raw, spectral_coords, sensor_ids)

        # Compute pixel film coordinates
        xs, ys = self._to_dataset_helper_pixel_coord_values(film_size)

        # Construct index if relevant
        spectral_index = self._to_dataset_helper_spectral_index(spectral_coords)

        # Construct dataset
        result = xr.Dataset(
            data_vars=(
                {
                    "raw": (
                        [spectral_coord_label, "sensor_id", "y", "x"],
                        data,
                        {"long_name": "raw sensor values"},
                    ),
                    "spp": (
                        [spectral_coord_label, "sensor_id"],
                        spps,
                        {"long_name": "sample count"},
                    ),
                }
            ),
            coords={
                "x": ("x", xs, {"long_name": "film width coordinate"}),
                "y": ("y", ys, {"long_name": "film height coordinate"}),
                spectral_coord_label: (
                    spectral_coord_label,
                    spectral_index,
                    self._to_dataset_spectral_coord_metadata(),
                ),
                "sensor_id": (
                    "sensor_id",
                    sensor_ids,
                    {"long_name": "sensor ID"},
                ),
            },
        )

        if not aggregate_spps:
            return result

        sensor_id_prefixes = deduplicate(
            [
                sensor_id.split("_spp")[0]
                for sensor_id in list(result["sensor_id"].values)
            ]
        )

        raw_aggregated = []
        spp_aggregated = []
        for sensor_id_prefix in sensor_id_prefixes:
            # Select SPP-varying components for the current sensor ID prefix
            sensor_ids = [
                id
                for id in result.coords["sensor_id"].values
                if id.split("_spp")[0] == sensor_id_prefix
            ]
            # Aggregate sample counts for the current prefix
            weights = xr.where(
                result.coords["sensor_id"].isin(sensor_ids),
                result.data_vars["spp"],
                0,
            )
            raw_aggregated.append(
                result["raw"]
                .weighted(weights)
                .mean(dim="sensor_id")
                .expand_dims(
                    {"sensor_id": [sensor_id_prefix]},
                    axis=1,
                )
                .to_dataset()
            )
            # Mask spp values so as to only include selected values
            # The sensor_id coordinate is redefined to strip the spp suffix
            weights_spp = xr.where(result.coords["sensor_id"].isin(sensor_ids), 1, 0)
            spp_aggregated.append(
                result.data_vars["spp"]
                .weighted(weights_spp)
                .sum(dim="sensor_id")
                .expand_dims(
                    {"sensor_id": [sensor_id_prefix]},
                    axis=1,
                )
                .to_dataset()
            )

        raw = xr.concat(raw_aggregated, "sensor_id")["raw"]

        spp = xr.concat(spp_aggregated, "sensor_id")["spp"]

        result_aggregated = xr.Dataset(
            data_vars={
                "raw": raw,
                "spp": spp,
                "sensor_id": sensor_id_prefixes,
            }
        )

        # Copy metadata
        for var in list(result_aggregated.data_vars) + list(result_aggregated.coords):
            result_aggregated[var].attrs = result[var].attrs.copy()

        return result_aggregated

    @staticmethod
    def _to_dataset_spectral_coord_metadata() -> t.Dict:
        """
        Return metadata for the spectral coordinate based on active mode.

        Returns
        -------
        dict
            Metadata dictionary, ready to attach to the appropriate xarray
            coordinate object.
        """
        wavelength_units = ucc.get("wavelength")

        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return {
                "standard_name": "wavelength",
                "long_name": "wavelength",
                "units": symbol(wavelength_units),
            }

        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return {"long_name": "bindex"}

        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

    @staticmethod
    def _to_dataset_helper_coord_values(
        raw: t.Dict,
    ) -> t.Tuple[t.List, t.List, t.Tuple[int, int]]:
        """
        Collect spectral and sensor coordinate values from raw result dictionary.

        Parameters
        ----------
        raw : dict
            Raw result dictionary.

        Returns
        -------
        spectral_coords : list
            Spectral coordinate values, sorted in ascending order (in CKD modes,
            numeric string bin IDs are sorted in natural order, meaning that
            "1000" will indeed be after "900").

        sensor_ids : list
            Sensor coordinate values, sorted in ascending order.

        film_size : tuple
            Sensor film size as a (int, int) pair.
        """
        spectral_coords = set()
        sensor_ids = list()
        film_size = np.zeros((2,), dtype=int)

        for spectral_coord, val in raw.items():
            spectral_coords.add(spectral_coord)
            for sensor_id, data in val["values"].items():
                if sensor_id not in sensor_ids:
                    sensor_ids.append(sensor_id)
                film_size = np.maximum(film_size, data.shape[:2])

        if eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            spectral_coords_sort_key = lambda x: (
                *natsort_alphanum_key(x[0]),
                int(x[1]),
            )
        else:
            spectral_coords_sort_key = lambda x: x

        return (
            sorted(spectral_coords, key=spectral_coords_sort_key),
            sorted(sensor_ids),
            tuple(film_size),
        )

    @staticmethod
    def _to_dataset_helper_data_values(
        raw: t.Dict,
        spectral_coords: t.List,
        sensor_ids: t.List,
        film_size: t.Tuple[int, int],
    ) -> np.ndarray:
        """
        Collect spectral and sensor coordinate values from raw result dictionary.

        Parameters
        ----------
        raw : dict
            Raw result dictionary.

        spectral_coords : list
            Spectral coordinate values.

        sensor_ids : list
            Sensor coordinate values.

        film_size : tuple[int, int]
            Sensor film size.

        Returns
        -------
        ndarray
            Sensor data values as a Numpy array. Dimensions are ordered as follows:

            * spectral;
            * sensor;
            * pixel index.
        """
        if not eradiate.mode().has_flags(ModeFlags.MTS_MONO | ModeFlags.ANY_CKD):
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))

        data = np.full(
            (
                len(spectral_coords),
                len(sensor_ids),
                *film_size,  # Note: Row-major order (width x comes last)
            ),
            np.nan,
        )

        for i_spectral, spectral_coord in enumerate(spectral_coords):
            for i_sensor, sensor_id in enumerate(sensor_ids):
                # Note: This doesn't handle heterogeneous sensor film sizes
                # (i.e. cases in which sensors have different film sizes).
                # To add support for it, blitting is probably a good approach
                # https://stackoverflow.com/questions/28676187/numpy-blit-copy-part-of-an-array-to-another-one-with-a-different-size
                data[i_spectral, i_sensor] = raw[spectral_coord]["values"][sensor_id][
                    ..., 0
                ]
                # This latter indexing selects only one channel in the raw data
                # array: this works with mono variants but will fail otherwise

        return data

    @staticmethod
    def _to_dataset_helper_spp_values(
        raw: t.Dict, spectral_coords: t.List, sensor_ids: t.List
    ) -> np.ndarray:
        """
        Collect sample count values for each (spectral_index, sensor_index) pair.

        Parameters
        ----------
        raw : dict
            Raw result dictionary.

        spectral_coords : list
            Spectral coordinate values.

        sensor_ids : list
            Sensor coordinate values.

        Returns
        -------
        array
            Sample count for each spectral channel and each sensor.
            Dimensions are ordered as follows:

            * spectral;
            * sensor.
        """
        spps = np.full((len(spectral_coords), len(sensor_ids)), np.nan, dtype=int)

        for i_spectral, spectral_coord in enumerate(spectral_coords):
            for i_sensor, sensor_id in enumerate(sensor_ids):
                spps[i_spectral, i_sensor] = raw[spectral_coord]["spp"][sensor_id]

        return spps

    @staticmethod
    def _to_dataset_helper_pixel_coord_values(
        film_size: t.Tuple[int, int]
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Compute pixel coordinates from a film size.

        Parameters
        ----------
        film_size : tuple of int
            Film size as a (int, int) pair.

        Returns
        -------
        x : array
            x pixel coordinates in the [0, 1] × [0, 1] space.

        y : array
            y pixel coordinates in the [0, 1] × [0, 1] space.
        """

        # Compute pixel film coordinates
        # As mentioned before, raw data shape is (y, x)
        xs = np.arange(0.5, film_size[1], 1.0) / film_size[1]
        ys = np.arange(0.5, film_size[0], 1.0) / film_size[0]

        return xs, ys

    @staticmethod
    def _to_dataset_helper_spectral_index(spectral_coords):
        """
        Create spectral index based on current mode.

        Parameters
        ----------
        spectral_coords : array-like
            List of spectral coordinate values.

        Returns
        -------
        pd.Index
            Generated index (possibly a multi-index).
        """
        if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
            return pd.Index(spectral_coords)
        elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            return pd.MultiIndex.from_tuples(spectral_coords, names=("bin", "index"))
        else:
            raise UnsupportedModeError(supported=("monochromatic", "ckd"))


# ------------------------------------------------------------------------------
#                             Measure base class
# ------------------------------------------------------------------------------


@parse_docs
@attr.s
class Measure(SceneElement, ABC):
    """
    Abstract base class for all measure scene elements.
    """

    id: t.Optional[str] = documented(
        attr.ib(
            default="measure",
            validator=attr.validators.optional((attr.validators.instance_of(str))),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default='"measure"',
    )

    spectral_cfg: MeasureSpectralConfig = documented(
        attr.ib(
            factory=MeasureSpectralConfig.new,
            converter=MeasureSpectralConfig.convert,
        ),
        doc="Spectral configuration of the measure. Must match the current "
        "operational mode. Can be passed as a dictionary, which will be "
        "interpreted by :meth:`.MeasureSpectralConfig.from_dict`.",
        type=":class:`.MeasureSpectralConfig`",
        init_type=":class:`.MeasureSpectralConfig` or dict",
        default=":meth:`MeasureSpectralConfig.new() <.MeasureSpectralConfig.new>`",
    )

    spp: int = documented(
        attr.ib(default=32, converter=int, validator=validators.is_positive),
        doc="Number of samples per pixel.",
        type="int",
        default="32",
    )

    results: MeasureResults = documented(
        attr.ib(factory=MeasureResults),
        doc="Storage for raw results yielded by the kernel.",
        type=":class:`.MeasureResults`",
        default=":class:`MeasureResults() <.MeasureResults>`",
    )

    # Private attributes
    # Sample count which, if exceeded, should trigger sample count splitting in
    # single-precision modes
    _spp_splitting_threshold: int = attr.ib(
        default=int(1e5), converter=int, validator=validators.is_positive, repr=False
    )

    @property
    @abstractmethod
    def film_resolution(self) -> t.Tuple[int, int]:
        """
        tuple: Getter for film resolution as a (int, int) pair.
        """
        pass

    def sensor_infos(self) -> t.List[SensorInfo]:
        """
        Return a tuple of sensor information data structures.
        Sensor ids are generated with a set of suffixes.
        E.g. in the case of spp splitting, multiple ids, with
        the suffix '_sppXX' are generated with XX being
        an integer number.

        Subclasses may override this method and add other suffixes
        for their specific purposes.

        Returns
        -------
        list of :class:`.SensorInfo`
            List of sensor information data structures.
        """
        spps = self._split_spp()

        if len(spps) == 1:
            return [SensorInfo(id=f"{self.id}_ms0", spp=spps[0])]

        else:
            return [
                SensorInfo(id=f"{self.id}_ms0_spp{i}", spp=spp)
                for i, spp in enumerate(spps)
            ]

    def _split_spp(self) -> t.List[int]:
        """
        Generate sensor specifications, possibly applying sample count splitting
        in single-precision mode.

        Sample count (or SPP) splitting consists in splitting sample
        count among multiple sensors if a high enough sample count (*i.e.*
        greater than ``self._spp_splitting_threshold``) is requested when using
        a single-precision mode in order to preserve the accuracy of results.

        Sensor records will have to be combined using
        :meth:`.postprocess_results`.

        Returns
        -------
        list of int
            List of split SPPs if relevant.
        """

        if (
            not eradiate.mode().has_flags(ModeFlags.ANY_DOUBLE)
            and self.spp > self._spp_splitting_threshold
        ):
            spps = [
                self._spp_splitting_threshold
                for i in range(int(self.spp / self._spp_splitting_threshold))
            ]
            if self.spp % self._spp_splitting_threshold:
                spps.append(self.spp % self._spp_splitting_threshold)

            return spps

        else:
            return [self.spp]

    def postprocess(self) -> xr.Dataset:
        """
        Measure post-processing pipeline. The default implementation simply
        aggregates SPP-split raw results and computes CKD quadrature if relevant.
        Overloads can perform additional post-processing tasks and add metadata.

        Returns
        -------
        Dataset
            Post-processed results.
        """
        result = self.results.to_dataset(aggregate_spps=True)

        if eradiate.mode().has_flags(ModeFlags.ANY_CKD):
            result = self._postprocess_ckd_eval_quad(result)

        return result

    def _postprocess_ckd_eval_quad(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Evaluate quadrature in CKD mode and reindex data.

        Parameters
        ----------
        ds : Dataset
            Raw result dataset as generated by the
            :meth:`.MeasureResults.to_dataset` method, *i.e.* with a ``raw``
            data variable and  a multi-level index attached to a ``bd``
            dimension.

        Returns
        -------
        Dataset
            Post-processed results with the quadrature computed for each film
            pixel, and the ``bd`` dimension replaced by a wavelength dimension
            coordinate.
        """
        bins = list(OrderedDict.fromkeys(ds.bin.to_index()))  # (deduplicate list)
        sensor_ids = ds.sensor_id.values
        ys = ds.y.values
        xs = ds.x.values
        quad = self.spectral_cfg.bin_set.quad

        n_bin = len(bins)
        n_sensor_ids = len(sensor_ids)
        n_y = len(ys)
        n_x = len(xs)

        # Collect wavelengths associated with each bin
        wavelength_units = ucc.get("wavelength")
        wavelengths = [
            bin.wcenter.m_as(wavelength_units)
            for bin in self.spectral_cfg.bin_set.select_bins(("ids", {"ids": bins}))
        ]

        # Init storage
        result = xr.Dataset(
            {
                "raw": (
                    ("w", "sensor_id", "y", "x"),
                    np.zeros((n_bin, n_sensor_ids, n_y, n_x)),
                )
            },
            coords={"w": wavelengths, "sensor_id": sensor_ids, "y": ys, "x": xs},
        )

        # For each bin and each pixel, compute quadrature and store the result
        for i_bin, bin in enumerate(bins):
            values_at_nodes = ds.raw.sel(bin=bin).values

            # Rationale: Avoid using xarray's indexing in this loop for
            # performance reasons (wrong data indexing method will result in
            # 10x+ speed reduction)
            for (i_sensor_id, i_y, i_x) in itertools.product(
                range(n_sensor_ids), range(n_y), range(n_x)
            ):
                result.raw.values[i_bin, i_sensor_id, i_y, i_x] = quad.integrate(
                    values_at_nodes[:, i_sensor_id, i_y, i_x],
                    interval=np.array([0.0, 1.0]),
                )

        # Copy lost metadata
        for var in list(result.data_vars) + list(result.coords):
            if var == "w":
                result[var].attrs = {
                    "standard_name": "wavelength",
                    "long_description": "wavelength",
                    "units": symbol(wavelength_units),
                }
            else:
                result[var].attrs = ds[var].attrs.copy()

        return result

    @abstractmethod
    def _base_dicts(self) -> t.List[t.Dict]:
        """
        Return a list (one item per sensor) of dictionaries defining parameters
        not related with the film or the sampler.
        """
        pass

    def _film_dicts(self) -> t.List[t.Dict]:
        """
        Return a list (one item per sensor) of dictionaries defining parameters
        related with the film.
        """
        return [
            {
                "film": {
                    "type": "hdrfilm",
                    "width": self.film_resolution[0],
                    "height": self.film_resolution[1],
                    "pixel_format": "luminance",
                    "component_format": "float32",
                    "rfilter": {"type": "box"},
                }
            }
        ] * len(self.sensor_infos())

    def _sampler_dicts(self) -> t.List[t.Dict]:
        """
        Return a list (one item per sensor) of dictionaries defining parameters
        related with the sampler.
        """
        return [
            {"sampler": {"type": "independent", "sample_count": sensor_info.spp}}
            for sensor_info in self.sensor_infos()
        ]

    def kernel_dict(self, ctx: KernelDictContext) -> KernelDict:
        result = {
            f"{sensor_info.id}": {
                **base_dict,
                **sampler_dict,
                **film_dict,
            }
            for i, (sensor_info, base_dict, sampler_dict, film_dict) in enumerate(
                zip(
                    self.sensor_infos(),
                    self._base_dicts(),
                    self._sampler_dicts(),
                    self._film_dicts(),
                )
            )
        }
        return KernelDict(result)
