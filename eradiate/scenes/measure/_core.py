from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import attr
import numpy as np
import pint
import pinttr
import xarray
import xarray as xr

import eradiate

from ..core import SceneElement
from ... import converters, validators
from ..._attrs import documented, get_doc, parse_docs
from ..._factory import BaseFactory
from ..._units import unit_context_config as ucc
from ..._units import unit_registry as ureg
from ..._util import ensure_array
from ...contexts import MonoSpectralContext, SpectralContext
from ...exceptions import ModeError, UnsupportedModeError


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


class MeasureSpectralConfig(ABC):
    """
    Data structure specifying the spectral configuration of a :class:`.Measure`
    in monochromatic modes.

    While this class is abstract, it should however be the main entry point
    to create :class:`.MeasureSpectralConfig` child class objects through the
    :meth:`.MeasureSpectralConfig.new` class method constructor.
    """

    @abstractmethod
    def spectral_ctxs(self) -> List[SpectralContext]:
        """
        Return a list of :class:`.SpectralContext` objects based on the
        stored spectral configuration. These data structures can be used to
        drive the evaluation of spectrally dependent components during a
        spectral loop.
        """
        pass

    @staticmethod
    def new(**kwargs) -> "MeasureSpectralConfig":
        """
        Create a new instance of one of the :class:`SpectralContext` child
        classes. *The instantiated class is defined based on the currently active
        mode.* Keyword arguments are passed to the instantiated class's
        constructor:

        .. rubric:: Monochromatic modes [:class:`MonoMeasureSpectralConfig`]

        Parameter ``wavelengths`` (:class:`pint.Quantity`):
            List of wavelengths (automatically converted to a Numpy array).
            Default: [550] nm.

            Unit-enabled field (default: ucc[wavelength]).

        .. seealso::

           * :func:`eradiate.mode`
           * :func:`eradiate.set_mode`
        """
        mode = eradiate.mode()

        if mode.is_monochromatic():
            return MonoMeasureSpectralConfig(**kwargs)

        raise ModeError(f"unsupported mode '{mode.id}'")

    @staticmethod
    def from_dict(d: Dict) -> "MeasureSpectralConfig":
        """
        Create from a dictionary. This class method will additionally pre-process
        the passed dictionary to merge any field with an associated ``"_units"``
        field into a :class:`pint.Quantity` container.

        Parameter ``d`` (dict):
            Configuration dictionary used for initialisation.

        Returns → :class:`.MeasureSpectralConfig`:
            Created object.
        """

        # Pre-process dict: apply units to unit-enabled fields
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Perform object creation
        return MeasureSpectralConfig.new(**d_copy)

    @staticmethod
    def convert(value):
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
            converter=lambda x: converters.on_quantity(ensure_array)(
                pinttr.converters.ensure_units(x, ucc.get("wavelength"))
            ),
        ),
        doc="List of wavelengths on which to perform the monochromatic spectral "
        "loop.\n\nUnit-enabled field (default: ucc[wavelength]).",
        type=":class:`pint.Quantity`",
        default="[550.0] nm",
    )

    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value

    def spectral_ctxs(self) -> List[MonoSpectralContext]:
        return [
            MonoSpectralContext(wavelength=wavelength)
            for wavelength in self._wavelengths
        ]


@parse_docs
@attr.s
class MeasureResults:
    """
    Data structure storing simulation results corresponding to a measure.
    """

    raw: Optional[Dict] = documented(
        attr.ib(
            factory=dict,
            validator=attr.validators.optional(attr.validators.instance_of(dict)),
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

    def to_dataset(self, aggregate_spps=False) -> xarray.Dataset:
        """
        Repack raw results as a :class:`xarray.Dataset`. Dimension coordinates are
        as follows:

        * spectral coordinate (varies vs active mode)
        * ``sensor_id``: kernel sensor identified (for SPP split, dropped if
          ``aggregate_spps`` is ``True``)
        * ``x``: film width
        * ``y``: film height

        Parameter ``aggregate_spps`` (bool):
            If ``True``, perform split SPP aggregation (*i.e.* sum results using
            SPP values as weights). This will result in the ``sensor_id``
            dimension being dropped.

        Returns → :class:`~xarray.Dataset`:
            Raw sensor data repacked as a :class:`~xarray.Dataset`.
        """
        if not self.raw:
            raise ValueError("no raw results to convert to xarray.Dataset")

        if eradiate.mode().is_monochromatic():
            spectral_coord_label = eradiate.mode().spectral_coord_label
            spectral_coord_metadata = {
                "long_name": "wavelength",
                "units": str(ucc.get("wavelength")),
            }
            # TODO: Add channel dimension to dataset (string-labelled;
            #  value 'intensity' in mono modes;
            #  values 'R', 'G', 'B' in render modes)
        else:
            raise UnsupportedModeError(supported="monochromatic")

        # Collect coordinate values
        spectral_coords = set()
        sensor_ids = set()
        film_size = np.zeros((2,), dtype=int)

        for spectral_coord, val in self.raw.items():
            spectral_coords.add(spectral_coord)
            for sensor_id, data in val["values"].items():
                sensor_ids.add(sensor_id)
                film_size = np.maximum(film_size, data.shape[:2])

        spectral_coords = sorted(spectral_coords)
        sensor_ids = sorted(sensor_ids)

        # Collect values
        data = np.full((len(spectral_coords), len(sensor_ids), *film_size), np.nan)

        for i_spectral, spectral_coord in enumerate(spectral_coords):
            for i_sensor, sensor_id in enumerate(sensor_ids):
                # Note: This doesn't handle heterogeneous sensor film sizes
                # (i.e. cases in which sensors have different film sizes).
                # To add support for it, blitting is probably a good approach
                # https://stackoverflow.com/questions/28676187/numpy-blit-copy-part-of-an-array-to-another-one-with-a-different-size
                data[i_spectral, i_sensor] = self.raw[spectral_coord]["values"][
                    sensor_id
                ][..., 0]
                # This latter indexing selects only one channel in the raw data
                # array: this works with mono variants but will fail otherwise

        # Collect sample counts
        spps = np.full((len(spectral_coords), len(sensor_ids)), np.nan, dtype=int)

        for i_spectral, spectral_coord in enumerate(spectral_coords):
            for i_sensor, sensor_id in enumerate(sensor_ids):
                spps[i_spectral, i_sensor] = self.raw[spectral_coord]["spp"][sensor_id]

        # Construct dataset
        result = xr.Dataset(
            data_vars=(
                {
                    "raw": (
                        [spectral_coord_label, "sensor_id", "x", "y"],
                        data,
                        {"long name": "raw sensor values"},
                    ),
                    "spp": (
                        [spectral_coord_label, "sensor_id"],
                        spps,
                        {"long name": "sample count"},
                    ),
                }
            ),
            coords={
                "x": (
                    "x",
                    [float(x) for x in range(film_size[0])],  # Conversion to float is
                    {
                        "long_name": "film width coordinate"
                    },  # intentional (interpolation)
                ),
                "y": (
                    "y",
                    [float(x) for x in range(film_size[1])],
                    {"long_name": "film height coordinate"},
                ),
                spectral_coord_label: (
                    spectral_coord_label,
                    spectral_coords,
                    spectral_coord_metadata,
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

        # Aggregate SPPs and drop sensor ID dimension
        # Note: This will not work if multiple sensors are used for a purpose other
        # than SPP splitting; should this happen, this part must be updated
        weights = result.data_vars["spp"]
        result = result.assign(
            {"raw": result["raw"].weighted(weights).mean(dim="sensor_id")}
        )
        result = result.assign({"spp": result.data_vars["spp"].sum(dim="sensor_id")})

        return result.drop_dims("sensor_id")


@parse_docs
@attr.s
class Measure(SceneElement, ABC):
    """
    Abstract base class for all measure scene elements.
    """

    id: Optional[str] = documented(
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
        "interpreted by :meth:`MeasureSpectralConfig.from_dict`.",
        type=":meth:`MeasureSpectralConfig.new() <.MeasureSpectralConfig.new>`",
        default="None",
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
        default=":class:`MeasureResults() <.MeasureResults>`",
        type=":class:`.MeasureResults`",
    )

    # Private attributes
    # Sample count which, if exceeded, should trigger sample count splitting in
    # single-precision modes
    _spp_splitting_threshold: int = attr.ib(
        default=int(1e5), converter=int, validator=validators.is_positive, repr=False
    )

    @property
    @abstractmethod
    def film_resolution(self) -> Tuple[int, int]:
        """
        Getter for film resolution.
        """
        pass

    def sensor_infos(self) -> List[SensorInfo]:
        """
        Return a tuple of sensor information data structures.

        Returns → list[:class:`.SensorInfo`]:
            List of sensor information data structures.
        """
        spps = self._split_spp()

        if len(spps) == 1:
            return [SensorInfo(id=f"{self.id}", spp=spps[0])]

        else:
            return [
                SensorInfo(id=f"{self.id}_{i}", spp=spp) for i, spp in enumerate(spps)
            ]

    def _split_spp(self) -> List[int]:
        """
        Generate sensor specifications, possibly applying sample count splitting
        in single-precision mode.

        Sample count (or SPP) splitting consists in splitting sample
        count among multiple sensors if a high enough sample count (*i.e.*
        greater than ``self._spp_splitting_threshold``) is requested when using
        a single-precision mode in order to preserve the accuracy of results.

        Sensor records will have to be combined using
        :meth:`.postprocess_results`.

        Returns → list[int]:
            List of split SPPs if relevant.
        """
        mode = eradiate.mode()

        if mode.is_single_precision() and self.spp > self._spp_splitting_threshold:
            spps = [
                self._spp_splitting_threshold
                for i in range(int(self.spp / self._spp_splitting_threshold))
            ]
            if self.spp % self._spp_splitting_threshold:
                spps.append(self.spp % self._spp_splitting_threshold)

            return spps

        else:
            return [self.spp]

    def _raw_results_aggregated(self) -> xarray.Dataset:
        """
        Aggregate raw sensor results if multiple sensors were used.

        Returns → :class:`xarray.Dataset`:
            Processed results.
        """
        if self.results is None:
            raise ValueError("no raw results stored, cannot aggregate")

        ds = self.results
        weights = xr.DataArray(
            ds.data_vars["spp"],
            coords={"sensor_id": ds.coords["sensor_id"]},
            dims=["sensor_id"],
        )
        return ds.weighted(weights).mean(dim="sensor_id")

    def postprocessed_results(self) -> xarray.Dataset:
        """
        Return post-processed raw sensor results.

        Returns → :class:`~xarray.Dataset`:
            Post-processed results.
        """
        # Default implementation simply aggregates SPP-split raw results;
        # overloads can perform additional post-processing and add metadata
        return self.results.to_dataset(aggregate_spps=True)

    @abstractmethod
    def _base_dicts(self):
        pass

    def _film_dicts(self):
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

    def _sampler_dicts(self):
        return [
            {"sampler": {"type": "independent", "sample_count": sensor_info.spp}}
            for sensor_info in self.sensor_infos()
        ]

    def kernel_dict(self, ctx=None):
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
        return result


class MeasureFactory(BaseFactory):
    """
    This factory constructs objects whose classes are derived from :class:`Measure`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: MeasureFactory
    """

    _constructed_type = Measure
    registry = {}
