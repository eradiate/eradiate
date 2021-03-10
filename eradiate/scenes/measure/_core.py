from abc import ABC, abstractmethod

import attr
import numpy as np

import eradiate

from ..core import SceneElement
from ... import validators
from ..._attrs import documented, get_doc, parse_docs
from ..._factory import BaseFactory


@parse_docs
@attr.s
class SensorInfo:
    """Data type to store information about a sensor associated with a measure."""

    id = documented(
        attr.ib(),
        doc="Sensor unique identifier.",
        type="str",
    )
    spp = documented(
        attr.ib(),
        doc="Sensor sample count.",
        type="int",
    )


@parse_docs
@attr.s
class Measure(SceneElement, ABC):
    """Abstract base class for all measure scene elements."""

    id = documented(
        attr.ib(
            default="measure",
            validator=attr.validators.optional((attr.validators.instance_of(str))),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default='"measure"',
    )

    spp = documented(
        attr.ib(default=32, converter=int, validator=validators.is_positive),
        doc="Number of samples per pixel.",
        type="int",
        default="32",
    )

    film_resolution = documented(
        attr.ib(
            default=(32, 32),
            validator=attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(int),
                iterable_validator=validators.has_len(2),
            ),
        ),
        doc="Film resolution as a (width, height) 2-tuple. "
        "If the height is set to 1, direction sampling will be restricted to a "
        "plane.",
        type="array-like[int, int]",
        default="(32, 32)",
    )

    # Private attributes
    # Sample count which, if exceeded, should trigger sample count splitting in
    # single-precision modes
    _spp_splitting_threshold = attr.ib(
        default=int(1e5), converter=int, validator=validators.is_positive, repr=False
    )

    def postprocess_results(self, runner_results):
        """Process sensor data to extract measure results. These post-processing
        operations can include (but are not limited to) array reshaping and
        sensor data aggregation and transformation operations.

        Parameter ``runner_results`` (dict):
            Dictionary mapping sensor IDs to their respective results.

            .. seealso:: :func:`eradiate.solvers.core.runner`

        Returns → array:
            Processed results.
        """
        sensor_values = np.array([runner_results[x.id] for x in self.sensor_infos()])
        sensor_spps = np.array([x.spp for x in self.sensor_infos()])
        spp_sum = np.sum(sensor_spps)

        # Compute weighted sum of sensor contributions
        # transpose() is required to correctly position the dimension on which
        # dot() operates
        result = np.dot(sensor_values.transpose(), sensor_spps).transpose()
        result /= spp_sum

        return np.reshape(result, (result.shape[0], result.shape[1]))

    def sensor_infos(self):
        """Return a tuple of sensor information data structures.

        Returns → list[:class:`.SensorInfo`]:
            List of sensor
        """
        spps = self._split_spp()

        if len(spps) == 1:
            return [SensorInfo(id=f"{self.id}", spp=spps[0])]

        else:
            return [
                SensorInfo(id=f"{self.id}_{i}", spp=spp) for i, spp in enumerate(spps)
            ]

    def _split_spp(self):
        """Generate sensor specifications, possibly applying sample count
        splitting in single-precision mode.

        Sample count (or SPP) splitting consists in splitting sample
        count among multiple sensors if a high enough sample count (_i.e._
        greater than ``self._spp_splitting_threshold``) is requested when using
        a single-precision mode in order to preserve the accuracy of results.

        Sensor records will have to be combined using
        :meth:`.postprocess_results`.

        Returns → list[float]:
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

    def kernel_dict(self, ref=True):
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
    """This factory constructs objects whose classes are derived from
    :class:`Measure`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: MeasureFactory
    """

    _constructed_type = Measure
    registry = {}
