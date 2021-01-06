"""Biosphere-related scene generation facilities.

.. admonition:: Registered factory members [:class:`BiosphereFactory`]
    :class: hint

    .. factorytable::
       :factory: BiosphereFactory
"""

from abc import ABC

import aabbtree
import attr
import numpy as np

from .core import SceneElement
from .spectra import Spectrum, SpectrumFactory
from ..util.attrs import (
    attrib_quantity,
    validator_has_len,
    validator_has_quantity,
    validator_is_positive
)
from ..util.factory import BaseFactory
from ..util.units import config_default_units as cdu
from ..util.units import kernel_default_units as kdu
from ..util.units import ureg


class BiosphereFactory(BaseFactory):
    """This factory constructs objects whose classes are derived from
    :class:`.SceneElement`.

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: BiosphereFactory
    """

    _constructed_type = SceneElement
    registry = {}


@attr.s
class Canopy(SceneElement, ABC):
    """An abstract base class defining a base type for all canopies.
    """


@BiosphereFactory.register("homogeneous_discrete_canopy")
@attr.s
class HomogeneousDiscreteCanopy(Canopy):
    """A generator for the `homogenous discrete canopy used in the RAMI benchmark
    <https://rami-benchmark.jrc.ec.europa.eu/_www/phase/phase_exp.php?strTag=level3&strNext=meas&strPhase=RAMI3&strTagValue=HOM_SOL_DIS>`_.
    This is a re-implementation of the generator in raytran.
    Leaf orientation distribution is set by approximating the beta distribution
    as given in :cite:`MyneniRoss1991PhotonVegetationInteraction`.

    The configuration of this scene element allows for different variants:

    - Either the number of leaves `n_leaves` or the horizontal extent of the
      canopy can be specified. In case both are given, the horizontal extent
      will take precedence
    - Either the `hdo`/`hvr` values or the vertical extent of the canopy can be
      specified. If both are given, the vertical extent will take precedence.

    Horizontal and vertical extent are specified through the `size`
    parameter.

    .. rubric:: Constructor arguments / instance attributes

    ``id`` (str):
        Identifier. Default value: "homogeneous_discrete_canopy"

    ``lai`` (float):
        Leaf area index. Default value: 3.

    ``radius`` (float):
        Leaf radius. Default 0.25.

        Unit-enabled field (default unit: cdu[length])

    ``mu`` (float):
        First parameter for the beta distribution. Default: 1.066.

    ``nu`` (float):
        Second parameter for the beta distribution. Default value: 1.853.

    ``n_leaves`` (int):
        Total number of leaves to generate. If ``size`` is set, it will override
        this parameter. Default: 4000

    ``hdo`` (float):
        Mean horizontal distance between leaves. If ``size`` is set, it will
        override this parameter. Default: 1.

        Unit-enabled field (default unit: cdu[length])

    ``hvr`` (float):
        Ratio of mean horizontal leaf distance and vertical canopy extent.
        If ``size`` is set, it will override this parameter. Default: 1.

    ``size`` (list[float]):
        Length of the canopy in the three dimensions. A canopy with size
        [x, y, z] will extend over [-x/2, x/2], [-y/2, y/2] and [0, z], centered
        at ``position`` parameter. Default: [0, 0, 0].

        If ``size`` is not set, it will be automatically computed from ``hdo``,
        ``hvr`` and ``n_leaves``.

        Unit-enabled field (default units: cdu[length]).

    ``position`` (list[float]):
        Three dimensional position of the canopy. Default: [0, 0, 0].

        Unit-enabled field (default units: cdu[length])

    ``seed`` (int):
        Seed for the random number generator. Default: 1.

    ``avoid_overlap`` (bool):
        If ``True``, the scene element will attempt to place the leaves such
        that they do not overlap. If a leaf cannot be placed without overlap
        after 1e6 tries, it will raise a ``RuntimeError``. Default: True.

        .. admonition:: Note

           To emulate the behaviour of the raytran leaf cloud generator
           simply try instantiating the leaf cloud several times with
           different ``seed`` values and after a certain amount of
           failures, run it without overlap avoidance.

    ``leaf_reflectance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
        Reflectance spectrum of the leaves in the cloud. Must be a reflectance
        spectrum (dimensionless). Default: 0.5.

    ``leaf_transmittance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
        Transmittance spectrum of the leaves in the cloud. Must be a
        transmittance spectrum (dimensionless). Default: 0.5.
    """

    id = attr.ib(
        default="homogeneous_discrete_canopy",
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )

    n_leaves = attr.ib(
        default=4000, converter=int, validator=validator_is_positive
    )

    seed = attr.ib(
        default=1,
        converter=int,
        validator=validator_is_positive,
    )

    avoid_overlap = attr.ib(default=True, converter=bool)

    leaf_area_index = attr.ib(
        default=3.,
        converter=float,
        validator=validator_is_positive,
    )

    mu = attr.ib(
        default=1.066,
        converter=float,
        validator=validator_is_positive,
    )

    nu = attr.ib(
        default=1.853,
        converter=float,
        validator=validator_is_positive,
    )

    hvr = attr.ib(
        default=1.,
        converter=float,
        validator=validator_is_positive,
    )

    hdo = attrib_quantity(
        default=ureg.Quantity(1., ureg.m),
        units_compatible=cdu.generator("length"),
    )

    leaf_radius = attrib_quantity(
        default=ureg.Quantity(0.25, ureg.m),
        units_compatible=cdu.generator("length"),
    )

    position = attrib_quantity(
        default=ureg.Quantity([0, 0, 0], ureg.m),
        validator=validator_has_len(3),
        units_compatible=cdu.generator("length"),
    )

    size = attrib_quantity(
        default=ureg.Quantity([0, 0, 0], ureg.m),
        validator=validator_has_len(3),
        units_compatible=cdu.generator("length"),
    )

    leaf_reflectance = attr.ib(
        default=0.5,
        converter=SpectrumFactory.converter("reflectance"),
        validator=[
            attr.validators.instance_of(Spectrum),
            validator_has_quantity("reflectance")
        ]
    )

    leaf_transmittance = attr.ib(
        default=0.5,
        converter=SpectrumFactory.converter("transmittance"),
        validator=[
            attr.validators.instance_of(Spectrum),
            validator_has_quantity("transmittance")
        ]
    )

    _positions = attrib_quantity(default=[] * ureg.m, init=False)
    _leaf_normals = attr.ib(default=[], init=False)
    _tries = attr.ib(default=1000000, init=False)

    def __attrs_post_init__(self):
        # set the seed for the RNG
        np.random.seed(self.seed)

        # n_leaves or horizontal extent
        if self.size[0].magnitude == 0 or self.size[1].magnitude == 0:
            self.size[0] = self.size[1] = \
                np.sqrt(
                    self.n_leaves * np.pi *
                    self.leaf_radius ** 2 / self.leaf_area_index
                )
        else:
            self.n_leaves = int(np.floor(
                self.size[0] * self.size[1] * self.leaf_area_index /
                (np.pi * self.leaf_radius ** 2)
            ))

        # hdo/hvr or vertical extent
        if self.size[2].magnitude == 0:
            self.size[2] = (
                    self.leaf_area_index * self.hdo ** 3 /
                    (np.pi * self.leaf_radius ** 2 * self.hvr)
            )
        else:
            self.hdo = np.power(
                np.pi * self.leaf_radius ** 2 * self.size[2] * self.hvr /
                self.leaf_area_index, 1 / 3.
            )

    def _inversebeta(self):
        while True:
            rands = np.random.rand(2)
            s1 = np.power(rands[0], 1. / self.mu)
            s2 = np.power(rands[1], 1. / self.nu)
            s = s1 + s2
            if s <= 1:
                return s1 / s

    def _compute_positions(self):
        size_magnitude = self.size.to(ureg.m).magnitude
        radius_mag = self.leaf_radius.to(ureg.m).magnitude
        positions_temp = []
        tree = aabbtree.AABBTree()

        for i in range(self.n_leaves):
            if not self.avoid_overlap:
                rand = np.random.rand(3)
                positions_temp.append([
                    rand[0] * size_magnitude[0] - size_magnitude[0] / 2.,
                    rand[1] * size_magnitude[1] - size_magnitude[1] / 2.,
                    rand[2] * size_magnitude[2]
                ])
            else:
                for j in range(self._tries):
                    rand = np.random.rand(3)
                    pos_candidate = [
                        rand[0] * size_magnitude[0] - size_magnitude[0] / 2.,
                        rand[1] * size_magnitude[1] - size_magnitude[1] / 2.,
                        rand[2] * size_magnitude[2]
                    ]
                    aabb = aabbtree.AABB([
                        (
                            pos_candidate[0] - radius_mag,
                            pos_candidate[0] + radius_mag
                        ),
                        (
                            pos_candidate[1] - radius_mag,
                            pos_candidate[1] + radius_mag
                        ),
                        (
                            pos_candidate[2] - radius_mag,
                            pos_candidate[2] + radius_mag
                        )
                    ])
                    if i == 0:
                        positions_temp.append(pos_candidate)
                        tree.add(aabb)
                        break
                    else:
                        if not tree.does_overlap(aabb):
                            positions_temp.append(pos_candidate)
                            tree.add(aabb)
                            break
                else:
                    raise RuntimeError(
                        "unable to place all leaves: "
                        "the specified canopy might be too dense"
                    )
        self._positions = positions_temp * ureg.m

    def kernel_dict(self, ref=True):
        from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f
        return_dict = {
            "leaf": {
                "type": "shapegroup",
                "shape_01": {
                    "type": "disk",
                    "bsdf": {
                        "type": "bilambertian",
                        "reflectance":
                            self.leaf_reflectance.kernel_dict()["spectrum"],
                        "transmittance":
                            self.leaf_transmittance.kernel_dict()["spectrum"],
                    }
                }
            }
        }

        self._compute_positions()
        kdu_length = kdu.get("length")
        radius = self.leaf_radius.to(kdu_length).magnitude

        for i in range(len(self._positions)):
            theta = np.rad2deg(self._inversebeta())
            phi = np.random.rand() * 360.

            to_world = (
                    ScalarTransform4f.translate(ScalarVector3f(
                        self.position.to(kdu_length).magnitude +
                        self._positions[i].to(kdu_length).magnitude)
                    ) *
                    ScalarTransform4f.rotate(ScalarVector3f(0, 0, 1), phi) *
                    ScalarTransform4f.rotate(ScalarVector3f(0, -1, 0), theta) *
                    ScalarTransform4f.scale(ScalarVector3f(radius, radius, 1))
            )

            return_dict[f"shape_{i}"] = {
                "type": "instance",
                "group": {
                    "type": "ref",
                    "id": "leaf"
                },
                "to_world": to_world
            }

        return return_dict
