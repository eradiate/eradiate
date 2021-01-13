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
import os
import pint
import warnings

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
from ..util.units import ureg, ensure_units


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
    """An abstract base class defining a base type for all canopies."""


@BiosphereFactory.register("homogeneous_discrete_canopy")
@attr.s
class HomogeneousDiscreteCanopy(Canopy):
    """A generator for the `homogenous discrete canopy used in the RAMI benchmark
    <https://rami-benchmark.jrc.ec.europa.eu/_www/phase/phase_exp.php?strTag=level3&strNext=meas&strPhase=RAMI3&strTagValue=HOM_SOL_DIS>`_.

    This canopy can be instantiated in two ways:

    - The classmethod :meth:`~eradiate.scenes.biosphere.HomogeneousDiscreteCanopy.from_parameters`
      takes a set of parameters and will generate the canopy from them. For details,
      please see the documentation of that method.
    - The classmethod :meth:`~eradiate.scenes.biosphere.HomogeneousDiscreteCanopy.from_files`
      will read a set of definition files that specify the leaves of the canopy.
      Please refer to this method for details on the file format.
    """

    lai = attr.ib()
    n_leaves = attr.ib()

    id = attr.ib(
        default="homogeneous_discrete_canopy",
        validator=attr.validators.optional(attr.validators.instance_of(str)),
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

    center_position = attrib_quantity(
        default=ureg.Quantity([0, 0, 0], ureg.m),
        validator=validator_has_len(3),
        units_compatible=cdu.generator("length"),
    )

    transforms = attr.ib(default=[])

    @staticmethod
    def get_quantity_param(params_dict, name, unit, default):
        try:
            param = params_dict[name]
        except KeyError:
            param = default

        try:
            unit = params_dict[f"{name}_units"]
        except KeyError:
            unit = unit

        return ensure_units(param, unit)

    @classmethod
    def from_parameters(cls, size=[30,30,3], lai=3, mu=1.066, nu=1.853,
                        leaf_reflectance=0.5, leaf_transmittance=0.5,
                        leaf_radius=0.1, n_leaves=4000, center_position=[0,0,0],
                        hdo=1, hvr=1, seed=1,
                        avoid_overlap=False):
        """
        This method allows the creation of a HomogeneousDiscreteCanopy
        from statistical parameters.
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

        ``size`` (list[float]):
            Length of the canopy in the three dimensions. A canopy with size
            [x, y, z] will extend over [-x/2, x/2], [-y/2, y/2] and [0, z], centered
            at ``position`` parameter. Default: [30, 30, 3].

            If ``size`` is not set, it will be automatically computed from ``hdo``,
            ``hvr`` and ``n_leaves``.

            Unit-enabled field (default units: cdu[length]).

        ``lai`` (float):
            Leaf area index. Physical range: [0, 10], Default value: 3.

        ``mu`` (float):
            First parameter for the beta distribution. Default: 1.066.

        ``nu`` (float):
            Second parameter for the beta distribution. Default value: 1.853.

        ``leaf_reflectance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
            Reflectance spectrum of the leaves in the cloud. Must be a reflectance
            spectrum (dimensionless). Default: 0.5.

        ``leaf_transmittance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
            Transmittance spectrum of the leaves in the cloud. Must be a
            transmittance spectrum (dimensionless). Default: 0.5.

        ``radius`` (float):
            Leaf radius. Physical range: [0, height/2.], Default 0.1.

            Unit-enabled field (default unit: cdu[length])

        ``n_leaves`` (int):
            Total number of leaves to generate. If ``size`` is set, it will override
            this parameter. Default: 4000

        ``center_position`` (list[float]):
            Three dimensional position of the canopy. Default: [0, 0, 0].

            Unit-enabled field (default units: cdu[length])

        ``hdo`` (float):
            Mean horizontal distance between leaves. If ``size`` is set, it will
            override this parameter. Default: 1.

            Unit-enabled field (default unit: cdu[length])

        ``hvr`` (float):
            Ratio of mean horizontal leaf distance and vertical canopy extent.
            If ``size`` is set, it will override this parameter. Default: 1.

        ``seed`` (int):
            Seed for the random number generator. Default: 1.

        ``avoid_overlap`` (bool):
            If ``True``, the scene element will attempt to place the leaves such
            that they do not overlap. If a leaf cannot be placed without overlap
            after 1e6 tries, it will raise a ``RuntimeError``. Default: False.

            .. warning::

                Depending on the canopy specification instantiation can take
                a very long time if overlap avoidance is active!

            .. admonition:: Note

               To emulate the behaviour of the raytran leaf cloud generator
               simply try instantiating the leaf cloud several times with
               different ``seed`` values and after a certain amount of
               failures, run it without overlap avoidance.
        """

        from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f

        size = ensure_units(size, ureg.m)
        lai = ensure_units(lai, None)
        leaf_radius = ensure_units(leaf_radius, ureg.m)
        center_position = ensure_units(center_position, ureg.m)
        hdo = ensure_units(hdo, None)
        hdo = ensure_units(hdo, None)

        np.random.seed(seed)
        
        # n_leaves or horizontal extent
        if size[0].magnitude == 0 or size[1].magnitude == 0:
            size[0] = size[1] = \
                np.sqrt(
                    n_leaves * np.pi *
                    leaf_radius ** 2 / lai
                )
        else:
            n_leaves = int(np.floor(
                size[0] * size[1] * lai /
                (np.pi * leaf_radius ** 2)
            ))

        # hdo/hvr or vertical extent
        if size[2].magnitude == 0:
            size[2] = (
                lai * hdo ** 3 /
                (
                    np.pi * leaf_radius ** 2 * hvr)
                )

        positions = cls._compute_positions(size, leaf_radius,
                                           n_leaves, avoid_overlap)
        kdu_length = kdu.get("length")
        radius = leaf_radius.to(kdu_length).magnitude

        # warnings for non physical values
        if radius > size[2].to(kdu_length).magnitude:
            warnings.warn(f"Leaf radius {radius} is larger than the canopy"
                          f"height {size[2]}. The leaves might not fit inside"
                          f"the specified volume.")

        transforms = []
        for pos in positions:
            theta = np.rad2deg(cls._inversebeta(mu, nu))
            phi = np.random.rand() * 360.

            to_world = (
                    ScalarTransform4f.translate(ScalarVector3f(
                        pos.to(kdu_length).magnitude)
                    ) *
                    ScalarTransform4f.rotate(ScalarVector3f(0, 0, 1), phi) *
                    ScalarTransform4f.rotate(ScalarVector3f(0, -1, 0), theta) *
                    ScalarTransform4f.scale(ScalarVector3f(radius, radius, 1))
            )
            transforms.append(to_world)

        return HomogeneousDiscreteCanopy(leaf_reflectance=leaf_reflectance,
                                         leaf_transmittance=leaf_transmittance,
                                         center_position=center_position,
                                         transforms=transforms,
                                         lai=lai, n_leaves=n_leaves)

    @staticmethod
    def _inversebeta(mu, nu):
        while True:
            rands = np.random.rand(2)
            s1 = np.power(rands[0], 1. / mu)
            s2 = np.power(rands[1], 1. / nu)
            s = s1 + s2
            if s <= 1:
                return s1 / s

    @staticmethod
    def _compute_positions(size, leaf_radius, n_leaves, tries=10000000, avoid_overlap=False):
        size_magnitude = size.to(ureg.m).magnitude
        radius_mag = leaf_radius.to(ureg.m).magnitude
        positions_temp = []
        tree = aabbtree.AABBTree()

        for i in range(n_leaves):
            if not avoid_overlap:
                rand = np.random.rand(3)
                positions_temp.append([
                    rand[0] * size_magnitude[0] - size_magnitude[0] / 2.,
                    rand[1] * size_magnitude[1] - size_magnitude[1] / 2.,
                    rand[2] * size_magnitude[2]
                ])
            else:
                for j in range(tries):
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
        return positions_temp * ureg.m

    @classmethod
    def from_file(cls, file_path, leaf_reflectance=0.5, leaf_transmittance=0.5,
                  center_position=[0,0,0]):
        """
        This method allows construcint the Canopy from a text file, specifying
        the individual leaves. The file must specify one leaf per line with the
        following seven parameters separated by one space:

        - The leaf radius
        - The x, y and z component of the leaf center
        - The x, y and z component of the leaf normal

        All values are given in meters.

        .. rubric:: Constructor arguments

        ``file_path`` (string or Path-like object):
            Path to the text file specifying the leaves in the canopy.
            Can be absolute or relative.

        ``leaf_reflectance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
            Reflectance spectrum of the leaves in the cloud. Must be a reflectance
            spectrum (dimensionless). Default: 0.5.

        ``leaf_transmittance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
            Transmittance spectrum of the leaves in the cloud. Must be a
            transmittance spectrum (dimensionless). Default: 0.5.

        ``center_position`` (list[float]):
            Three dimensional position of the canopy. Default: [0, 0, 0].

            Unit-enabled field (default units: cdu[length])
        """

        from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file at {file_path} found.")

        transforms = []
        leaf_area = 0
        max_x = 0
        min_x = 0
        max_y = 0
        min_y = 0
        with open(os.path.abspath(file_path), "r") as definition_file:
            for line in definition_file:
                values  = line.split(" ")
                kdu_length = kdu.get("length")

                radius = ensure_units(float(values[0]), "meter")
                radius = radius.to(kdu_length).magnitude

                leaf_area += radius*radius*np.pi

                position = ensure_units([float(values[1]),
                            float(values[2]),
                            float(values[3])], "meter")

                position = position.to(kdu_length).magnitude

                if position[0] > max_x:
                    max_x = position[0]
                if position[0] < min_x:
                    min_x = position[0]
                if position[1] > max_y:
                    max_y = position[1]
                if position[1] < min_y:
                    min_y = position[1]

                normal = ensure_units([float(values[4]),
                                       float(values[5]),
                                       float(values[6])], "meter")

                normal = normal.to(kdu_length).magnitude

                to_world = (
                    ScalarTransform4f.look_at(
                        origin=position,
                        target=position+normal,
                        up=np.cross(position, normal)
                    ) *
                    ScalarTransform4f.scale(
                        ScalarVector3f(radius, radius, 1))
                )
                transforms.append(to_world)

            lai = leaf_area / abs((max_x - min_x) * (max_y - min_y))

            return HomogeneousDiscreteCanopy(leaf_reflectance=leaf_reflectance,
                                             leaf_transmittance=leaf_transmittance,
                                             center_position=center_position,
                                             transforms=transforms,
                                             _lai=lai, _n_leaves=len(transforms))

    def kernel_dict(self, ref=True):
        from eradiate.kernel.core import ScalarTransform4f

        return_dict = {
            "leaf_bsdf": {
                "type": "bilambertian",
                "reflectance":
                    self.leaf_reflectance.kernel_dict()["spectrum"],
                "transmittance":
                    self.leaf_transmittance.kernel_dict()["spectrum"],
            },
            "leaves": {
                "type": "shapegroup",
            }
        }

        for i, transform in enumerate(self.transforms):
            return_dict["leaves"][f"leaf_{i}"] = {
                "type": "disk",
                "bsdf": {
                    "type": "ref",
                    "id": "leaf_bsdf"
                },
                "to_world": transform
            }

        return_dict["canopy"] = {
            "type": "instance",
            "group": {
                "type": "ref",
                "id": "leaves"
            },
            "to_world": ScalarTransform4f.translate(
                self.center_position.to(kdu.get("length")).magnitude
            )
        }

        return return_dict
