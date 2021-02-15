"""Biosphere-related scene generation facilities.

.. admonition:: Registered factory members [:class:`BiosphereFactory`]
    :class: hint

    .. factorytable::
       :factory: BiosphereFactory
"""

__all__ = ["BiosphereFactory", "HomogeneousDiscreteCanopy"]

import os
from abc import ABC, abstractmethod
from copy import copy

import aabbtree
import attr
import numpy as np

from .core import SceneElement
from .spectra import Spectrum, SpectrumFactory
from ..util.attrs import (
    attrib_quantity,
    documented, get_doc, parse_docs, validator_has_len,
    validator_has_quantity,
    validator_is_positive,
)
from ..util.factory import BaseFactory
from ..util.units import config_default_units as cdu
from ..util.units import ensure_units, ureg
from ..util.units import kernel_default_units as kdu


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
    """
    An abstract base class defining a base type for all canopies.
    """

    # fmt: off
    id = documented(
        attr.ib(
            default="canopy",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        default="\"canopy\"",
    )

    size = documented(
        attrib_quantity(
            default=None,
            # TODO: add other validators maybe
            validator=validator_has_len(3),
            units_compatible=cdu.generator("length"),
        ),
        doc="Canopy size.\n"
            "\n"
            "Unit-enabled field (default: cdu[length]).",
        type="array-like[float, float, float]",
    )
    # fmt: on

    @abstractmethod
    def bsdfs(self):
        pass

    @abstractmethod
    def shapes(self, ref=False):
        pass

    def kernel_dict(self, ref=True):
        kernel_dict = {}

        if not ref:
            kernel_dict[self.id] = self.shapes(ref=False)[f"shape_{self.id}"]
        else:
            kernel_dict[f"bsdf_{self.id}"] = self.bsdfs()[f"bsdf_{self.id}"]
            kernel_dict[self.id] = self.shapes(ref=True)[f"shape_{self.id}"]

        return kernel_dict


def _inversebeta(mu, nu):
    """Approximates the inverse beta distributionas given in
    :cite:`MyneniRoss1991PhotonVegetationInteraction`
    """
    while True:
        rands = np.random.rand(2)
        s1 = np.power(rands[0], 1.0 / mu)
        s2 = np.power(rands[1], 1.0 / nu)
        s = s1 + s2
        if s <= 1:
            return s1 / s


@ureg.wraps(
    (ureg.m, None),
    (None, ureg.m, ureg.m, ureg.m, ureg.m, None, ureg.m, None, None, None, None),
    False,
)
def _create_leaf_cloud(
    shape_type,
    cube_size=[0, 0, 0] * ureg.m,
    sphere_radius=0 * ureg.m,
    cylinder_radius=0 * ureg.m,
    cylinder_height=0 * ureg.m,
    n_leaves=4000,
    leaf_radius=0.1 * ureg.m,
    mu=1.066,
    nu=1.853,
    tries=1000000,
    avoid_overlap=False,
):
    """
    Creates a set of transform matrices for RAMI canopy scenes.

    Parameter ``shape_type`` (str):
        Specifying the shape of the leaf cloud.
        Can be set to ``cube``, ``sphere`` and ``cylinder`` and each needs
        additional function parameters to be set:

        - A cuboid leaf cloud needs a ``cube_size`` parameter, that holds
          three floating point values. The canopy will extend over the
          [-x/2, x/2], [-y/2, y/2], [0, z] range.
        - A spherical leaf cloud needs a ``sphere_radius`` parameter.
          The sphere will be centered on [0, 0, 0.
        - A cylindrical leaf cloud needs a ``cylinder_radius`` and
          ``cylinder_length`` parameter. One cylinder endcap will be centered
          on [0, 0, 0] and it will extend in the positive z-direction.

    Parameter ``cube_size`` (list[float]):
        Three dimensional extent of the cuboid leaf cloud.
        Must be set when ``shape_type`` is equal to ``cube``

        Default value: 0.0

    Parameter ``sphere_radius`` (float):
        Radius of the spherical leaf cloud.
        Must be set when ``shape_type`` is equal to ``sphere``.

        Default value: 0.0

    Parameter ``cylinder_radius`` (float)
        Radius of the cylindrical leaf cloud.
        Must be set when ``shape_type`` is equal to ``cylinder``.

        Default value: 0.0

    Parameter ``cylinder_height`` (float)
        Height of the cylindrical leaf cloud.
        Must be set when ``shape_type`` is equal to ``cylinder``.

        Default value: 0.0

    Parameter ``n_leaves`` (int):
        Number of leaves to be placed.

        Default value: 4000

    Parameter ``leaf_radius`` (pint.Quantity):
        Radius of the leaves, given in units of length.

        Default value: 0.1

    Parameter ``mu`` (float):
        First parameter of the inversebeta function approximation.

        Default value: 1.066

    Parameter ``nu`` (float):
        Second parameter of the inversebeta function approximation.

        Default value: 1.853

    Parameter ``tries`` (int):
        Number of tries to place a leaf in a leaf cloud with overlap avoidance.

        Default value: 1e6

    Parameter ``avoid_overlap`` (bool):
        For cuboid canopies only. Tries to place the leaves without overlap.

        Default value: False

    Returns → list[ScalarTransform4f]:
        Transformation matrices for the leaves of a canopy.

    Raises → KeyError:
        If the necessary parameters are not present in ``shape_dict``


    """

    positions = np.empty((n_leaves, 3))
    if shape_type == "cube":
        if np.allclose(cube_size, [0, 0, 0]):
            raise ValueError("Parameter cube_size must be set for cuboid leaf cloud.")
        tree = aabbtree.AABBTree()

        if not avoid_overlap:
            for i in range(n_leaves):
                rand = np.random.rand(3)
                positions[i] = [
                    rand[0] * cube_size[0] - cube_size[0] / 2.0,
                    rand[1] * cube_size[1] - cube_size[1] / 2.0,
                    rand[2] * cube_size[2],
                ]
        # try placing the leaves such that they do not overlap by creating
        # axes alingned bounding boxes and checking them for intersection
        else:
            for i in range(n_leaves):
                for j in range(tries):
                    rand = np.random.rand(3)
                    pos_candidate = [
                        rand[0] * cube_size[0] - cube_size[0] / 2.0,
                        rand[1] * cube_size[1] - cube_size[1] / 2.0,
                        rand[2] * cube_size[2],
                    ]
                    aabb = aabbtree.AABB(
                        [
                            (
                                pos_candidate[0] - leaf_radius,
                                pos_candidate[0] + leaf_radius,
                            ),
                            (
                                pos_candidate[1] - leaf_radius,
                                pos_candidate[1] + leaf_radius,
                            ),
                            (
                                pos_candidate[2] - leaf_radius,
                                pos_candidate[2] + leaf_radius,
                            ),
                        ]
                    )
                    if i == 0:
                        positions[i] = pos_candidate
                        tree.add(aabb)
                        break
                    else:
                        if not tree.does_overlap(aabb):
                            positions.append(pos_candidate)
                            tree.add(aabb)
                            break
                else:
                    raise RuntimeError(
                        "unable to place all leaves: "
                        "the specified canopy might be too dense"
                    )
    elif shape_type == "sphere":
        if sphere_radius == 0:
            raise ValueError(
                "Parameter sphere_radius must be set for spherical leaf cloud."
            )
        for i in range(n_leaves):
            rand = np.random.rand(3)
            theta = rand[0] * np.pi
            phi = rand[1] * 2 * np.pi
            r = rand[2] * sphere_radius
            positions[i] = [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ]
    elif shape_type == "cylinder":
        if cylinder_radius == 0 or cylinder_height == 0:
            raise ValueError(
                "Parameters cylinder_height and cylinder_radius"
                "must be set for cylindrical leaf cloud."
            )

        for i in range(n_leaves):
            rand = np.random.rand(3)
            phi = rand[0] * 2 * np.pi
            r = rand[1] * cylinder_radius
            z = rand[2] * cylinder_height
            positions[i] = [r * np.cos(phi), r * np.sin(phi), z]

    # compute leaf normals and create transformation matrices
    orientations = np.empty((n_leaves, 3))
    for i in range(np.shape(orientations)[0]):
        theta = np.rad2deg(_inversebeta(mu, nu))
        phi = np.random.rand() * 360.0

        orientation = [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
        orientations[i] = orientation

    return positions, orientations


@BiosphereFactory.register("homogeneous_discrete_canopy")
@parse_docs
@attr.s
class HomogeneousDiscreteCanopy(Canopy):
    """A generator for the
    `homogenous discrete canopy used in the RAMI benchmark <https://rami-benchmark.jrc.ec.europa.eu/_www/phase/phase_exp.php?strTag=level3&strNext=meas&strPhase=RAMI3&strTagValue=HOM_SOL_DIS>`_.

    This canopy can be instantiated in two ways:

    * The classmethod
      :meth:`~eradiate.scenes.biosphere.HomogeneousDiscreteCanopy.from_parameters`
      takes a set of parameters and will generate the canopy from them. For
      details, please see the documentation of that method.
    * The classmethod
      :meth:`~eradiate.scenes.biosphere.HomogeneousDiscreteCanopy.from_file`
      will read a set of definition files that specify the leaves of the canopy.
      Please refer to this method for details on the file format.
    """

    # fmt: off
    id = documented(
        attr.ib(
            default="homogeneous_discrete_canopy",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(Canopy, "id", "doc"),
        type=get_doc(Canopy, "id", "type"),
        default="homogeneous_discrete_canopy",
    )

    leaf_positions = documented(
        attrib_quantity(
            default=[],
            units_compatible=cdu.generator("length"),
        ),
        doc="Lists all leaf positions as 3-vectors in cartesian coordinates.\n"
            "\n"
            "Unit-enabled field (default units: cdu[length]).",
        type="list[list[float]]",
        default="list()",
    )

    leaf_orientations = documented(
        attr.ib(default=[]),
        doc="Lists all leaf orientations as 3-vectors in cartesian coordinates.",
        type="list[list[float]]",
        default="list()",
    )

    leaf_reflectance = documented(
        attr.ib(
            default=0.5,
            converter=SpectrumFactory.converter("reflectance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validator_has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum of the leaves in the canopy. Must be a "
            "reflectance spectrum (dimensionless).",
        type="float or :class:`~eradiate.scenes.spectra.Spectrum`",
        default="0.5",
    )

    leaf_transmittance = documented(
        attr.ib(
            default=0.5,
            converter=SpectrumFactory.converter("transmittance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validator_has_quantity("transmittance"),
            ],
        ),
        doc="Transmittance spectrum of the leaves in the canopy. Must be a "
            "transmittance spectrum (dimensionless).",
        type="float or :class:`~eradiate.scenes.spectra.Spectrum`",
        default="0.5",
    )

    leaf_radius = documented(
        attrib_quantity(
            default=ureg.Quantity(0.1, ureg.m),
            validator=validator_is_positive,
            units_compatible=cdu.generator("length"),
        ),
        doc="Leaf radius.\n"
            "\n"
            "Unit-enabled field (default unit: cdu[length]).",
        type="float",
        default="0.1 m",
    )
    # fmt: on

    @leaf_positions.validator
    @leaf_orientations.validator
    def _positions_orientations_validator(self, attribute, value):
        if not len(self.leaf_positions) == len(self.leaf_orientations):
            raise ValueError(
                f"While validating {attribute}:"
                f"leaf_positions and leaf_orientations must have"
                f"the same length!"
                f"Got positions: {len(self.leaf_positions)},"
                f"orientations: {len(self.leaf_orientations)}."
            )

    # Hidden parameters, initialised during post-init
    # -- Leaf area index of the created canopy. Used only to display information
    #    about the object.
    _lai = attr.ib(default=None, init=False)
    # -- Number of leaves in the canopy. Used only to display information about
    #    the object.
    _n_leaves = attr.ib(default=None, init=False)

    def __str__(self):
        height = self.size[2]
        # fmt: off
        return (
            f"HomogeneousDiscreteCanopy:\n"
            f"    Canopy height:      {height.magnitude} {height.units}\n"
            f"    LAI:                {self._lai}\n"
            f"    Number of leaves:   {self._n_leaves}\n"
            f"    Leaf reflectance:   {self.leaf_reflectance}\n"
            f"    Leaf transmittance: {self.leaf_transmittance}"
        )
        # fmt:on

    def __attrs_post_init__(self):

        leaf_area = (
            np.pi * self.leaf_radius * self.leaf_radius * len(self.leaf_positions)
        )
        self._lai = leaf_area / (self.size[0] * self.size[1])

        self._n_leaves = len(self.leaf_positions)

    def bsdfs(self):
        # fmt: off
        return {
            f"bsdf_{self.id}":
                {
                    "type":
                        "bilambertian",
                    "reflectance":
                        self.leaf_reflectance.kernel_dict()["spectrum"],
                    "transmittance":
                        self.leaf_transmittance.kernel_dict()["spectrum"],
                },
        }
        # fmt: on

    def shapes(self, ref=True):
        from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f

        kdu_length = kdu.get("length")

        shapes_dict = {f"shape_{self.id}": {"type": "shapegroup",}}

        if ref:
            bsdf = {"type": "ref", "id": f"bsdf_{self.id}"}
        else:
            bsdf = self.bsdfs()[f"bsdf_{self.id}"]

        radius = self.leaf_radius.to(kdu_length).magnitude
        for i in range(len(self.leaf_positions)):
            position_mag = self.leaf_positions[i].to(kdu_length).magnitude
            normal = self.leaf_orientations[i]
            to_world = ScalarTransform4f.look_at(
                origin=position_mag,
                target=position_mag + normal,
                up=np.cross(position_mag, normal),
            ) * ScalarTransform4f.scale(ScalarVector3f(radius, radius, 1))

            shapes_dict[f"shape_{self.id}"][f"leaf_{i}"] = {
                "type": "disk",
                "bsdf": bsdf,
                "to_world": to_world,
            }

        return shapes_dict

    @classmethod
    @ureg.wraps(
        None,
        (
            None,
            ureg.m,
            None,
            None,
            None,
            ureg.m,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        False,
    )
    def from_parameters(
        cls,
        size=[30, 30, 3] * ureg.m,
        lai=3,
        mu=1.066,
        nu=1.853,
        leaf_radius=0.1 * ureg.m,
        leaf_reflectance=0.5,
        leaf_transmittance=0.5,
        n_leaves=4000,
        hdo=1,
        hvr=1,
        seed=1,
        avoid_overlap=False,
    ):
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

        Parameter ``size`` (list[float]):
            Length of the canopy in the three dimensions. A canopy with size
            [x, y, z] will extend over [-x/2, x/2], [-y/2, y/2] and [0, z], centered
            at ``position`` parameter. Default: [30, 30, 3]m.

            If ``size`` is not set, it will be automatically computed from ``hdo``,
            ``hvr`` and ``n_leaves``.

            Unit-enabled field (default units: cdu[length]).

        Parameter ``lai`` (float):
            Leaf area index. Physical range: [0, 10], Default value: 3.

        Parameter ``mu`` (float):
            First parameter for the beta distribution. Default: 1.066.

        Parameter ``nu`` (float):
            Second parameter for the beta distribution. Default value: 1.853.

        Parameter ``leaf_radius`` (float):
            Leaf radius. Physical range: [0, height/2.], Default 0.1m.

            Unit-enabled field (default unit: cdu[length])

        Parameter ``leaf_reflectance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
            Reflectance spectrum of the leaves in the cloud. Must be a reflectance
            spectrum (dimensionless). Default: 0.5.

        Parameter ``leaf_transmittance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
            Transmittance spectrum of the leaves in the cloud. Must be a
            transmittance spectrum (dimensionless). Default: 0.5.

        Parameter ``n_leaves`` (int):
            Total number of leaves to generate. If ``size`` is set, it will override
            this parameter. Default: 4000

        Parameter ``center_position`` (list[float]):
            Three dimensional position of the canopy. Default: [0, 0, 0]m.

            Unit-enabled field (default units: cdu[length])

        Parameter ``hdo`` (float):
            Mean horizontal distance between leaves. If ``size`` is set, it will
            override this parameter. Default: 1.

            Unit-enabled field (default unit: cdu[length])

        Parameter ``hvr`` (float):
            Ratio of mean horizontal leaf distance and vertical canopy extent.
            If ``size`` is set, it will override this parameter. Default: 1.

        Parameter ``seed`` (int):
            Seed for the random number generator. Default: 1.

        Parameter ``avoid_overlap`` (bool):
            If ``True``, the scene element will attempt to place the leaves such
            that they do not overlap. If a leaf cannot be placed without overlap
            after 1e6 tries, it will raise a ``RuntimeError``. Default: False.

            .. warning::

                Depending on the canopy specification instantiation can take
                a very long time if overlap avoidance is active!

            .. note::

               To emulate the behaviour of the raytran leaf cloud generator
               simply try instantiating the leaf cloud several times with
               different ``seed`` values and after a certain amount of
               failures, run it without overlap avoidance.
        """

        np.random.seed(seed)

        # n_leaves or horizontal extent
        if size[0] == 0 or size[1] == 0:
            size[0] = size[1] = np.sqrt(n_leaves * np.pi * leaf_radius ** 2 / lai)
        else:
            n_leaves = int(
                np.floor((size[0] * size[1] * lai) / (np.pi * leaf_radius ** 2))
            )

        # hdo/hvr or vertical extent
        if size[2] == 0:
            size[2] = lai * hdo ** 3 / (np.pi * leaf_radius ** 2 * hvr)
        else:
            hdo = np.power(np.pi * leaf_radius ** 2 * size[2] * hvr / lai, 1 / 3.0)

        positions, orientations = _create_leaf_cloud(
            shape_type="cube",
            cube_size=size,
            n_leaves=n_leaves,
            leaf_radius=leaf_radius,
            mu=mu,
            nu=nu,
            avoid_overlap=avoid_overlap,
        )

        return cls(
            leaf_reflectance=leaf_reflectance,
            leaf_transmittance=leaf_transmittance,
            leaf_radius=leaf_radius,
            leaf_positions=positions,
            leaf_orientations=orientations,
            size=size,
        )

    @classmethod
    def from_rami(
        cls, file_path, leaf_reflectance=0.5, leaf_transmittance=0.5, size=[0, 0, 0]
    ):
        """
        This method allows construction of the Canopy from a text file, specifying
        the individual leaves. The file must specify one leaf per line with the
        following seven parameters separated by one or more spaces:

        - The leaf radius
        - The x, y and z component of the leaf center
        - The x, y and z component of the leaf normal

        All values are given in meters.

        Parameter ``file_path`` (string or PathLike):
            Path to the text file specifying the leaves in the canopy.
            Can be absolute or relative.

        Parameter ``leaf_reflectance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
            Reflectance spectrum of the leaves in the cloud. Must be a reflectance
            spectrum (dimensionless). Default: 0.5.

        Parameter ``leaf_transmittance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
            Transmittance spectrum of the leaves in the cloud. Must be a
            transmittance spectrum (dimensionless). Default: 0.5.

            Unit-enabled field (default units: cdu[length])
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file at {file_path} found.")

        positions_ = []
        orientations_ = []
        with open(os.path.abspath(file_path), "r") as definition_file:
            for i, line in enumerate(definition_file):
                values = line.split()

                if i == 0:
                    radius = ensure_units(float(values[0].strip()), "meter")

                position = [
                    float(values[1].strip()),
                    float(values[2].strip()),
                    float(values[3].strip()),
                ]

                positions_.append(position)
                normal = [
                    float(values[4].strip()),
                    float(values[5].strip()),
                    float(values[6].strip()),
                ]

                orientations_.append(normal)

        positions = np.array(positions_) * ureg.m
        orientations = np.array(orientations_)

        return cls(
            leaf_reflectance=leaf_reflectance,
            leaf_transmittance=leaf_transmittance,
            leaf_radius=radius,
            leaf_positions=positions,
            leaf_orientations=orientations,
            size=size,
        )

    @classmethod
    def from_dict(cls, d):
        """Create from a dictionary. This class method will additionally
        pre-process the passed dictionary to merge any field with an
        associated ``"_units"`` field into a :class:`pint.Quantity` container.

        Since the class can be instantiated by either passing a file that
        specifies the locations and orientations of leaves or by passing
        parameters for a procedural generation of these locations and
        orientations, this method also selects the proper instantiation method.
        If the ``file_path`` parameter is set, instantiation from this file
        will be performed. Otherwise procedural generation of the canopy
        will be chosen.

        Parameter ``d`` (dict):
            Configuration dictionary used for initialisation.

        Returns → wrapped_cls:
            Created object.
        """

        # Pre-process dict: apply units to unit-enabled fields
        d_copy = copy(d)

        for field in cls._fields_supporting_units():
            # Fetch user-specified unit if any
            try:
                field_units = d_copy.pop(f"{field}_units")
            except KeyError:
                # If no unit is specified, don't attempt conversion and let the
                # constructor take care of it
                continue

            # If a unit is found, try to apply it
            # Bonus: if a unit field *and* a quantity were found, we convert the
            # quantity to the unit
            field_value = d_copy[field]
            d_copy[field] = ensure_units(field_value, field_units, convert=True)

        if "file_path" in d_copy:
            return cls.from_rami(**d_copy)
        else:
            return cls.from_parameters(**d_copy)

    def kernel_dict(self, ref=True):
        kernel_dict = {}

        if not ref:
            kernel_dict[self.id] = self.shapes(ref=False)[f"shape_{self.id}"]
        else:
            kernel_dict[f"bsdf_{self.id}"] = self.bsdfs()[f"bsdf_{self.id}"]
            kernel_dict[self.id] = self.shapes(ref=True)[f"shape_{self.id}"]

        return kernel_dict
