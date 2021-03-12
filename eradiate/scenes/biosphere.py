"""Biosphere-related scene generation facilities.

.. admonition:: Registered factory members [:class:`BiosphereFactory`]
    :class: hint

    .. factorytable::
       :factory: BiosphereFactory
"""

__all__ = [
    "BiosphereFactory",
    "HomogeneousDiscreteCanopy",
    "FloatingSpheresCanopy",
    "RealZoomInCanopy",
]

import os
from abc import ABC, abstractmethod

import aabbtree
import attr
import numpy as np
import pinttr

from .core import SceneElement
from .spectra import Spectrum, SpectrumFactory
from .._attrs import documented, get_doc, parse_docs
from .._factory import BaseFactory
from .._units import unit_context_config as ucc
from .._units import unit_context_kernel as uck
from .._units import unit_registry as ureg
from ..validators import has_len, has_quantity, is_positive


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


@parse_docs
@attr.s
class LeafCloud:

    """
    A container class for leaf clouds in abstract discrete canopies.
    Holds the specification of leaves inside the crown.

    A leaf cloud can be specified in two ways:

    - The classmethod :meth:`~eradiate.scenes.biosphere.LeafCloud.from_parameters`
      takes a set of parameters and will generate the leaves from them. For details,
      please see the documentation of that method.
    - The classmethod :meth:`~eradiate.scenes.biosphere.LeafCloud.from_file`
      will read a set of definition files that specify the leaves.
      Please refer to this method for details on the file format.
    """

    # fmt: off
    id = documented(
        attr.ib(
            default="leaf_cloud",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="Identifier for the object",
        type="str",
        default="leaf_cloud"
    )

    leaf_positions = documented(
        pinttr.ib(
            default=[],
            units=uck.deferred("length"),
        ),
        doc="Lists all leaf positions as 3-vectors in cartesian coordinates.\n"
            "\n"
            "Unit-enabled field (default: cdu[angle]).",
        type="list[list[float]]",
        default="[]"
    )

    leaf_orientations = documented(
        attr.ib(
            default=[]
        ),
        doc="Lists all leaf orientations as 3-vectors in cartesian coordinates.",
        type="list[list[float]]",
        default="[]"
    )

    @leaf_positions.validator
    @leaf_orientations.validator
    def _positions_orientations_validator(self, attribute, value):
        if not len(self.leaf_positions) == len(self.leaf_orientations):
            raise ValueError(
                f"While validating {attribute}: "
                f"leaf_positions and leaf_orientations must have the same length! "
                f"Got positions: {len(self.leaf_positions)}, "
                f"orientations: {len(self.leaf_orientations)}."
            )

    leaf_reflectance = documented(
        attr.ib(
            default=0.5,
            converter=SpectrumFactory.converter("reflectance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum of the leaves in the cloud. "
            "Must be a reflectance spectrum (dimensionless).",
        type=":class:`.Spectrum`",
        default="0.5"
    )

    leaf_transmittance = documented(
        attr.ib(
            default=0.5,
            converter=SpectrumFactory.converter("transmittance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                has_quantity("transmittance"),
            ],
        ),
        doc="Transmittance spectrum of the leaves in the cloud. "
            "Must be a transmittance spectrum (dimensionless).",
        type=":class:`.Spectrum`",
        default="0.5"
    )

    leaf_radius = documented(
        pinttr.ib(
            default=ureg.Quantity(0.1, ureg.m),
            validator=[pinttr.validators.has_compatible_units, is_positive],
            units=uck.deferred("length"),
        ),
        doc="Radius of the leaves",
        type="float",
        default="0.1"
    )
    # fmt: on

    def n_leaves(self):
        return len(self.leaf_positions)

    @classmethod
    def from_file(
        cls, fname_leaves, leaf_transmittance, leaf_reflectance, id="leaf_cloud"
    ):
        """
        This method allows construction of the LeafCloud from a text file,
        specifying the individual leaves. The file must specify one leaf per
        line with the following seven parameters separated by one or more spaces:

        - The leaf radius
        - The x, y and z component of the leaf center
        - The x, y and z component of the leaf normal

        All values are assumed to be given in meters.

        Parameter ``fname_leaves`` (string or PathLike):
            Path to the text file specifying the leaves in the canopy.
            Can be absolute or relative.

        Parameter ``leaf_reflectance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
            Reflectance spectrum of the leaves in the cloud. Must be a reflectance
            spectrum (dimensionless). Default: 0.5.

        Parameter ``leaf_transmittance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
            Transmittance spectrum of the leaves in the cloud. Must be a
            transmittance spectrum (dimensionless). Default: 0.5.

            Unit-enabled field (default units: ucc[length])
        """

        if not os.path.exists(fname_leaves):
            raise FileNotFoundError(f"No file at {fname_leaves} found.")

        positions_ = []
        orientations_ = []
        with open(os.path.abspath(fname_leaves), "r") as definition_file:
            for i, line in enumerate(definition_file):
                values = line.split()

                if i == 0:
                    radius = float(values[0].strip()) * ureg.m

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
            id=id,
            leaf_positions=positions,
            leaf_orientations=orientations,
            leaf_reflectance=leaf_reflectance,
            leaf_transmittance=leaf_transmittance,
            leaf_radius=radius,
        )

    @classmethod
    def from_parameters(
        cls,
        id="leaf_cloud",
        leaf_transmittance=0.5,
        leaf_reflectance=0.5,
        shape_type="cube",
        size=[30, 30, 3] * ureg.m,
        sphere_radius=0 * ureg.m,
        cylinder_radius=0 * ureg.m,
        cylinder_height=0 * ureg.m,
        lai=3,
        mu=1.066,
        nu=1.853,
        leaf_radius=0.1 * ureg.m,
        n_leaves=4000,
        hdo=1,
        hvr=1,
        seed=1,
        avoid_overlap=False,
    ):
        """
        This method allows the creation of a HomogeneousDiscreteCanopy
        from statistical parameters.
        It uses a re-implementation of the generator in raytran. :func:`~eradiate.scenes.biosphere._create_leaf_cloud`
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

        Parameter ``leaf_reflectance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
            Reflectance spectrum of the leaves in the cloud. Must be a reflectance
            spectrum (dimensionless). Default: 0.5.

        Parameter ``leaf_transmittance`` (float or :class:`~eradiate.scenes.spectra.Spectrum`):
            Transmittance spectrum of the leaves in the cloud. Must be a
            transmittance spectrum (dimensionless). Default: 0.5.

        Parameter ``size`` (list[float]):
            Length of the canopy in the three dimensions. A canopy with size
            [x, y, z] will extend over [-x/2, x/2], [-y/2, y/2] and [0, z], centered
            at ``position`` parameter. Default: [30, 30, 3]m.

            If ``size`` is not set, it will be automatically computed from ``hdo``,
            ``hvr`` and ``n_leaves``.

            Unit-enabled field (default units: ucc[length]).
            Unit-enabled field (default units: ucc[length]).

        Parameter ``lai`` (float):
            Leaf area index. Physical range: [0, 10], Default value: 3.

        Parameter ``mu`` (float):
            First parameter for the beta distribution. Default: 1.066.

        Parameter ``nu`` (float):
            Second parameter for the beta distribution. Default value: 1.853.

        Parameter ``leaf_radius`` (float):
            Leaf radius. Physical range: [0, height/2.], Default 0.1m.

            Unit-enabled field (default unit: ucc[length])

        Parameter ``n_leaves`` (int):
            Total number of leaves to generate. If ``size`` is set, it will override
            this parameter. Default: 4000

        Parameter ``hdo`` (float):
            Mean horizontal distance between leaves. If ``size`` is set, it will
            override this parameter. Default: 1.

            Unit-enabled field (default unit: ucc[length])

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

        leaf_positions, leaf_orientations = _create_leaf_cloud(
            shape_type=shape_type,
            cube_size=size,
            sphere_radius=sphere_radius,
            cylinder_radius=cylinder_radius,
            cylinder_height=cylinder_height,
            n_leaves=n_leaves,
            leaf_radius=leaf_radius,
            mu=mu,
            nu=nu,
            seed=seed,
            avoid_overlap=avoid_overlap,
        )

        return cls(
            id=id,
            leaf_positions=leaf_positions,
            leaf_orientations=leaf_orientations,
            leaf_reflectance=leaf_reflectance,
            leaf_transmittance=leaf_transmittance,
            leaf_radius=leaf_radius,
        )

    def shapes(self, ref=False):
        from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f

        kernel_length = uck.get("length")
        shapes_dict = {self.id: {"type": "shapegroup"}}

        if ref:
            bsdf = {"type": "ref", "id": f"bsdf_{self.id}"}
        else:
            bsdf = self.bsdfs()[f"bsdf_{self.id}"]

        radius = self.leaf_radius.to(kernel_length).magnitude
        for i in range(len(self.leaf_positions)):
            position_mag = self.leaf_positions[i].m_as(kernel_length)
            normal = self.leaf_orientations[i]
            to_world = (
                ScalarTransform4f.look_at(
                    origin=position_mag,
                    target=position_mag + normal,
                    up=np.cross(position_mag, normal),
                )
                * ScalarTransform4f.scale(radius)
            )

            shapes_dict[f"{self.id}"][f"leaf_{i}"] = {
                "type": "disk",
                "bsdf": bsdf,
                "to_world": to_world,
            }

        return shapes_dict

    def bsdfs(self):
        return {
            f"bsdf_{self.id}": {
                "type": "bilambertian",
                "reflectance": self.leaf_reflectance.kernel_dict()["spectrum"],
                "transmittance": self.leaf_transmittance.kernel_dict()["spectrum"],
            },
        }

    def kernel_dict(self, ref=True):
        kernel_dict = {}

        if not ref:
            kernel_dict[self.id] = self.shapes(ref=False)[f"{self.id}"]
        else:
            kernel_dict[f"bsdf_{self.id}"] = self.bsdfs()[f"bsdf_{self.id}"]
            kernel_dict[self.id] = self.shapes(ref=True)[f"{self.id}"]

        return kernel_dict


@parse_docs
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
        pinttr.ib(
            default=None,
            # TODO: add other validators maybe
            validator=[pinttr.validators.has_compatible_units, has_len(3)],
            units=ucc.deferred("length"),
        ),
        doc="Canopy size.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="array-like[float, float, float]",
    )
    # fmt: on

    @abstractmethod
    def bsdfs(self):
        pass

    @abstractmethod
    def shapes(self, ref=False):
        pass

    @abstractmethod
    def kernel_dict(self, ref=True):
        pass


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
    (None, ureg.m, ureg.m, ureg.m, ureg.m, None, ureg.m, None, None, None, None, None),
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
    seed=1,
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

    Parameter ``seed`` (int):
        Seed for the random number generator. Default: 1.

    Parameter ``avoid_overlap`` (bool):
        For cuboid canopies only. Tries to place the leaves without overlap.

        Default value: False

    Returns → list[list[float]], list[list[float]]:
        Lists holding the positions and orientations of the leaves in the
        tree crown.

    Raises → KeyError:
        If the necessary parameters are not present in ``shape_dict``


    """

    np.random.seed(seed)

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
                "Parameters cylinder_height and cylinder_radius "
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


@parse_docs
@attr.s
class DiscreteCanopy(Canopy):
    """
    Base class for abstract discrete canopies, *i.e.* consisting of disk shaped
    leaf clouds.
    """

    # fmt: off
    leaf_cloud_specs = documented(
        attr.ib(
            default=[]
        ),
        doc="Tuples containing the leaf cloud objects and a list of points, "
        "specifying the locations of the leaf cloud inside the canopy.",
        type="list(tuple(:class:`.LeafCloud`, list(list())))",
        default="[]",
    )
    # fmt: on

    def bsdfs(self):
        returndict = {}
        for (leaf_cloud, leaf_cloud_id) in self.leaf_cloud_specs:
            returndict = {**returndict, **leaf_cloud.bsdfs()}
        return returndict

    def shapes(self, ref=False):
        returndict = {}
        for (leaf_cloud, leaf_cloud_id) in self.leaf_cloud_specs:
            returndict = {**returndict, **leaf_cloud.shapes(ref)}
        return returndict

    def kernel_dict(self, ref=True):
        leaf_cloud_ids = [spec[0].id for spec in self.leaf_cloud_specs]
        kernel_dict = {}

        for leaf_cloud_id in leaf_cloud_ids:
            if not ref:
                kernel_dict[leaf_cloud_id] = self.shapes(ref=False)[leaf_cloud_id]
            else:
                kernel_dict[f"bsdf_{leaf_cloud_id}"] = self.bsdfs()[
                    f"bsdf_{leaf_cloud_id}"
                ]
                kernel_dict[leaf_cloud_id] = self.shapes(ref=True)[leaf_cloud_id]

        return kernel_dict


@BiosphereFactory.register("homogeneous_discrete_canopy")
@parse_docs
@attr.s
class HomogeneousDiscreteCanopy(DiscreteCanopy):
    """Generates the `homogenous discrete canopy <https://rami-benchmark.jrc.ec.europa.eu/_www/phase/phase_exp.php?strTag=level3&strNext=meas&strPhase=RAMI3&strTagValue=HOM_SOL_DIS>`_ in the RAMI benchmark.

    The leaf cloud in this canopy can be created in two ways. Please refer to
    :class:`~eradiate.scenes.biosphere.LeafCloud` for the details.

    This canopy is composed of a single leaf cloud shaped as a cuboid.
    """

    id = documented(
        attr.ib(
            default="homogeneous_discrete_canopy",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(Canopy, "id", "doc"),
        type=get_doc(Canopy, "id", "type"),
        default="homogeneous_discrete_canopy",
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
        kernel_length = uck.get("length")

        leaf_cloud = self.leaf_cloud_specs[0][0]

        self._n_leaves = leaf_cloud.n_leaves()
        leaf_area = (
            np.pi
            * np.square(leaf_cloud.leaf_radius.m_as(kernel_length))
            * leaf_cloud.n_leaves()
            * kernel_length
        )

        self._lai = leaf_area / (self.size[0] * self.size[1])

    @classmethod
    def from_dict(cls, d):
        """Create from a dictionary. This class method will
        pre-process the passed dictionary to merge any field with an
        associated ``"_units"`` field into a :class:`pint.Quantity` container.

        A dictionary must follow this form:

        .. code-block:: python

           {
               "type": "real_zoom_in_canopy",
               "leaf_cloud": {
                   <LeafCloud parameters>
               },
               "size": <3-vector specifying the size of the canopy>,
           }

        For the parametrization of the leaf clouds, see
        :func:`~eradiate.scenes.biosphere.LeafCloud.from_file` and :func:`~eradiate.scenes.biosphere.LeafCloud.from_parameters`.

         Note that this method requires the parametric leaf cloud to be specified
         as a cuboid. The `shape_type` parameter must be set to `cube`.

        Parameter ``d`` (dict):
            Configuration dictionary used for initialisation.

        Returns → wrapped_cls:
            Created object.
        """

        # Pre-process dict: apply units to unit-enabled fields
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        size = d_copy.pop("size")
        leaf_cloud_spec = d_copy["leaf_cloud"]
        if "fname_leaves" in leaf_cloud_spec:
            leaf_cloud = LeafCloud.from_file(**leaf_cloud_spec)
        else:
            if leaf_cloud_spec["shape_type"] != "cube":
                raise ValueError(
                    "The HomogeneousDiscreteCanopy must be used"
                    "with a cuboid leaf cloud."
                )
            leaf_cloud = LeafCloud.from_parameters(**leaf_cloud_spec)

        return cls(leaf_cloud_specs=[(leaf_cloud, [[0, 0, 0]])], size=size)


@BiosphereFactory.register("floating_spheres_canopy")
@parse_docs
@attr.s
class FloatingSpheresCanopy(DiscreteCanopy):
    """Generates the canopy for the `floating sphere scenario <https://rami-benchmark.jrc.ec.europa.eu/HTML/RAMI3/EXPERIMENTS3/HETEROGENEOUS/FLOATING_SPHERES/SOLAR_DOMAIN/DISCRETE/DISCRETE.php>`_ in the RAMI benchmark

    This canopy is composed of a set of leaf clouds shaped as spheres."""

    id = documented(
        attr.ib(
            default="floating_spheres_canopy",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(Canopy, "id", "doc"),
        type=get_doc(Canopy, "id", "type"),
        default="floating_spheres_canopy",
    )

    @classmethod
    def from_dict(cls, d):
        """Create from a dictionary.

        Dictionary format:

        .. code-block:: python

           {
               "type": "real_zoom_in_canopy",
               "leaf_cloud": {
                   "fname_positions": <path to a file>,
                   <LeafCloud parameters>
               },
               "size": <3-vector specifying the size of the canopy>,
           }

        .. note::
           This class method will pre-process the passed dictionary to merge any field with an
           associated ``"_units"`` field into a :class:`pint.Quantity` container.

        Sphere positions are specified in a separate text file, holding the x,y and z
        coordinates for each sphere per line, separated by whitespace.
        The `fname_positions` field holds the path to this file.

        For the parametrization of the leaf clouds, see
        :func:`~.LeafCloud.from_file` and
        :func:`~.LeafCloud.from_parameters`.

        If `fname_leaves` is present in the leaf cloud specification, the former method
        will be used, otherwise the latter.

        Note that parametric instantiation requires the leaf cloud to be specified
        as a sphere. The `shape_type` parameter must be set to `sphere`.

        Also note that in the case of instantiating the leaf cloud from a text file,
        using the :func:`~eradiate.scenes.biosphere.LeafCloud.from_file` method,
        two files must be provided. One file to specify the placement of the leaves
        inside the leaf cloud and one file to position the leaf cloud instances in the
        scene.

        Parameter ``d`` (dict):
            Configuration dictionary used for initialisation.

        Returns → wrapped_cls:
            Created object.

        Raises → ValueError:
            If the parametric leaf cloud is not specified as `sphere`.
        """
        # Pre-process dict: apply units to unit-enabled fields
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # parse sphere positions
        leaf_cloud_spec = d_copy["leaf_cloud"]
        sphere_positions_file = leaf_cloud_spec.pop("fname_positions")
        sphere_positions = []
        with open(sphere_positions_file, "r") as spf:
            for line in spf:
                coords = line.split()
                if len(coords) != 3:
                    raise ValueError(
                        f"Leaf positions must have three coordinates."
                        f"Found: {len(coords)}!"
                    )
                sphere_positions.append(
                    [float(coord.strip()) for coord in line.split()]
                )
        if "fname_leaves" in leaf_cloud_spec:
            leaf_cloud = LeafCloud.from_file(**leaf_cloud_spec)
        else:
            if leaf_cloud_spec["shape_type"] != "sphere":
                raise ValueError(
                    "The FloatingSphereCanopy must be used"
                    "with a spherical leaf cloud."
                )
            leaf_cloud = LeafCloud.from_parameters(**leaf_cloud_spec)

        size = d_copy["size"]

        return cls(leaf_cloud_specs=[(leaf_cloud, sphere_positions)], size=size)


@BiosphereFactory.register("real_zoom_in_canopy")
@parse_docs
@attr.s
class RealZoomInCanopy(DiscreteCanopy):
    """Generates the canopy for the `real zoom in scenario <https://rami-benchmark.jrc.ec.europa.eu/HTML/RAMI3/EXPERIMENTS3/HETEROGENEOUS/REAL_ZOOM-IN/REAL_ZOOM-IN.php>`_ in the RAMI benchmark.

    This canopy is composed of a set of leaf clouds shaped as spheres."""

    id = documented(
        attr.ib(
            default="real_zoom_in_canopy",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc=get_doc(Canopy, "id", "doc"),
        type=get_doc(Canopy, "id", "type"),
        default="real_zoom_in_canopy",
    )

    @classmethod
    def from_dict(cls, d):
        """Create from a dictionary.

        Dictionary format:

                .. code-block:: python

           {
               "type": "real_zoom_in_canopy",
               "spherical_leaf_cloud": {
                   "fname_positions": <path to a file>,
                   <LeafCloud parameters>
               },
               "cylindrical_leaf_cloud": {
                   "fname_positions": <path to a file>,
                   <LeafCloud parameters>
               },
               "size": <3-vector specifying the size of the canopy>,
           }

        .. note:: This class method will pre-process the passed dictionary
           to merge any field with an associated ``"_units"`` field into a
           :class:`pint.Quantity` container.

        Sphere and cylinder positions are specified in a text file, holding
        the x,y and z coordinates for each sphere per line, separated by whitespace.
        The `fname_sphere_positions` field holds the path to this file.

        For the parametrization of the leaf clouds, see
        :func:`~.LeafCloud.from_file` and
        :func:`~.LeafCloud.from_parameters`.

        If `fname_leaves` is present in the leaf cloud specification, the former method
        will be used, otherwise the latter.

        Note that parametric instantiation requires the ``spherical_leaf_cloud`` to
        be specified with the ``shape_type`` parameter set to ``sphere`` and the
        ``cylindrical_leaf_cloud`` with ``cylinder``.

        Also note that in the case of instantiating the leaf cloud from a text file,
        using the :func:`~.LeafCloud.from_file` method,
        two files must be provided for each leaf cloud. One file to specify the
        placement of the leaves inside the leaf cloud and one file to position
        the leaf cloud instances in the scene.

        Parameter ``d`` (dict):
            Configuration dictionary used for initialisation.

        Returns → wrapped_cls:
            Created object.

        Raises → ValueError:
            If the parametric leaf clouds are not specified as `sphere` and
            `cylinder` respectively.

        """
        # Pre-process dict: apply units to unit-enabled fields
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # parse sphere specifications
        sphere_spec = d_copy["spherical_leaf_cloud"]
        sphere_positions_file = sphere_spec.pop("fname_positions")
        sphere_positions = []
        with open(sphere_positions_file, "r") as spf:
            for line in spf:
                coords = line.split()
                if len(coords) != 3:
                    raise ValueError(
                        f"Leaf positions must have three coordinates. "
                        f"Found: {len(coords)}!"
                    )
                sphere_positions.append(
                    [float(coord.strip()) for coord in line.split()]
                )
        if "fname_leaves" in sphere_spec:
            spherical_leaf_cloud = LeafCloud.from_file(**sphere_spec)
        else:
            if sphere_spec["shape_type"] != "sphere":
                raise ValueError(
                    "The spherical leaf clouds must be used"
                    "with a spherical leaf cloud."
                )
            spherical_leaf_cloud = LeafCloud.from_parameters(**sphere_spec)

        # parse cylinder specifications
        cylinder_spec = d_copy["cylindrical_leaf_cloud"]
        cylinder_positions_file = cylinder_spec.pop("fname_positions")
        cylinder_positions = []
        with open(cylinder_positions_file, "r") as spf:
            for line in spf:
                coords = line.split()
                if len(coords) != 3:
                    raise ValueError(
                        f"Leaf positions must have three coordinates."
                        f"Found: {len(coords)}!"
                    )
                cylinder_positions.append(
                    [float(coord.strip()) for coord in line.split()]
                )
        if "fname_leaves" in cylinder_spec:
            cylindrical_leaf_cloud = LeafCloud.from_file(**cylinder_spec)
        else:
            if cylinder_spec["shape_type"] != "cylinder":
                raise ValueError(
                    "The cylindrical leaf clouds must be used"
                    "with a cylindrical leaf cloud."
                )
            cylindrical_leaf_cloud = LeafCloud.from_parameters(**cylinder_spec)

        size = d_copy["size"]

        return cls(
            leaf_cloud_specs=[
                (spherical_leaf_cloud, sphere_positions),
                (cylindrical_leaf_cloud, cylinder_positions),
            ],
            size=size,
        )
