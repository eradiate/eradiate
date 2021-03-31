import itertools
import os
from copy import deepcopy

import aabbtree
import attr
import numpy as np
import pinttr

from ._core import BiosphereFactory, Canopy
from ..core import SceneElement
from ..spectra import Spectrum, SpectrumFactory
from ... import validators
from ..._attrs import documented, get_doc, parse_docs
from ..._units import unit_context_config as ucc
from ..._units import unit_context_kernel as uck
from ..._units import unit_registry as ureg


def _inversebeta(mu, nu, rng):
    """Approximates the inverse beta distribution as given in
    :cite:`Ross1991MonteCarloMethods` (appendix 1).
    """
    while True:
        rands = rng.random(2)
        s1 = np.power(rands[0], 1.0 / mu)
        s2 = np.power(rands[1], 1.0 / nu)
        s = s1 + s2
        if s <= 1:
            return s1 / s


@ureg.wraps(ureg.m, (None, ureg.m, ureg.m, None))
def _leaf_cloud_positions_cuboid(n_leaves, l_horizontal, l_vertical, rng):
    """Compute leaf positions for a cuboid-shaped leaf cloud (square footprint)."""
    positions = np.empty((n_leaves, 3))

    for i in range(n_leaves):
        rand = rng.random(3)
        positions[i, :] = [
            rand[0] * l_horizontal - 0.5 * l_horizontal,
            rand[1] * l_horizontal - 0.5 * l_horizontal,
            rand[2] * l_vertical,
        ]

    return positions


@ureg.wraps(ureg.m, (None, ureg.m, ureg.m, ureg.m, None, None))
def _leaf_cloud_positions_cuboid_avoid_overlap(
    n_leaves, l_horizontal, l_vertical, leaf_radius, n_attempts, rng
):
    """Compute leaf positions for a cuboid-shaped leaf cloud (square footprint).
    This function also performs conservative collision checks to avoid leaf
    overlapping.
    """
    n_attempts = int(n_attempts)  # For safety, ensure conversion to int

    # try placing the leaves such that they do not overlap by creating
    # axis-aligned bounding boxes and checking them for intersection
    positions = np.empty((n_leaves, 3))
    tree = aabbtree.AABBTree()

    for i in range(n_leaves):
        for j in range(n_attempts):
            rand = rng.random(3)
            pos_candidate = [
                rand[0] * l_horizontal - 0.5 * l_horizontal,
                rand[1] * l_horizontal - 0.5 * l_horizontal,
                rand[2] * l_vertical,
            ]
            aabb = aabbtree.AABB(
                [
                    (pos_candidate[0] - leaf_radius, pos_candidate[0] + leaf_radius),
                    (pos_candidate[1] - leaf_radius, pos_candidate[1] + leaf_radius),
                    (pos_candidate[2] - leaf_radius, pos_candidate[2] + leaf_radius),
                ]
            )
            if i == 0:
                positions[i, :] = pos_candidate
                tree.add(aabb)
                break
            else:
                if not tree.does_overlap(aabb):
                    positions[i, :] = pos_candidate
                    tree.add(aabb)
                    break
        else:
            raise RuntimeError(
                "unable to place all leaves: the specified canopy might be too dense"
            )

    return positions


@ureg.wraps(ureg.m, (None, ureg.m, None))
def _leaf_cloud_positions_sphere(n_leaves, radius, rng):
    """Compute leaf positions for a sphere-shaped leaf cloud."""

    radius = float(radius)

    positions = np.empty((n_leaves, 3))

    for i in range(n_leaves):
        rand = rng.random(3)
        theta = rand[0] * np.pi
        phi = rand[1] * 2 * np.pi
        r = rand[2] * radius
        positions[i, :] = [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]

    return positions


@ureg.wraps(ureg.m, (None, ureg.m, ureg.m, None))
def _leaf_cloud_positions_cylinder(n_leaves, radius, l_vertical, rng):
    """Compute leaf positions for a cylinder-shaped leaf cloud (vertical
    orientation.)"""

    positions = np.empty((n_leaves, 3))

    for i in range(n_leaves):
        rand = rng.random(3)
        phi = rand[0] * 2 * np.pi
        r = rand[1] * radius
        z = rand[2] * l_vertical
        positions[i, :] = [r * np.cos(phi), r * np.sin(phi), z]

    return positions


@ureg.wraps(None, (None, None, None, None))
def _leaf_cloud_orientations(n_leaves, mu, nu, rng):
    """Compute leaf orientations."""
    orientations = np.empty((n_leaves, 3))
    for i in range(np.shape(orientations)[0]):
        theta = np.rad2deg(_inversebeta(mu, nu, rng))
        phi = rng.random() * 360.0

        orientations[i, :] = [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]

    return orientations


@ureg.wraps(ureg.m, (None, ureg.m))
def _leaf_cloud_radii(n_leaves, leaf_radius):
    """Compute leaf radii."""
    return np.full((n_leaves,), leaf_radius)


@parse_docs
@attr.s
class LeafCloudParams:
    """Base class to implement advanced parameter checking for :class:`LeafCloud`
    generators."""

    _id = documented(
        attr.ib(default="leaf_cloud"),
        doc="Leaf cloud identifier.",
        type="str",
        default='"leaf_cloud"',
    )

    _leaf_reflectance = documented(
        attr.ib(default=0.5), doc="Leaf reflectance.", type="float", default="0.5"
    )

    _leaf_transmittance = documented(
        attr.ib(default=0.5), doc="Leaf transmittance.", type="float", default="0.5"
    )

    _mu = documented(
        attr.ib(default=1.066),
        doc="First parameter of the inverse beta distribution approximation used "
        "to generate leaf orientations.",
        type="float",
        default="1.066",
    )

    _nu = documented(
        attr.ib(default=1.853),
        doc="Second parameter of the inverse beta distribution approximation used "
        "to generate leaf orientations.",
        type="float",
        default="1.853",
    )

    _n_leaves = documented(attr.ib(default=None), doc="Number of leaves.", type="int")

    _leaf_radius = documented(
        pinttr.ib(default=None, units=ucc.deferred("length")),
        doc="Leaf radius.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
    )

    def update(self):
        try:
            for field in [x.name.lstrip("_") for x in self.__attrs_attrs__]:
                self.__getattribute__(field)
        except Exception as e:
            raise Exception(
                f"cannot compute field '{field}', parameter set is likely under-constrained"
            ) from e

    def __attrs_post_init__(self):
        self.update()

    @property
    def id(self):
        return self._id

    @property
    def leaf_reflectance(self):
        return self._leaf_reflectance

    @property
    def leaf_transmittance(self):
        return self._leaf_transmittance

    @property
    def nu(self):
        return self._nu

    @property
    def mu(self):
        return self._mu

    @property
    def n_leaves(self):
        return self._n_leaves

    @property
    def leaf_radius(self):
        return self._leaf_radius


@parse_docs
@attr.s
class CuboidLeafCloudParams(LeafCloudParams):
    """
    Advanced parameter checking class for the cuboid :class:`.LeafCloud`
    generator. Some of the parameters can be inferred from each other.

    Parameters defined below can be used (without leading underscore) as
    keyword arguments to the :meth:`.LeafCloud.cuboid` class method
    constructor. Parameters without defaults are connected by a dependency
    graph used to compute required parameters (outlined in the figure below).

    .. warning:: In case of over-specification, no consistency check is
       performed.

    .. admonition:: Examples

       The following parameter sets are valid:

       * ``n_leaves``, ``leaf_radius``, ``l_horizontal``, ``l_vertical``;
       * ``lai``, ``leaf_radius``, ``l_horizontal``, ``l_vertical``;
       * ``lai``, ``leaf_radius``, ``l_horizontal``, ``hdo``, ``hvr``;
       * and more!

    .. only:: latex

       .. figure:: ../../../fig/cuboid_leaf_cloud_params.png

    .. only:: not latex

       .. figure:: ../../../fig/cuboid_leaf_cloud_params.svg

    .. seealso:: :meth:`.LeafCloud.cuboid`
    """

    _l_horizontal = documented(
        pinttr.ib(default=None, units=ucc.deferred("length")),
        doc="Leaf cloud horizontal extent. Suggested default: 30 m.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
    )

    _l_vertical = documented(
        pinttr.ib(default=None, units=ucc.deferred("length")),
        doc="Leaf cloud vertical extent. Suggested default: 3 m.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
    )

    _lai = documented(
        pinttr.ib(default=None, units=ureg.dimensionless),
        doc="Leaf cloud leaf area index (LAI). Physical range: [0, 10]; "
        "suggested default: 3.\n"
        "\n"
        "Unit-enabled field (default: ucc[dimensionless]).",
        type="float",
    )

    _hdo = documented(
        pinttr.ib(default=None, units=ucc.deferred("length")),
        doc="Mean horizontal distance between leaves.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
    )

    _hvr = documented(
        pinttr.ib(default=None),
        doc="Ratio of mean horizontal leaf distance and vertical leaf cloud extent. "
        "Suggested default: 0.1.",
        type="float",
    )

    @property
    def n_leaves(self):
        if self._n_leaves is None:
            self._n_leaves = int(
                self.lai * (self.l_horizontal / self.leaf_radius) ** 2 / np.pi
            )
        return self._n_leaves

    @property
    def lai(self):
        if self._lai is None:
            self._lai = (
                np.pi * (self.leaf_radius / self.l_horizontal) ** 2 * self.n_leaves
            )
        return self._lai

    @property
    def leaf_radius(self):
        if self._leaf_radius is None:
            self._leaf_radius = (
                np.sqrt(self.lai / (self.n_leaves * np.pi)) * self.l_horizontal
            )
        return self._leaf_radius

    @property
    def l_horizontal(self):
        if self._l_horizontal is None:
            self._l_horizontal = (
                np.pi * self.leaf_radius ** 2 * self.n_leaves / self.lai
            )
        return self._l_horizontal

    @property
    def l_vertical(self):
        if self._l_vertical is None:
            self._l_vertical = (
                self.lai * self.hdo ** 3 / (np.pi * self.leaf_radius ** 2 * self.hvr)
            )
        return self._l_vertical

    @property
    def hdo(self):
        return self._hdo

    @property
    def hvr(self):
        return self._hvr

    def __str__(self):
        result = []

        for field in [
            "id",
            "lai",
            "leaf_radius",
            "l_horizontal",
            "l_vertical",
            "n_leaves",
            "leaf_reflectance",
            "leaf_transmittance",
        ]:
            value = self.__getattribute__(field)
            result.append(f"{field}={value.__repr__()}")

        return f"CuboidLeafCloudParams({', '.join(result)})"


@parse_docs
@attr.s
class SphereLeafCloudParams(LeafCloudParams):
    """
    Advanced parameter checking class for the sphere :class:`.LeafCloud`
    generator.

    .. seealso:: :meth:`.LeafCloud.sphere`
    """

    _radius = documented(
        pinttr.ib(default=1.0 * ureg.m, units=ucc.deferred("length")),
        doc="Leaf cloud radius.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
        default="1 m",
    )

    @property
    def radius(self):
        return self._radius


@parse_docs
@attr.s
class CylinderLeafCloudParams(LeafCloudParams):
    """
    Advanced parameter checking class for the cylinder :class:`.LeafCloud`
    generator.

    .. seealso:: :meth:`.LeafCloud.cylinder`
    """

    _radius = documented(
        pinttr.ib(default=1.0 * ureg.m, units=ucc.deferred("length")),
        doc="Leaf cloud radius.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
        default="1 m",
    )

    _l_vertical = documented(
        pinttr.ib(default=1.0 * ureg.m, units=ucc.deferred("length")),
        doc="Leaf cloud vertical extent.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
        default="1 m",
    )

    @property
    def radius(self):
        return self._radius

    @property
    def l_vertical(self):
        return self._l_vertical


@parse_docs
@attr.s
class LeafCloud(SceneElement):
    """
    A container class for leaf clouds in abstract discrete canopies.
    Holds parameters completely characterising the leaf cloud's leaves.

    In practice, this class should rarely be instantiated directly using its
    constructor. Instead, several class method constructors are available:

    * generators create leaf clouds from a set of parameters:

      * :meth:`.LeafCloud.cuboid`;
      * :meth:`.LeafCloud.sphere`;
      * :meth:`.LeafCloud.cylinder`;

    * :meth:`.LeafCloud.from_file` loads leaf positions and orientations from a
      text file;
    * :meth:`.LeafCloud.from_dict` dispatches calls to the other constructors.
    """

    id = documented(
        attr.ib(
            default="leaf_cloud",
            validator=attr.validators.optional(attr.validators.instance_of(str)),
        ),
        doc="Identifier for the object.",
        type="str",
        default="leaf_cloud",
    )

    leaf_positions = documented(
        pinttr.ib(factory=list, units=ucc.deferred("length")),
        doc="Leaf positions in cartesian coordinates as a (n, 3)-array.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="array-like",
        default="[]",
    )

    leaf_orientations = documented(
        attr.ib(factory=list),
        doc="Leaf orientations (normal vectors) in Cartesian coordinates as a "
        "(n, 3)-array.",
        type="array-like",
        default="[]",
    )

    leaf_radii = documented(
        pinttr.ib(
            factory=list,
            validator=[
                pinttr.validators.has_compatible_units,
                attr.validators.deep_iterable(member_validator=validators.is_positive),
            ],
            units=ucc.deferred("length"),
        ),
        doc="Leaf radii as a n-array.\n\nUnit-enabled field (default: ucc[length]).",
        type="array-like",
        default="[]",
    )

    @leaf_positions.validator
    @leaf_orientations.validator
    @leaf_radii.validator
    def _positions_orientations_validator(self, attribute, value):
        if not (
            len(self.leaf_positions)
            == len(self.leaf_orientations)
            == len(self.leaf_radii)
        ):
            raise ValueError(
                f"While validating {attribute.name}: "
                f"leaf_positions, leaf_orientations and leaf_radii must have the "
                f"same length. Got "
                f"len(leaf_positions) = {len(self.leaf_positions)}, "
                f"len(leaf_orientations) = {len(self.leaf_orientations)}, "
                f"len(leaf_radii) = {len(self.leaf_radii)}."
            )

    leaf_reflectance = documented(
        attr.ib(
            default=0.5,
            converter=SpectrumFactory.converter("reflectance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum of the leaves in the cloud. "
        "Must be a reflectance spectrum (dimensionless).",
        type=":class:`.Spectrum`",
        default="0.5",
    )

    leaf_transmittance = documented(
        attr.ib(
            default=0.5,
            converter=SpectrumFactory.converter("transmittance"),
            validator=[
                attr.validators.instance_of(Spectrum),
                validators.has_quantity("transmittance"),
            ],
        ),
        doc="Transmittance spectrum of the leaves in the cloud. "
        "Must be a transmittance spectrum (dimensionless).",
        type=":class:`.Spectrum`",
        default="0.5",
    )

    def n_leaves(self):
        """Return the number of leaves in the leaf cloud.

        Returns → int:
            Number of leaves in the leaf cloud."""
        return len(self.leaf_positions)

    @classmethod
    def cuboid(cls, seed=12345, avoid_overlap=False, **kwargs):
        """Generate a leaf cloud with an axis-aligned cuboid shape (and a square
        footprint on the gound). Parameters are checked by the
        :class:`.CuboidLeafCloudParams` class, which allows for many parameter
        combinations.

        .. seealso:: :class:`.CuboidLeafCloudParams`

        The produced leaf cloud covers uniformly the
        :math:`(x, y, z) \\in [-\\dfrac{l_h}{2}, + \\dfrac{l_h}{2}] \\times [-\\dfrac{l_h}{2}, + \\dfrac{l_h}{2}] \\times [0, l_v]`
        region. Leaf orientation is controlled by the ``mu`` and ``nu`` parameters
        of an approximated inverse beta distribution
        :cite:`Ross1991MonteCarloMethods`.

        Finally, extra parameters control the random number generator and a
        basic and conservative leaf collision detection algorithm.

        Parameter ``seed`` (int):
            Seed for the random number generator.

        Parameter ``avoid_overlap`` (bool):
            If ``True``, generate leaf positions with strict collision checks to
            avoid overlapping.

        Parameter ``n_attempts`` (int):
            If ``avoid_overlap`` is ``True``, number of attempts made at placing
            a leaf without collision before giving up. Default: 1e5.
        """
        rng = np.random.default_rng(seed=seed)
        n_attempts = kwargs.pop("n_attempts", int(1e5))

        params = CuboidLeafCloudParams(**kwargs)

        if avoid_overlap:
            leaf_positions = _leaf_cloud_positions_cuboid_avoid_overlap(
                params.n_leaves,
                params.l_horizontal,
                params.l_vertical,
                params.leaf_radius,
                n_attempts,
                rng,
            )
        else:
            leaf_positions = _leaf_cloud_positions_cuboid(
                params.n_leaves, params.l_horizontal, params.l_vertical, rng
            )

        leaf_orientations = _leaf_cloud_orientations(
            params.n_leaves, params.mu, params.nu, rng
        )

        leaf_radii = _leaf_cloud_radii(params.n_leaves, params.leaf_radius)

        # Create leaf cloud object
        return cls(
            id=params.id,
            leaf_positions=leaf_positions,
            leaf_orientations=leaf_orientations,
            leaf_radii=leaf_radii,
            leaf_reflectance=params.leaf_reflectance,
            leaf_transmittance=params.leaf_transmittance,
        )

    @classmethod
    def sphere(cls, seed=12345, **kwargs):
        """
        Generate a leaf cloud with spherical shape. Parameters are checked by
        the :class:`.SphereLeafCloudParams` class.

        .. seealso:: :class:`.SphereLeafCloudParams`

        The produced leaf cloud covers uniformly the :math:`r < \\mathit{radius}`
        region. Leaf orientation is controlled by the ``mu`` and ``nu`` parameters
        of an approximated inverse beta distribution
        :cite:`Ross1991MonteCarloMethods`.

        An additional parameter controls the random number generator.

        Parameter ``seed`` (int):
            Seed for the random number generator.
        """
        rng = np.random.default_rng(seed=seed)
        params = SphereLeafCloudParams(**kwargs)
        leaf_positions = (
            _leaf_cloud_positions_sphere(params.radius, params.n_leaves) * ureg.m,
            rng,
        )
        leaf_orientations = _leaf_cloud_orientations(
            params.n_leaves, params.mu, params.nu, rng
        )
        leaf_radii = _leaf_cloud_radii(params.n_leaves, params.leaf_radius) * ureg.m

        # Create leaf cloud object
        return cls(
            id=id,
            leaf_positions=leaf_positions,
            leaf_orientations=leaf_orientations,
            leaf_radii=leaf_radii,
            leaf_reflectance=params.leaf_reflectance,
            leaf_transmittance=params.leaf_transmittance,
        )

    @classmethod
    def cylinder(cls, seed=12345, **kwargs):
        """
        Generate a leaf cloud with a cylindrical shape (vertical orientation).
        Parameters are checked by the :class:`.CylinderLeafCloudParams` class.

        .. seealso:: :class:`.CylinderLeafCloudParams`

        The produced leaf cloud covers uniformly the :math:`r < \\mathit{radius}, z \\in [0, l_v]`
        region. Leaf orientation is controlled by the ``mu`` and ``nu`` parameters
        of an approximated inverse beta distribution
        :cite:`Ross1991MonteCarloMethods`.

        An additional parameter controls the random number generator.

        Parameter ``seed`` (int):
            Seed for the random number generator.
        """
        rng = np.random.default_rng(seed=seed)
        params = CylinderLeafCloudParams(**kwargs)
        leaf_positions = (
            _leaf_cloud_positions_cylinder(
                params.n_leaves, params.radius, params.l_vertical, rng
            )
            * ureg.m
        )
        leaf_orientations = _leaf_cloud_orientations(
            params.n_leaves, params.mu, params.nu, rng
        )
        leaf_radii = _leaf_cloud_radii(params.n_leaves, params.leaf_radius) * ureg.m

        # Create leaf cloud object
        return cls(
            id=id,
            leaf_positions=leaf_positions,
            leaf_orientations=leaf_orientations,
            leaf_radii=leaf_radii,
            leaf_reflectance=params.leaf_reflectance,
            leaf_transmittance=params.leaf_transmittance,
        )

    @classmethod
    def from_file(
        cls,
        filename=None,
        leaf_transmittance=0.5,
        leaf_reflectance=0.5,
        id="leaf_cloud",
    ):
        """
        Construct a :class:`.LeafCloud` from a text file specifying the leaf
        positions and orientations.

        .. admonition:: File format

           Each line defines a single leaf with the 7 following numerical
           parameters separated by one or more spaces:

           * leaf radius;
           * leaf center (x, y and z coordinates);
           * leaf orientation (x, y and z of normal vector).

        .. important::

           Location coordinates are assumed to be given in meters.

        Parameter ``filename`` (str or PathLike):
            Path to the text file specifying the leaves in the leaf cloud.
            Can be absolute or relative. Required (setting to ``None`` will
            raise an exception).

        Parameter ``leaf_reflectance`` (float or :class:`.Spectrum`):
            Reflectance spectrum of the leaves in the cloud. Must be a reflectance
            spectrum (dimensionless). Default: 0.5.

        Parameter ``leaf_transmittance`` (float or :class:`.Spectrum`):
            Transmittance spectrum of the leaves in the cloud. Must be a
            transmittance spectrum (dimensionless). Default: 0.5.

        Parameter ``id`` (str):
            ID of the created :class:`LeafCloud` instance.

        Raises → ValueError:
            If ``filename`` is set to ``None``.

        Raises → FileNotFoundError:
            If ``filename`` does not point to an existing file.
        """
        if filename is None:
            raise ValueError("parameter 'filename' is required")

        if not os.path.isfile(filename):
            raise FileNotFoundError(f"no file at {filename} found.")

        radii_ = []
        positions_ = []
        orientations_ = []
        with open(os.path.abspath(filename), "r") as definition_file:
            for i, line in enumerate(definition_file):
                values = [float(x) for x in line.split()]
                radii_.append(values[0])
                positions_.append(values[1:4])
                orientations_.append(values[4:7])

        radii = np.array(radii_) * ureg.m
        positions = np.array(positions_) * ureg.m
        orientations = np.array(orientations_)

        return cls(
            id=id,
            leaf_positions=positions,
            leaf_orientations=orientations,
            leaf_radii=radii,
            leaf_reflectance=leaf_reflectance,
            leaf_transmittance=leaf_transmittance,
        )

    @classmethod
    def from_dict(cls, d):
        """
        Construct from a dictionary. This function first queries for a
        ``construct`` parameter. If it is found, dictionary parameters are used
        to call another class method constructor:

        * ``cuboid``: :meth:`.cuboid`;
        * ``sphere``: :meth:`.sphere`;
        * ``cylinder``: :meth:`.cylinder`;
        * ``from_file``: :meth:`.from_file`.

        If ``construct`` is missing, parameters are forwarded to the regular
        :class:`.LeafCloud` constructor.

        Parameter ``d`` (dict):
            Dictionary containing parameters passed to the selected constructor.
            Unit fields are pre-processed with :func:`pinttr.interpret_units`.
        """
        # Interpret unit fields if any
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Dispatch call based on 'construct' parameter
        construct = d_copy.pop("construct", None)

        if construct == "cuboid":
            return cls.cuboid(**d_copy)

        elif construct == "sphere":
            return cls.sphere(**d_copy)

        elif construct == "cylinder":
            return cls.cylinder(**d_copy)

        elif construct == "from_file":
            return cls.from_file(**d_copy)

        elif construct is None:
            return cls(**d_copy)

        else:
            raise ValueError(f"parameter 'construct': unsupported value '{construct}'")

    @staticmethod
    def convert(value):
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`from_dict` to
        create a :class:`.LeafCloud`.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return LeafCloud.from_dict(value)

        return value

    def shapes(self, ref=False):
        """Return shape plugin specifications.

        Parameter ``ref`` (bool):
            If ``True``, declare leaf BSDFs using references.

        Returns → dict:
            A dictionary suitable for merge with a :class:`.KernelDict`
            containing all the shapes in the leaf cloud.
        """
        from eradiate.kernel.core import ScalarTransform4f, coordinate_system

        kernel_length = uck.get("length")
        shapes_dict = {}

        if ref:
            bsdf = {"type": "ref", "id": f"bsdf_{self.id}"}
        else:
            bsdf = self.bsdfs()[f"bsdf_{self.id}"]

        for i_leaf, (position, normal, radius) in enumerate(
            zip(
                self.leaf_positions.m_as(kernel_length),
                self.leaf_orientations,
                self.leaf_radii.m_as(kernel_length),
            )
        ):
            _, up = coordinate_system(normal)
            to_world = ScalarTransform4f.look_at(
                origin=position, target=position + normal, up=up
            ) * ScalarTransform4f.scale(radius)

            shapes_dict[f"{self.id}_leaf_{i_leaf}"] = {
                "type": "disk",
                "bsdf": bsdf,
                "to_world": to_world,
            }

        return shapes_dict

    def bsdfs(self):
        """
        Return BSDF plugin specifications.

        Returns → dict:
            Return a dictionary suitable for merge with a :class:`.KernelDict`
            containing all the BSDFs attached to the shapes in the leaf cloud.
        """
        return {
            f"bsdf_{self.id}": {
                "type": "bilambertian",
                "reflectance": self.leaf_reflectance.kernel_dict()["spectrum"],
                "transmittance": self.leaf_transmittance.kernel_dict()["spectrum"],
            }
        }

    def kernel_dict(self, ref=True):
        if not ref:
            return self.shapes(ref=False)
        else:
            return {**self.bsdfs(), **self.shapes(ref=True)}


@parse_docs
@attr.s
class InstancedLeafCloud(SceneElement):
    """
    Specification a leaf cloud, alongside the locations of instances (*i.e.*
    clones) of it.
    """

    leaf_cloud = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(LeafCloud.convert),
            validator=attr.validators.optional(attr.validators.instance_of(LeafCloud)),
        ),
        doc="Instanced leaf cloud. Can be specified as a dictionary, which will "
        "be interpreted by :meth:`.LeafCloud.from_dict`.",
        type=":class:`LeafCloud`",
        default="None",
    )

    instance_positions = documented(
        pinttr.ib(factory=list, units=ucc.deferred("length")),
        doc="Instance positions as an (n, 3)-array.\n"
        "\n"
        "Unit-enabled field (default: ucc[length])",
        type="array-like",
        default="[]",
    )

    @instance_positions.validator
    def _instance_positions_validator(self, attribute, value):
        if value.shape and value.shape[0] > 0 and value.shape[1] != 3:
            raise ValueError(
                f"while validating {attribute.name}, must be an array of shape "
                f"(n, 3), got {value.shape}"
            )

    @classmethod
    def from_file(cls, filename=None, leaf_cloud=None):
        """
        Construct a :class:`.InstancedLeafCloud` from a text file specifying
        instance positions.

        .. admonition:: File format

           Each line defines an instance position as a whitespace-separated
           3-vector of Cartesian coordinates.

        .. important::

           Location coordinates are assumed to be given in meters.

        Parameter ``filename`` (str or PathLike):
            Path to the text file specifying the leaves in the canopy.
            Can be absolute or relative. Required (setting to ``None`` will
            raise an exception).

        Parameter ``leaf_cloud`` (:class:`.LeafCloud` or dict):
            :class:`.LeafCloud` to be instanced. If a dictionary is passed,
            if is interpreted by :meth:`.LeafCloud.from_dict`. If set to
            ``None``, an empty leaf cloud will be created.

        Raises → ValueError:
            If ``filename`` is set to ``None``.

        Raises → FileNotFoundError:
            If ``filename`` does not point to an existing file.
        """
        if filename is None:
            raise ValueError("parameter 'filename' is required")

        if not os.path.isfile(filename):
            raise FileNotFoundError(f"no file at {filename} found.")

        if leaf_cloud is None:
            leaf_cloud = LeafCloud()

        instance_positions = []

        with open(filename, "r") as f:
            for i_line, line in enumerate(f):
                try:
                    coords = np.array(line.split(), dtype=float)
                except ValueError as e:
                    raise ValueError(
                        f"while reading {filename}, on line {i_line+1}: "
                        f"cannot convert {line} to a 3-vector!"
                    ) from e

                if len(coords) != 3:
                    raise ValueError(
                        f"while reading {filename}, on line {i_line+1}: "
                        f"cannot convert {line} to a 3-vector!"
                    )

                instance_positions.append(coords)

        instance_positions = np.array(instance_positions) * ureg.m
        return cls(leaf_cloud=leaf_cloud, instance_positions=instance_positions)

    @classmethod
    def from_dict(cls, d):
        """
        Construct from a dictionary. This function first queries for a
        ``type`` parameter. If it is found, dictionary parameters are used to
        call another class method constructor:

        * ``file``: :meth:`.from_file`.

        If ``construct`` is missing, parameters are forwarded to the regular
        :class:`.InstancedLeafCloud` constructor.

        Parameter ``d`` (dict):
            Dictionary containing parameters passed to the selected constructor.
            Unit fields are pre-processed with :func:`pinttr.interpret_units`.
        """
        # Interpret unit fields if any
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Dispatch call based on 'construct' parameter
        construct = d_copy.pop("construct", None)

        if construct == "from_file":
            return cls.from_file(**d_copy)
        elif construct is None:
            return cls(**d_copy)
        else:
            raise ValueError(f"parameter 'construct': unsupported value '{construct}'")

    @staticmethod
    def convert(value):
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`from_dict` to
        create a :class:`.InstancedLeafCloud`.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return InstancedLeafCloud.from_dict(value)

        return value

    def bsdfs(self):
        """
        Return BSDF plugin specifications.

        Returns → dict:
            Return a dictionary suitable for merge with a :class:`.KernelDict`
            containing all the BSDFs attached to the shapes in the leaf cloud.
        """
        return self.leaf_cloud.bsdfs()

    def shapes(self, ref=False):
        """
        Return shape plugin specifications.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            in the canopy.
        """
        return {
            self.leaf_cloud.id: {"type": "shapegroup", **self.leaf_cloud.shapes(ref)}
        }

    def instances(self):
        """
        Return instance plugin specifications.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing instances.
        """
        from eradiate.kernel.core import ScalarTransform4f

        kernel_length = uck.get("length")

        return {
            f"{self.leaf_cloud.id}_instance_{i}": {
                "type": "instance",
                "group": {"type": "ref", "id": self.leaf_cloud.id},
                "to_world": ScalarTransform4f.translate(position.m_as(kernel_length)),
            }
            for i, position in enumerate(self.instance_positions)
        }

    def kernel_dict(self, ref=True):
        return {**self.bsdfs(), **self.shapes(ref), **self.instances()}


@BiosphereFactory.register("discrete_canopy")
@parse_docs
@attr.s
class DiscreteCanopy(Canopy):
    """
    An abstract discrete canopy, consisting of one or several clouds of
    disk-shaped leaves. Each leaf cloud can be instanced arbitrarily. The
    produced canopy can be padded with more clones of itself using the
    :meth:`~.DiscreteCanopy.padded` method.

    .. admonition:: Tutorials

       * Practical usage ⇒ :ref:`sphx_glr_examples_generated_tutorials_biosphere_01_discrete_canopy.py`
    """

    instanced_leaf_clouds = documented(
        attr.ib(
            factory=list,
            converter=lambda value: [
                InstancedLeafCloud.convert(x)
                for x in pinttr.util.always_iterable(value)
            ]
            if not isinstance(value, dict)
            else [InstancedLeafCloud.convert(value)],
        ),
        doc="List of :class:`.InstancedLeafCloud` defining the canopy. Can be "
        "initialised with a :class:`.InstancedLeafCloud`, which will be "
        "automatically wrapped into a list. Dictionary-based specifications are "
        "allowed as well.",
        type="list[:class:`.InstancedLeafCloud`]",
        default="[]",
    )

    def bsdfs(self):
        """Return BSDF plugin specifications.

        Returns → dict:
            Return a dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the BSDFs
            attached to the shapes in the canopy.
        """
        result = {}
        for leaf_cloud_instance in self.leaf_cloud_instances:
            result = {**result, **leaf_cloud_instance.bsdfs()}
        return result

    def shapes(self, ref=False):
        """
        Return shape plugin specifications.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing all the shapes
            in the canopy.
        """
        result = {}
        for leaf_cloud_instance in self.leaf_cloud_instances:
            result = {**result, **leaf_cloud_instance.shapes(ref)}
        return result

    def instances(self):
        """
        Return instance plugin specifications.

        Returns → dict:
            A dictionary suitable for merge with a
            :class:`~eradiate.scenes.core.KernelDict` containing instances.
        """
        result = {}
        for instanced_leaf_cloud in self.instanced_leaf_clouds:
            result = {**result, **instanced_leaf_cloud.instances()}
        return result

    def kernel_dict(self, ref=True):
        if not ref:
            raise ValueError("'ref' must be set to True")

        result = {}
        for instanced_leaf_cloud in self.instanced_leaf_clouds:
            result = {
                **result,
                **instanced_leaf_cloud.bsdfs(),
                **instanced_leaf_cloud.shapes(),
                **instanced_leaf_cloud.instances(),
            }

        return result

    def padded(self, padding):
        """
        Return a copy of the current canopy padded with additional copies.

        Parameter ``padding`` (int):
            Amount of padding around the canopy. Must be positive or zero.
            The resulting padded canopy is a grid of
            :math:`2 \\times \\mathit{padding} + 1` copies.

        Returns → :class:`.DiscreteCanopy`:
            Padded copy.
        """
        if padding < 0:
            raise ValueError("padding must be >= 0")

        if padding == 0:
            return self

        # We'll return a copy
        result = deepcopy(self)

        # Convenience aliases
        config_length = ucc.get("length")
        x_size, y_size = result.size.m_as(config_length)[:2]
        padding_factors = np.array(list(range(-padding, padding + 1)))

        for instanced_leaf_cloud in result.instanced_leaf_clouds:
            # More convenience aliases
            old_instance_positions = instanced_leaf_cloud.instance_positions.m_as(
                config_length
            )
            n_instances_per_cell = old_instance_positions.shape[0]
            # Allocate array for results
            new_instance_positions = np.empty(
                (len(padding_factors) ** 2 * n_instances_per_cell, 3)
            )

            for k, (x_offset_factor, y_offset_factor) in enumerate(
                itertools.product(padding_factors, padding_factors)
            ):
                # Set vector by which we will translate instance positions
                offset = np.array(
                    [x_size * x_offset_factor, y_size * y_offset_factor, 0.0]
                )

                # Compute new instances
                start_idx = k * n_instances_per_cell
                stop_idx = (k + 1) * n_instances_per_cell
                new_instance_positions[start_idx:stop_idx, :] = (
                    old_instance_positions[:, :] + offset
                )

            instanced_leaf_cloud.instance_positions = (
                new_instance_positions * config_length
            )

        # Update size
        result.size[:2] *= len(padding_factors)

        return result

    @classmethod
    def homogeneous(cls, id="homogeneous_discrete_canopy", **leaf_cloud_kwargs):
        """
        Generate a homogeneous discrete canopy.

        Parameter ``id`` (str):
            Canopy object ID.

        Parameter ``leaf_cloud_kwargs``:
            Keyword arguments forwarded to :meth:`.LeafCloud.cuboid`.

            .. note:: The leaf cloud's ID will be set to ``f"{id}_leaf_cloud"``.

        Returns → :class:`.DiscreteCanopy`:
            Created canopy object.
        """
        # Check parameters
        leaf_cloud_params = CuboidLeafCloudParams(**leaf_cloud_kwargs)
        config_length = ucc.get("length")
        size = [
            leaf_cloud_params.l_horizontal.m_as(config_length),
            leaf_cloud_params.l_horizontal.m_as(config_length),
            leaf_cloud_params.l_vertical.m_as(config_length),
        ] * config_length
        leaf_cloud_id = f"{id}_leaf_cloud"

        # Construct canopy
        return cls(
            id=id,
            size=size,
            instanced_leaf_clouds=[
                InstancedLeafCloud(
                    instance_positions=[[0, 0, 0]],
                    leaf_cloud=LeafCloud.cuboid(**leaf_cloud_kwargs, id=leaf_cloud_id),
                )
            ],
        )

    @classmethod
    def from_files(cls, id="discrete_canopy", size=None, leaf_cloud_dicts=None):
        """
        Create an abstract discrete canopy from leaf cloud and instance text
        file specifications.

         .. admonition:: Leaf cloud dictionary format

           Each item of the ``leaf_cloud_dicts`` list shall have the following
           structure:

           .. code:: python

              {
                  "sub_id": "some_value",  # leaf cloud ID string part, optional if leaf_cloud_dicts has only 1 entry
                  "instance_filename": "some_path",  # path to instance specification file
                  "leaf_cloud_filename": "some_other_path", # path to leaf cloud specification file
                  "leaf_reflectance": 0.5,  # optional, leaf reflectance (default: 0.5)
                  "leaf_transmittance": 0.5,  # optional, leaf transmittance (default: 0.5)
              }


        Parameter ``id`` (str):
            Canopy ID.

        Parameter ``size`` (array-like):
            Canopy size as a 3-vector (in metres).

        Parameter ``leaf_cloud_dicts`` (list[dict]):
            List of dictionary specifying leaf clouds and instances (see format
            above).

        Returns → :class:`.DiscreteCanopy`:
            Created canopy object.
        """
        for param in ["size", "instance_filename", "leaf_cloud_dicts"]:
            if param is None:
                raise ValueError(f"parameter '{param}' is required")

        instanced_leaf_clouds = []

        for leaf_cloud_dict in leaf_cloud_dicts:
            instance_filename = leaf_cloud_dict.get("instance_filename", None)

            leaf_cloud_params = {}
            sub_id = leaf_cloud_dict.get("sub_id", None)

            if sub_id is None:
                if len(leaf_cloud_dicts) > 1:
                    raise ValueError("parameter 'sub_id' must be set")
                leaf_cloud_params["id"] = f"{id}_leaf_cloud"
            else:
                leaf_cloud_params["id"] = f"{id}_{sub_id}_leaf_cloud"
            leaf_cloud_params["filename"] = leaf_cloud_dict.get(
                "leaf_cloud_filename", None
            )
            leaf_cloud_params["leaf_reflectance"] = leaf_cloud_dict.get(
                "leaf_reflectance", 0.5
            )
            leaf_cloud_params["leaf_transmittance"] = leaf_cloud_dict.get(
                "leaf_transmittance", 0.5
            )

            instanced_leaf_clouds.append(
                InstancedLeafCloud.from_file(
                    filename=instance_filename,
                    leaf_cloud=LeafCloud.from_file(**leaf_cloud_params),
                )
            )

        return cls(id=id, size=size, instanced_leaf_clouds=instanced_leaf_clouds)

    @classmethod
    def from_dict(cls, d):
        """
        Construct from a dictionary. This function first queries for a
        ``construct`` parameter. If it is found, dictionary parameters are used
        to call another class method constructor:

        * ``homogeneous``: :meth:`.DiscreteCanopy.homogeneous`;
        * ``from_files``: :meth:`.DiscreteCanopy.from_files`.

        If ``construct`` is missing, parameters are forwarded to the regular
        :class:`.InstancedLeafCloud` constructor.

        Parameter ``d`` (dict):
            Dictionary containing parameters passed to the selected constructor.
            Unit fields are pre-processed with :func:`pinttr.interpret_units`.
        """
        # Interpret unit fields if any
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Store padding value
        padding = d_copy.pop("padding", 0)

        # Dispatch call based on 'construct' parameter
        construct = d_copy.pop("construct", None)

        if construct == "homogeneous":
            result = cls.homogeneous(**d_copy)
        elif construct == "from_files":
            result = cls.from_files(**d_copy)
        elif construct is None:
            result = cls(**d_copy)
        else:
            raise ValueError(f"parameter 'construct': unsupported value '{construct}'")

        return result.padded(padding)


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

        **Dictionary format:**

        .. code:: python

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

        return cls(
            leaf_cloud_instances_list=[
                InstancedLeafCloud(
                    leaf_cloud=leaf_cloud, instance_positions=sphere_positions
                )
            ],
            size=size,
        )


@BiosphereFactory.register("real_zoom_in_canopy")
@parse_docs
@attr.s
class RealZoomInCanopy(DiscreteCanopy):
    """Generates the canopy for the
    `real zoom in scenario <https://rami-benchmark.jrc.ec.europa.eu/HTML/RAMI3/EXPERIMENTS3/HETEROGENEOUS/REAL_ZOOM-IN/REAL_ZOOM-IN.php>`_ in the RAMI benchmark.

    This canopy is composed of a set of sphere- and cylinder-shaped leaf clouds.
    """

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

        **Dictionary format:**

        .. code:: python

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
            leaf_cloud_instances_list=[
                InstancedLeafCloud(
                    leaf_cloud=spherical_leaf_cloud, instance_positions=sphere_positions
                ),
                InstancedLeafCloud(
                    leaf_cloud=cylindrical_leaf_cloud,
                    instance_positions=cylinder_positions,
                ),
            ],
            size=size,
        )
