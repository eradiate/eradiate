from __future__ import annotations

import os
import warnings

import attrs
import mitsuba as mi
import numpy as np
import pint
import pinttr
import scipy as sp
import scipy.special

from ._core import CanopyElement
from ..core import SceneElement, traverse
from ..spectra import Spectrum, spectrum_factory
from ... import validators
from ...attrs import define, documented, get_doc
from ...kernel import SceneParameter, SearchSceneParameter
from ...units import unit_context_config as ucc
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


def _sample_lad(mu, nu, rng):
    """
    Generate an angle sample from the Leaf angle distribution function according to
    :cite:`GoelStrebel1984`, using the rejection method.
    """

    while True:
        rands = rng.random(2)
        theta_candidate = rands[0] * np.pi / 2.0
        gs_lad = (
            2.0
            / np.pi
            * sp.special.gamma(mu + nu)
            / (sp.special.gamma(mu) * sp.special.gamma(mu))
            * pow((1 - (2 * theta_candidate) / np.pi), mu - 1)
            * pow((2 * theta_candidate) / np.pi, nu - 1)
        )

        # scaling factor for the rejection method set to 2.0 to encompass the
        # entire distribution
        if rands[1] * 2.0 <= gs_lad:
            return theta_candidate


@ureg.wraps(ureg.m, (None, ureg.m, ureg.m, None))
def _leaf_cloud_positions_cuboid(n_leaves, l_horizontal, l_vertical, rng):
    """
    Compute leaf positions for a cuboid-shaped leaf cloud (square footprint).
    """
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
    """
    Compute leaf positions for a cuboid-shaped leaf cloud (square footprint).
    This function also performs conservative collision checks to avoid leaf
    overlapping. This process might take a very long time, if the parameters
    specify a very dense leaf cloud. Consider using
    :func:`_leaf_cloud_positions_cuboid`.
    """
    try:
        import aabbtree
    except ModuleNotFoundError:
        warnings.warn(
            "To use the collision detection feature, you must install AABBTree.\n"
            "See instructions on https://aabbtree.readthedocs.io/#installation."
        )
        raise

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


@ureg.wraps(ureg.m, (None, None, ureg.m, ureg.m, ureg.m))
def _leaf_cloud_positions_ellipsoid(n_leaves: int, rng, a: float, b: float, c: float):
    """
    Compute leaf positions for an ellipsoid leaf cloud.
    The ellipsoid follows the equation:
    :math:`\frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{c^2}= 1`
    """

    positions = []

    while len(positions) < n_leaves:
        rand = rng.random(3)
        x = (rand[0] - 0.5) * 2 * a
        y = (rand[1] - 0.5) * 2 * b
        z = (rand[2] - 0.5) * 2 * c

        if (x**2 / a**2) + (y**2 / b**2) + (z**2 / c**2) <= 1.0:
            positions.append([x, y, z])

    return positions


@ureg.wraps(ureg.m, (None, ureg.m, ureg.m, None))
def _leaf_cloud_positions_cylinder(n_leaves, radius, l_vertical, rng):
    """
    Compute leaf positions for a cylinder-shaped leaf cloud (vertical
    orientation).
    """

    positions = np.empty((n_leaves, 3))

    for i in range(n_leaves):
        rand = rng.random(3)
        phi = rand[0] * 2 * np.pi
        r = rand[1] * radius
        z = rand[2] * l_vertical
        positions[i, :] = [r * np.cos(phi), r * np.sin(phi), z]

    return positions


@ureg.wraps(ureg.m, (None, ureg.m, ureg.m, None))
def _leaf_cloud_positions_cone(n_leaves, radius, l_vertical, rng):
    """
    Compute leaf positions for a cone-shaped leaf cloud (vertical
    orientation, tip pointing towards positive z).
    """

    positions = np.empty((n_leaves, 3))

    # uniform cone sampling from here:
    # https://stackoverflow.com/questions/41749411/uniform-sampling-by-volume-within-a-cone
    for i in range(n_leaves):
        rand = rng.random(3)
        h = l_vertical * (rand[0] ** (1 / 3))
        r = radius / l_vertical * h * np.sqrt(rand[1])
        phi = rand[2] * 2 * np.pi
        positions[i, :] = [r * np.cos(phi), r * np.sin(phi), l_vertical - h]

    return positions


@ureg.wraps(None, (None, None, None, None))
def _leaf_cloud_orientations(n_leaves, mu, nu, rng):
    """Compute leaf orientations."""
    orientations = np.empty((n_leaves, 3))
    for i in range(np.shape(orientations)[0]):
        theta = _sample_lad(mu, nu, rng)
        phi = rng.random() * 2.0 * np.pi

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


@define
class LeafCloudParams:
    """
    Base class to implement advanced parameter checking for :class:`.LeafCloud`
    generators.
    """

    _id = documented(
        attrs.field(default="leaf_cloud"),
        doc="Leaf cloud identifier.",
        type="str",
        default='"leaf_cloud"',
    )

    _leaf_reflectance = documented(
        attrs.field(default=0.5), doc="Leaf reflectance.", type="float", default="0.5"
    )

    _leaf_transmittance = documented(
        attrs.field(default=0.5), doc="Leaf transmittance.", type="float", default="0.5"
    )

    _mu = documented(
        attrs.field(default=1.066),
        doc="First parameter of the inverse beta distribution approximation used "
        "to generate leaf orientations.",
        type="float",
        default="1.066",
    )

    _nu = documented(
        attrs.field(default=1.853),
        doc="Second parameter of the inverse beta distribution approximation used "
        "to generate leaf orientations.",
        type="float",
        default="1.853",
    )

    _n_leaves = documented(
        attrs.field(default=None), doc="Number of leaves.", type="int"
    )

    _leaf_radius = documented(
        pinttr.field(default=None, units=ucc.deferred("length")),
        doc="Leaf radius.\n\nUnit-enabled field (default: ucc['length']).",
        type="float",
    )

    def update(self):
        try:
            for field in [x.name.lstrip("_") for x in self.__attrs_attrs__]:
                self.__getattribute__(field)
        except Exception as e:
            raise Exception(
                f"cannot compute field '{field}', parameter set is likely "
                "under-constrained"
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


@define
class CuboidLeafCloudParams(LeafCloudParams):
    """
    Advanced parameter checking class for the cuboid :class:`.LeafCloud`
    generator. Some of the parameters can be inferred from each other.

    Parameters defined below can be used (without leading underscore) as
    keyword arguments to the :meth:`.LeafCloud.cuboid` class method
    constructor. Parameters without defaults are connected by a dependency
    graph used to compute required parameters (outlined in the figure below).

    The following parameter sets are valid:

    * ``n_leaves``, ``leaf_radius``, ``l_horizontal``, ``l_vertical``;
    * ``lai``, ``leaf_radius``, ``l_horizontal``, ``l_vertical``;
    * ``lai``, ``leaf_radius``, ``l_horizontal``, ``hdo``, ``hvr``;
    * and more!

    .. only:: latex

       .. figure:: /_images/cuboid_leaf_cloud_params.png

    .. only:: not latex

       .. figure:: /_images/cuboid_leaf_cloud_params.svg

    Warnings
    --------
    In case of over-specification, no consistency check is
    performed.

    See Also
    --------
    :meth:`.LeafCloud.cuboid`
    """

    _l_horizontal = documented(
        pinttr.field(default=None, units=ucc.deferred("length")),
        doc="Leaf cloud horizontal extent. *Suggested default: 30 m.*\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="float",
    )

    _l_vertical = documented(
        pinttr.field(default=None, units=ucc.deferred("length")),
        doc="Leaf cloud vertical extent. *Suggested default: 3 m.*\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="float",
    )

    _lai = documented(
        pinttr.field(default=None, units=ureg.dimensionless),
        doc="Leaf cloud leaf area index (LAI). *Physical range: [0, 10]; "
        "suggested default: 3.*\n"
        "\n"
        "Unit-enabled field (default: ucc['dimensionless']).",
        type="float",
    )

    _hdo = documented(
        pinttr.field(default=None, units=ucc.deferred("length")),
        doc="Mean horizontal distance between leaves.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="float",
    )

    _hvr = documented(
        pinttr.field(default=None),
        doc="Ratio of mean horizontal leaf distance and vertical leaf cloud extent. "
        "*Suggested default: 0.1.*",
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
            self._l_horizontal = np.pi * self.leaf_radius**2 * self.n_leaves / self.lai
        return self._l_horizontal

    @property
    def l_vertical(self):
        if self._l_vertical is None:
            self._l_vertical = (
                self.lai * self.hdo**3 / (np.pi * self.leaf_radius**2 * self.hvr)
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


@define
class SphereLeafCloudParams(LeafCloudParams):
    """
    Advanced parameter checking class for the sphere :class:`.LeafCloud`
    generator.

    See Also
    --------
    :meth:`.LeafCloud.sphere`
    """

    _radius = documented(
        pinttr.field(default=1.0 * ureg.m, units=ucc.deferred("length")),
        doc="Leaf cloud radius.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
        default="1 m",
    )

    @property
    def radius(self):
        return self._radius


@define
class EllipsoidLeafCloudParams(LeafCloudParams):
    """
    Advanced parameter checking class for the ellipsoid :class:`.LeafCloud`
    generator. Parameters ``a``, ``b`` and ``c`` denote the ellipsoid's half
    axes along the x, y, and z directions respectively. If either ``b`` or ``c``
    are not set by the user, they default to being equal to ``a``.
    Accordingly, a sphere of radius ``r`` can be parametrized by setting ``a=r``.

    See Also
    --------
    :meth:`.LeafCloud.ellipsoid`
    """

    _a = documented(
        pinttr.field(default=1.0 * ureg.m, units=ucc.deferred("length")),
        doc="Leaf cloud radius.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
        default="1 m",
    )

    _b = documented(
        pinttr.field(default=None, units=ucc.deferred("length")),
        doc="Leaf cloud radius.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
        default="1 m",
    )

    _c = documented(
        pinttr.field(default=None, units=ucc.deferred("length")),
        doc="Leaf cloud radius.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
        default="1 m",
    )

    @property
    def a(self):
        if self._a <= 0:
            raise ValueError(
                "Ellipsoid half axis parameters must be strictly larger than zero!"
            )
        return self._a

    @property
    def b(self):
        if self._b is None:
            self._b = self.a
        elif self._b <= 0:
            raise ValueError(
                "Ellipsoid half axis parameters must be strictly larger than zero!"
            )
        return self._b

    @property
    def c(self):
        if self._c is None:
            self._c = self.a
        elif self._c <= 0:
            raise ValueError(
                "Ellipsoid half axis parameters must be strictly larger than zero!"
            )
        return self._c


@define
class CylinderLeafCloudParams(LeafCloudParams):
    """
    Advanced parameter checking class for the cylinder :class:`.LeafCloud`
    generator.

    See Also
    --------
    :meth:`.LeafCloud.cylinder`
    """

    _radius = documented(
        pinttr.field(default=1.0 * ureg.m, units=ucc.deferred("length")),
        doc="Leaf cloud radius.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
        default="1 m",
    )

    _l_vertical = documented(
        pinttr.field(default=1.0 * ureg.m, units=ucc.deferred("length")),
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


@define
class ConeLeafCloudParams(LeafCloudParams):
    """
    Advanced parameter checking class for the cone :class:`.LeafCloud`
    generator.

    See Also
    --------
    :meth:`.LeafCloud.cone`
    """

    _radius = documented(
        pinttr.field(default=1.0 * ureg.m, units=ucc.deferred("length")),
        doc="Leaf cloud radius.\n\nUnit-enabled field (default: ucc[length]).",
        type="float",
        default="1 m",
    )

    _l_vertical = documented(
        pinttr.field(default=1.0 * ureg.m, units=ucc.deferred("length")),
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


@define(eq=False, slots=False)
class LeafCloud(CanopyElement):
    """
    A container class for leaf clouds in abstract discrete canopies.
    Holds parameters completely characterizing the leaf cloud's leaves.

    In practice, this class should rarely be instantiated directly using its
    constructor. Instead, several class method constructors are available:

    * generators create leaf clouds from a set of parameters:

      * :meth:`.LeafCloud.cone`;
      * :meth:`.LeafCloud.cuboid`;
      * :meth:`.LeafCloud.cylinder`;
      * :meth:`.LeafCloud.ellipsoid`;
      * :meth:`.LeafCloud.sphere`;

    * :meth:`.LeafCloud.from_file` loads leaf positions and orientations from a
      text file.

    .. admonition:: Class method constructors

       .. autosummary::

          cuboid
          cylinder
          ellipsoid
          from_file
          sphere
    """

    # --------------------------------------------------------------------------
    #                                 Fields
    # --------------------------------------------------------------------------

    id: str | None = documented(
        attrs.field(
            default="leaf_cloud",
            validator=attrs.validators.optional(attrs.validators.instance_of(str)),
        ),
        doc=get_doc(SceneElement, "id", "doc"),
        type=get_doc(SceneElement, "id", "type"),
        init_type=get_doc(SceneElement, "id", "init_type"),
        default="'leaf_cloud'",
    )

    leaf_positions: pint.Quantity = documented(
        pinttr.field(factory=list, units=ucc.deferred("length")),
        doc="Leaf positions in cartesian coordinates as a (n, 3)-array.\n"
        "\n"
        "Unit-enabled field (default: ucc['length']).",
        type="quantity",
        init_type="array-like",
        default="[]",
    )

    leaf_orientations: np.ndarray = documented(
        attrs.field(factory=list, converter=np.array),
        doc="Leaf orientations (normal vectors) in Cartesian coordinates as a "
        "(n, 3)-array.",
        type="ndarray",
        default="[]",
    )

    leaf_radii: pint.Quantity = documented(
        pinttr.field(
            factory=list,
            validator=[
                pinttr.validators.has_compatible_units,
                attrs.validators.deep_iterable(member_validator=validators.is_positive),
            ],
            units=ucc.deferred("length"),
        ),
        doc="Leaf radii as a n-array.\n\nUnit-enabled field (default: ucc[length]).",
        init_type="array-like",
        type="quantity",
        default="[]",
    )

    @leaf_positions.validator
    @leaf_orientations.validator
    def _positions_orientations_validator(self, attribute, value):
        if not len(value):
            return

        if not value.ndim == 2 or value.shape[1] != 3:
            raise ValueError(
                f"While validating {attribute.name}: shape should be (N, 3), "
                f"got {value.shape}"
            )

    @leaf_positions.validator
    @leaf_orientations.validator
    @leaf_radii.validator
    def _positions_orientations_radii_validator(self, attribute, value):
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

    leaf_reflectance: Spectrum = documented(
        attrs.field(
            default=0.5,
            converter=spectrum_factory.converter("reflectance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("reflectance"),
            ],
        ),
        doc="Reflectance spectrum of the leaves in the cloud. "
        "Must be a reflectance spectrum (dimensionless).",
        type=":class:`.Spectrum`",
        init_type=":class:`.Spectrum` or dict",
        default="0.5",
    )

    leaf_transmittance: Spectrum = documented(
        attrs.field(
            default=0.5,
            converter=spectrum_factory.converter("transmittance"),
            validator=[
                attrs.validators.instance_of(Spectrum),
                validators.has_quantity("transmittance"),
            ],
        ),
        doc="Transmittance spectrum of the leaves in the cloud. "
        "Must be a transmittance spectrum (dimensionless).",
        type=":class:`.Spectrum`",
        init_type=":class:`.Spectrum` or dict",
        default="0.5",
    )

    # --------------------------------------------------------------------------
    #                          Properties and accessors
    # --------------------------------------------------------------------------

    def n_leaves(self) -> int:
        """
        Returns
        -------
        int
            Number of leaves in the leaf cloud.
        """
        return len(self.leaf_positions)

    def surface_area(self) -> pint.Quantity:
        """
        Returns
        -------
        quantity
            Total surface area.
        """
        return np.sum(np.pi * self.leaf_radii * self.leaf_radii).squeeze()

    # --------------------------------------------------------------------------
    #                              Constructors
    # --------------------------------------------------------------------------

    @classmethod
    def cuboid(
        cls, seed: int = 12345, avoid_overlap: bool = False, **kwargs
    ) -> LeafCloud:
        """
        Generate a leaf cloud with an axis-aligned cuboid shape (and a square
        footprint on the ground). Parameters are checked by the
        :class:`.CuboidLeafCloudParams` class, which allows for many parameter
        combinations.

        The produced leaf cloud uniformly covers the
        :math:`(x, y, z) \\in \\left[ -\\dfrac{l_h}{2}, + \\dfrac{l_h}{2} \\right] \\times \\left[ -\\dfrac{l_h}{2}, + \\dfrac{l_h}{2} \\right] \\times [0, l_v]`
        region. Leaf orientation is controlled by the ``mu`` and ``nu`` parameters
        of an approximated inverse beta distribution
        :cite:`Ross1991MonteCarloMethods`.

        Finally, extra parameters control the random number generator and a
        basic and conservative leaf collision detection algorithm.

        Parameters
        ----------
        seed : int
            Seed for the random number generator.

        avoid_overlap : bool
            If ``True``, generate leaf positions with strict collision checks to
            avoid overlapping.

        n_attempts : int
            If ``avoid_overlap`` is ``True``, number of attempts made at placing
            a leaf without collision before giving up. Default: 1e5.

        **kwargs
            Keyword arguments interpreted by :class:`.CuboidLeafCloudParams`.

        Returns
        -------
        .LeafCloud
            Generated leaf cloud.

        See Also
        --------
        :class:`.CuboidLeafCloudParams`
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
    def sphere(cls, seed: int = 12345, **kwargs) -> LeafCloud:
        """
        Generate a leaf cloud with spherical shape. Parameters are checked by
        the :class:`.SphereLeafCloudParams` class.

        The produced leaf cloud covers uniformly the :math:`r < \\mathtt{radius}`
        region. Leaf orientation is controlled by the ``mu`` and ``nu`` parameters
        of an approximated inverse beta distribution
        :cite:`Ross1991MonteCarloMethods`.

        An additional parameter controls the random number generator.

        Parameters
        ----------
        seed : int
            Seed for the random number generator.

        **kwargs
            Keyword arguments interpreted by :class:`.SphereLeafCloudParams`.

        Returns
        -------
        :class:`.LeafCloud`
            Generated leaf cloud.

        See Also
        --------
        :class:`.SphereLeafCloudParams`
        """
        rng = np.random.default_rng(seed=seed)
        params = SphereLeafCloudParams(**kwargs)
        leaf_positions = _leaf_cloud_positions_ellipsoid(
            params.n_leaves, rng, params.radius, params.radius, params.radius
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
    def ellipsoid(cls, seed: int = 12345, **kwargs) -> LeafCloud:
        """
        Generate a leaf cloud with ellipsoid shape. Parameters are checked by
        the :class:`.EllipsoidLeafCloudParams` class.

        The produced leaf cloud covers uniformly the volume enclosed by
        :math:`\\frac{x^2}{a^2} + \\frac{y^2}{b^2} + \\frac{z^2}{c^2}= 1` .

        Leaf orientation is controlled by the ``mu`` and ``nu`` parameters
        of an approximated inverse beta distribution
        :cite:`Ross1991MonteCarloMethods`.

        An additional parameter controls the random number generator.

        Parameters
        ----------
        seed : int
            Seed for the random number generator.

        **kwargs
            Keyword arguments interpreted by :class:`.EllipsoidLeafCloudParams`.

        Returns
        -------
        :class:`.LeafCloud`
            Generated leaf cloud.

        See Also
        --------
        :class:`.EllipsoidLeafCloudParams`
        """
        rng = np.random.default_rng(seed=seed)
        params = EllipsoidLeafCloudParams(**kwargs)
        leaf_positions = _leaf_cloud_positions_ellipsoid(
            params.n_leaves, rng, params.a, params.b, params.c
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
    def cylinder(cls, seed: int = 12345, **kwargs) -> LeafCloud:
        """
        Generate a leaf cloud with a cylindrical shape (vertical orientation).
        Parameters are checked by the :class:`.CylinderLeafCloudParams` class.

        The produced leaf cloud covers uniformly the
        :math:`r < \\mathtt{radius}, z \\in [0, l_v]`
        region. Leaf orientation is controlled by the ``mu`` and ``nu`` parameters
        of an approximated inverse beta distribution
        :cite:`Ross1991MonteCarloMethods`.

        An additional parameter controls the random number generator.

        Parameters
        ----------
        seed : int
            Seed for the random number generator.

        **kwargs
            Keyword arguments interpreted by :class:`.CylinderLeafCloudParams`.

        Returns
        -------
        :class:`.LeafCloud`
            Generated leaf cloud.

        See Also
        --------
        :class:`.CylinderLeafCloudParams`
        """
        rng = np.random.default_rng(seed=seed)
        params = CylinderLeafCloudParams(**kwargs)
        leaf_positions = _leaf_cloud_positions_cylinder(
            params.n_leaves, params.radius, params.l_vertical, rng
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
    def cone(cls, seed: int = 12345, **kwargs) -> LeafCloud:
        """
        Generate a leaf cloud with a right conical shape (vertical orientation).
        Parameters are checked by the :class:`.ConeLeafCloudParams` class.

        The produced leaf cloud covers uniformly the
        :math:`r < \\mathtt{radius} \\cdot \\left( 1 - \\frac{z}{l_v} \\right), z \\in [0, l_v]`
        region. Leaf orientation is controlled by the ``mu`` and ``nu`` parameters
        of an approximated inverse beta distribution
        :cite:`Ross1991MonteCarloMethods`.

        An additional parameter controls the random number generator.

        Parameters
        ----------
        seed : int
            Seed for the random number generator.

        **kwargs
            Keyword arguments interpreted by :class:`.ConeLeafCloudParams`.

        Returns
        -------
        :class:`.LeafCloud`
            Generated leaf cloud.

        See Also
        --------
        :class:`.ConeLeafCloudParams`
        """
        rng = np.random.default_rng(seed=seed)
        params = ConeLeafCloudParams(**kwargs)
        leaf_positions = _leaf_cloud_positions_cone(
            params.n_leaves, params.radius, params.l_vertical, rng
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
    def from_file(
        cls,
        filename,
        leaf_transmittance: float | Spectrum = 0.5,
        leaf_reflectance: float | Spectrum = 0.5,
        id: str = "leaf_cloud",
    ) -> LeafCloud:
        """
        Construct a :class:`.LeafCloud` from a text file specifying the leaf
        positions and orientations.

        .. admonition:: File format

           Each line defines a single leaf with the following 7 numerical
           parameters separated by one or more spaces:

           * leaf radius;
           * leaf center (x, y and z coordinates);
           * leaf orientation (x, y and z of normal vector).

        .. important::

           All quantities are assumed to be given in metre.

        Parameters
        ----------
        filename : path-like
            Path to the text file specifying the leaves in the leaf cloud.
            Can be absolute or relative.

        leaf_reflectance : :class:`.Spectrum` or float
            Reflectance spectrum of the leaves in the cloud. Must be a reflectance
            spectrum (dimensionless). Default: 0.5.

        leaf_transmittance : :class:`.Spectrum` of float
            Transmittance spectrum of the leaves in the cloud. Must be a
            transmittance spectrum (dimensionless). Default: 0.5.

        id : str
            ID of the created :class:`.LeafCloud` instance.

        Returns
        -------
        :class:`.LeafCloud`:
            Generated leaf cloud.

        Raises
        ------
        Raises
        ------
        FileNotFoundError
            If ``filename`` does not point to an existing file.
        """
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

    # --------------------------------------------------------------------------
    #                       Kernel dictionary generation
    # --------------------------------------------------------------------------

    @property
    def bsdf_id(self) -> str:
        return f"bsdf_{self.id}"

    @property
    def _template_bsdfs(self) -> dict:
        objects = {
            "reflectance": traverse(self.leaf_reflectance)[0].data,
            "transmittance": traverse(self.leaf_transmittance)[0].data,
        }

        result = {f"{self.bsdf_id}.type": "bilambertian"}

        for obj_key, obj_template in objects.items():
            for key, param in obj_template.items():
                result[f"{self.bsdf_id}.{obj_key}.{key}"] = param

        return result

    @property
    def _template_shapes(self) -> dict:
        length_units = uck.get("length")
        result = {}
        bsdf_dict = {"type": "ref", "id": self.bsdf_id}

        for i_leaf, (position, normal, radius) in enumerate(
            zip(
                self.leaf_positions.m_as(length_units),
                self.leaf_orientations,
                self.leaf_radii.m_as(length_units),
            )
        ):
            _, up = mi.coordinate_system(normal)
            to_world = mi.ScalarTransform4f.look_at(
                origin=position, target=position + normal, up=up
            ) @ mi.ScalarTransform4f.scale(radius)

            result[f"{self.id}_leaf_{i_leaf}"] = {
                "type": "disk",
                "bsdf": bsdf_dict,
                "to_world": to_world,
            }

        return result

    @property
    def _params_bsdfs(self) -> dict:
        objects = {
            "reflectance": traverse(self.leaf_reflectance)[1].data,
            "transmittance": traverse(self.leaf_transmittance)[1].data,
        }

        result = {}

        for obj_key, obj_template in objects.items():
            for key, param in obj_template.items():
                # If no lookup strategy is set, we must add one
                if isinstance(param, SceneParameter) and param.search is None:
                    param = attrs.evolve(
                        param,
                        search=SearchSceneParameter(
                            mi.BSDF,
                            self.bsdf_id,
                            parameter_relpath=f"{obj_key}.{key}",
                        ),
                    )

                result[f"{self.bsdf_id}.{obj_key}.{key}"] = param

        return result

    @property
    def _params_shapes(self) -> dict:
        return {}

    # --------------------------------------------------------------------------
    #                               Other methods
    # --------------------------------------------------------------------------

    def translated(self, xyz: pint.Quantity) -> LeafCloud:
        """
        Return a copy of self translated by the vector ``xyz``.

        Parameters
        ----------
        xyz : :class:`pint.Quantity`
            A 3-vector or a (N, 3)-array by which leaves will be translated. If
            (N, 3) variant is used, the array shape must match that of
            ``leaf_positions``.

        Returns
        -------
        :class:`LeafCloud`
            Translated copy of self.

        Raises
        ------
        ValueError
            Sizes of ``xyz`` and ``self.leaf_positions`` are incompatible.
        """
        if xyz.ndim <= 1:
            xyz = xyz.reshape((1, 3))
        elif xyz.shape != self.leaf_positions.shape:
            raise ValueError(
                f"shapes xyz {xyz.shape} and self.leaf_positions "
                f"{self.leaf_positions.shape} do not match"
            )

        return attrs.evolve(self, leaf_positions=self.leaf_positions + xyz)
