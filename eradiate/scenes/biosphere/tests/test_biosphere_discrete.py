import os
import tempfile

import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.biosphere._discrete import (
    AbstractTree,
    DiscreteCanopy,
    InstancedCanopyElement,
    LeafCloud,
    _inversebeta,
    _leaf_cloud_orientations,
    _leaf_cloud_positions_cuboid,
    _leaf_cloud_positions_cuboid_avoid_overlap,
    _leaf_cloud_positions_cylinder,
    _leaf_cloud_positions_ellipsoid,
    _leaf_cloud_radii,
)
from eradiate.scenes.core import KernelDict

# -- Fixture definitions -------------------------------------------------------


@pytest.fixture(scope="function")
def rng():
    yield np.random.default_rng(seed=12345)


@pytest.fixture(scope="module")
def tempfile_leaves():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tempfile_leaves.txt")
        with open(filename, "w") as tf:
            tf.write("0.100 8.864 9.040 1.878 -0.314 0.025 0.949\n")
            tf.write("0.100 9.539 -10.463 0.627 0.489 -0.276 0.828\n")
            tf.write("0.100 -2.274 -9.204 0.797 0.618 0.184 0.764\n")
            tf.write("0.100 -9.957 -4.971 0.719 -0.066 0.100 0.993\n")
            tf.write("0.100 5.339 9.153 0.500 0.073 -0.294 0.953\n")
        yield filename


@pytest.fixture(scope="module")
def tempfile_spheres():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "tempfile_leaves.txt")
        with open(filename, "w") as tf:
            tf.write("8.864 9.040 1.878\n")
            tf.write("0.100 9.539 -10.463\n")
            tf.write("0.100 -2.274 -9.204\n")
            tf.write("0.100 -9.957 -4.971\n")
            tf.write("0.100 5.339 9.153\n")
        yield filename


# -- Basic function tests ------------------------------------------------------


def test_inversebeta(rng):
    """Unit tests for :func:`_inversebeta`."""
    assert np.allclose(_inversebeta(1, 1, rng), 0.4178246)


def test_leaf_cloud_positions_cuboid(rng):
    """Unit tests for :func:`_leaf_cloud_positions_cuboid`."""
    positions = _leaf_cloud_positions_cuboid(1, 1.0 * ureg.m, 1.0 * ureg.m, rng)
    assert positions.shape == (1, 3)
    assert np.allclose(positions, [-0.27266398, -0.18324166, 0.79736546] * ureg.m)


def test_leaf_cloud_positions_cuboid_avoid_overlap(rng):
    """Unit tests for :func:`_leaf_cloud_positions_cuboid_avoid_overlap`."""
    positions = _leaf_cloud_positions_cuboid_avoid_overlap(
        1000, 1.0 * ureg.m, 1.0 * ureg.m, 1.0 * ureg.cm, 1e3, rng
    )
    assert positions.shape == (1000, 3)


def test_leaf_cloud_positions_ellipsoid(rng):
    """Unit tests for :func:`_leaf_cloud_positions_ellipsoid`."""
    positions = _leaf_cloud_positions_ellipsoid(
        1, rng, 1.0 * ureg.m, 1.0 * ureg.m, 1.0 * ureg.m
    )
    assert positions.shape == (1, 3)
    assert np.allclose(positions, [-0.54532796, -0.36648332, 0.59473091] * ureg.m)


def test_leaf_cloud_positions_cylinder(rng):
    """Unit tests for :func:`_leaf_cloud_positions_cylinder`."""
    positions = _leaf_cloud_positions_cylinder(1, 1.0 * ureg.m, 1.0 * ureg.m, rng)
    assert positions.shape == (1, 3)
    assert np.allclose(positions, [0.04495472, 0.3135521, 0.79736546] * ureg.m)


def test_leaf_cloud_orientations(rng):
    """Unit tests for :func:`_leaf_cloud_orientations`."""
    orientations = _leaf_cloud_orientations(1, 1, 1, rng)
    assert orientations.shape == (1, 3)
    assert np.allclose(orientations, [0.36553973, 0.85464537, 0.36873025])


def test_leaf_cloud_radii():
    radii = _leaf_cloud_radii(1, 10.0 * ureg.cm)
    assert radii.shape == (1,)
    assert np.allclose(radii, 10.0 * ureg.cm)


# -- LeafCloud tests -----------------------------------------------------------


def test_leaf_cloud_instantiate(mode_mono):
    """Unit tests for :class:`LeafCloud`'s default constructor."""

    # Empty constructor does not raise
    assert LeafCloud()

    # Now with more sensible values
    assert LeafCloud(
        leaf_positions=[[0, 0, 0]], leaf_orientations=[[0, 0, 1]], leaf_radii=[0.1]
    )

    # Incorrect dimensionality raises
    with pytest.raises(ValueError):
        LeafCloud(
            leaf_positions=[0, 0, 0],
            leaf_orientations=[0, 0, 1],
            leaf_radii=[0.1] * 3,
        )


@pytest.mark.parametrize(
    "leaf_positions, xyz, expected_leaf_positions",
    [
        (
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
            [0, 0, 1],
            [[0, 0, 1], [0, 0, 2], [0, 1, 1], [1, 0, 1]],
        ),
        (
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
            [[0, 0, 1]] + [[0, 0, 0]] * 3,
            [[0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
        ),
    ],
)
def test_leaf_cloud_translated(mode_mono, leaf_positions, xyz, expected_leaf_positions):
    """Unit tests for :meth:`LeafCloud.translated`."""

    leaf_positions_size = np.array(leaf_positions).size
    leaf_orientations = [[0, 0, 1]] * (leaf_positions_size // 3)
    leaf_radii = [0.1] * (leaf_positions_size // 3)

    leaf_cloud = LeafCloud(
        leaf_positions=leaf_positions,
        leaf_orientations=leaf_orientations,
        leaf_radii=leaf_radii,
    )

    # Translated object is a copy
    translated_leaf_cloud = leaf_cloud.translated(xyz * ureg.m)
    assert translated_leaf_cloud is not leaf_cloud

    # Translated object has offset leaf positions
    assert np.allclose(
        translated_leaf_cloud.leaf_positions,
        np.array(expected_leaf_positions) * ureg.m,
    )

    # Shapes do not match: translated() raises
    with pytest.raises(ValueError):
        leaf_cloud.translated([[1, 0, 0]] * 2 * ureg.m)


@pytest.mark.slow
def test_leaf_cloud_generate(mode_mono):
    """Test the instantiation of the leaf cloud class from parameters"""

    # Try constructing leaf clouds with various param combinations
    # n_leaves, leaf_radius, l_horizontal, l_vertical
    cloud = LeafCloud.cuboid(
        n_leaves=1000,
        leaf_radius=10.0 * ureg.cm,
        l_horizontal=30.0 * ureg.m,
        l_vertical=3.0 * ureg.m,
    )
    assert cloud
    assert cloud.n_leaves() == 1000
    # Cloud covers the expected extent
    assert np.allclose(
        np.min(cloud.leaf_positions, axis=0).m_as("m"), [-15, -15, 0], atol=0.05
    )
    assert np.allclose(
        np.max(cloud.leaf_positions, axis=0).m_as("m"), [15, 15, 3], atol=0.05
    )

    # lai, leaf_radius, l_horizontal, l_vertical
    cloud = LeafCloud.cuboid(
        lai=3.0,
        leaf_radius=15.0 * ureg.cm,
        l_horizontal=10.0 * ureg.m,
        l_vertical=3.0 * ureg.m,
    )
    assert cloud
    assert cloud.n_leaves() == 4244
    assert np.allclose(
        np.min(cloud.leaf_positions, axis=0).m_as("m"), [-5, -5, 0], atol=0.05
    )
    assert np.allclose(
        np.max(cloud.leaf_positions, axis=0).m_as("m"), [5, 5, 3], atol=0.05
    )

    # lai, leaf_radius, l_horizontal, hdo, hvr
    cloud = LeafCloud.cuboid(
        lai=2.0,
        leaf_radius=10.0 * ureg.cm,
        l_horizontal=10.0 * ureg.m,
        hdo=10 * ureg.cm,
        hvr=0.05,
    )
    assert cloud
    assert cloud.n_leaves() == 6366
    assert np.allclose(
        np.min(cloud.leaf_positions, axis=0).m_as("m"), [-5, -5, 0], atol=0.05
    )
    assert np.allclose(
        np.max(cloud.leaf_positions, axis=0).m_as("m"), [5, 5, 1.272], atol=0.05
    )

    # n_leaves, leaf_radius, l_horizontal, l_vertical, collision check
    cloud = LeafCloud.cuboid(
        n_leaves=1000,
        leaf_radius=10.0 * ureg.cm,
        l_horizontal=30.0 * ureg.m,
        l_vertical=3.0 * ureg.m,
        avoid_overlap=True,
    )
    assert cloud
    assert cloud.n_leaves() == 1000
    assert np.allclose(
        np.min(cloud.leaf_positions, axis=0).m_as("m"), [-15, -15, 0], atol=0.05
    )
    assert np.allclose(
        np.max(cloud.leaf_positions, axis=0).m_as("m"), [15, 15, 3], atol=0.05
    )


def test_leaf_cloud_from_file(mode_mono, tempfile_leaves):
    """Unit testing for :meth:`LeafCloud.from_file`."""
    ctx = KernelDictContext()

    # A LeafCloud instance can be loaded from a file on the hard drive
    cloud = LeafCloud.from_file(tempfile_leaves)
    assert len(cloud.leaf_positions) == 5
    assert np.allclose(cloud.leaf_radii, 0.1 * ureg.m)
    # Produced kernel dict is valid
    assert KernelDict.new(cloud, ctx=ctx).load()


def test_leaf_cloud_from_dict(mode_mono, tempfile_leaves):
    """Unit testing for :meth:`LeafCloud.from_dict`."""

    # Use regular constructor if no "type" parameter is specified
    assert LeafCloud.from_dict(
        {
            "leaf_positions": [[0, 0, 0], [1, 1, 1]],
            "leaf_orientations": [[0, 0, 1], [1, 0, 0]],
            "leaf_radii": [0.1, 0.1],
        }
    )

    # Dispatch to from_file if requested
    cloud1 = LeafCloud.from_file(tempfile_leaves)
    cloud2 = LeafCloud.from_dict(
        {"construct": "from_file", "filename": tempfile_leaves}
    )
    assert np.all(cloud1.leaf_positions == cloud2.leaf_positions)
    assert np.all(cloud1.leaf_orientations == cloud2.leaf_orientations)
    assert np.all(cloud1.leaf_radii == cloud2.leaf_radii)
    assert np.all(cloud1.leaf_transmittance == cloud2.leaf_transmittance)
    assert np.all(cloud1.leaf_reflectance == cloud2.leaf_reflectance)

    # Dispatch to generator if requested
    cloud = LeafCloud.from_dict(
        {
            "construct": "cuboid",
            "n_leaves": 100,
            "l_horizontal": 10.0,
            "l_vertical": 1.0,
            "leaf_radius": 10.0,
            "leaf_radius_units": "cm",
        }
    )
    assert np.allclose(cloud.leaf_radii, 10.0 * ureg.cm)


def test_leaf_cloud_kernel_dict(mode_mono):
    ctx = KernelDictContext()

    """Partial unit testing for :meth:`LeafCloud.kernel_dict`."""
    cloud_id = "my_cloud"
    cloud = LeafCloud(
        id=cloud_id,
        leaf_positions=[[0, 0, 0], [1, 1, 1]],
        leaf_orientations=[[1, 0, 0], [0, 1, 0]],
        leaf_radii=[0.1, 0.1],
        leaf_reflectance=0.5,
        leaf_transmittance=0.5,
    )

    kernel_dict = cloud.kernel_dict(ctx=ctx)

    # The BSDF is bilambertian with the parameters we initially set
    assert kernel_dict[f"bsdf_{cloud_id}"] == {
        "type": "bilambertian",
        "reflectance": {"type": "uniform", "value": 0.5},
        "transmittance": {"type": "uniform", "value": 0.5},
    }

    # Leaves are disks
    for shape_key in [f"{cloud_id}_leaf_0", f"{cloud_id}_leaf_1"]:
        assert kernel_dict[shape_key]["type"] == "disk"

    # Kernel dict is valid
    assert KernelDict.new(kernel_dict).load()


# -- AbstractTree tests --------------------------------------------------------


def test_abstract_tree_instantiate(mode_mono):
    """Unit tests for :class:`AbstractTree`'s default constructor."""

    # Empty constructor does not raise
    assert AbstractTree()

    # Now with more sensible values
    assert AbstractTree(
        leaf_cloud=LeafCloud(
            leaf_positions=[[0, 0, 0]], leaf_orientations=[[0, 0, 1]], leaf_radii=[0.1]
        ),
        trunk_height=0.5,
        trunk_radius=0.1,
        trunk_reflectance=0.5,
    )


def test_abstract_tree_dispatch_leaf_cloud(mode_mono, tempfile_leaves):
    """Test if contained LeafCloud is instantiated in all variants"""
    ctx = KernelDictContext()

    # A LeafCloud instance can be loaded from a file on the hard drive
    tree = AbstractTree(leaf_cloud=LeafCloud.from_file(tempfile_leaves))
    assert len(tree.leaf_cloud.leaf_positions) == 5
    assert np.allclose(tree.leaf_cloud.leaf_radii, 0.1 * ureg.m)
    # Produced kernel dict is valid
    assert KernelDict.new(tree.leaf_cloud, ctx=ctx).load()

    # Use regular constructor if no "type" parameter is specified
    assert AbstractTree(
        leaf_cloud=LeafCloud.from_dict(
            {
                "leaf_positions": [[0, 0, 0], [1, 1, 1]],
                "leaf_orientations": [[0, 0, 1], [1, 0, 0]],
                "leaf_radii": [0.1, 0.1],
            }
        )
    )

    # Dispatch to from_file if requested
    tree1 = AbstractTree(leaf_cloud=LeafCloud.from_file(tempfile_leaves))
    tree2 = AbstractTree(
        leaf_cloud=LeafCloud.from_dict(
            {"construct": "from_file", "filename": tempfile_leaves}
        )
    )
    # assert leaf clouds are equal
    assert np.all(tree1.leaf_cloud.leaf_positions == tree2.leaf_cloud.leaf_positions)
    assert np.all(
        tree1.leaf_cloud.leaf_orientations == tree2.leaf_cloud.leaf_orientations
    )
    assert np.all(tree1.leaf_cloud.leaf_radii == tree2.leaf_cloud.leaf_radii)
    assert np.all(
        tree1.leaf_cloud.leaf_transmittance == tree2.leaf_cloud.leaf_transmittance
    )
    assert np.all(
        tree1.leaf_cloud.leaf_reflectance == tree2.leaf_cloud.leaf_reflectance
    )

    # Dispatch to generator if requested
    tree = AbstractTree(
        leaf_cloud=LeafCloud.from_dict(
            {
                "construct": "cuboid",
                "n_leaves": 100,
                "l_horizontal": 10.0,
                "l_vertical": 1.0,
                "leaf_radius": 10.0,
                "leaf_radius_units": "cm",
            }
        )
    )
    assert len(tree.leaf_cloud.leaf_radii) == 100
    assert len(tree.leaf_cloud.leaf_positions) == 100
    assert np.allclose(tree.leaf_cloud.leaf_radii, 10.0 * ureg.cm)


def test_abstract_tree_from_dict(mode_mono, tempfile_leaves):
    """Unit testing for :meth:`AbstractTree.from_dict`."""

    assert AbstractTree.from_dict(
        {
            "leaf_cloud": {
                "construct": "cuboid",
                "n_leaves": 100,
                "l_horizontal": 10.0,
                "l_vertical": 1.0,
                "leaf_radius": 10.0,
                "leaf_radius_units": "cm",
            },
            "trunk_height": 2.0,
            "trunk_radius": 0.2,
            "trunk_reflectance": 0.5,
        }
    )

    tree1 = AbstractTree.from_dict(
        {
            "leaf_cloud": {
                "construct": "cuboid",
                "n_leaves": 100,
                "l_horizontal": 10.0,
                "l_vertical": 1.0,
                "leaf_radius": 10.0,
                "leaf_radius_units": "cm",
            },
            "trunk_height": 2.0,
            "trunk_radius": 0.2,
            "trunk_reflectance": 0.5,
        }
    )

    tree2 = AbstractTree(
        leaf_cloud=LeafCloud.from_dict(
            {
                "construct": "cuboid",
                "n_leaves": 100,
                "l_horizontal": 10.0,
                "l_vertical": 1.0,
                "leaf_radius": 10.0,
                "leaf_radius_units": "cm",
            }
        ),
        trunk_height=2.0,
        trunk_radius=0.2,
        trunk_reflectance=0.5,
    )

    assert tree1.trunk_height == tree2.trunk_height
    assert tree1.trunk_radius == tree2.trunk_radius
    assert tree1.trunk_reflectance == tree2.trunk_reflectance

    assert np.allclose(tree1.leaf_cloud.leaf_positions, tree2.leaf_cloud.leaf_positions)
    assert np.allclose(
        tree1.leaf_cloud.leaf_orientations, tree2.leaf_cloud.leaf_orientations
    )
    assert np.all(
        tree1.leaf_cloud.leaf_transmittance == tree2.leaf_cloud.leaf_transmittance
    )
    assert np.all(
        tree1.leaf_cloud.leaf_reflectance == tree2.leaf_cloud.leaf_reflectance
    )


def test_abstract_tree_kernel_dict(mode_mono):
    ctx = KernelDictContext()

    """Partial unit testing for :meth:`LeafCloud.kernel_dict`."""
    tree_id = "my_tree"
    cloud_id = "my_cloud"
    tree = AbstractTree(
        leaf_cloud=LeafCloud(
            id=cloud_id,
            leaf_positions=[[0, 0, 0], [1, 1, 1]],
            leaf_orientations=[[1, 0, 0], [0, 1, 0]],
            leaf_radii=[0.1, 0.1],
            leaf_reflectance=0.5,
            leaf_transmittance=0.5,
        ),
        id=tree_id,
        trunk_height=2.0,
        trunk_radius=0.2,
        trunk_reflectance=0.5,
    )

    kernel_dict = tree.kernel_dict(ctx=ctx)

    # The BSDF is bilambertian with the parameters we initially set
    assert kernel_dict[f"bsdf_{cloud_id}"] == {
        "type": "bilambertian",
        "reflectance": {"type": "uniform", "value": 0.5},
        "transmittance": {"type": "uniform", "value": 0.5},
    }

    # Leaves are disks
    for shape_key in [f"{cloud_id}_leaf_0", f"{cloud_id}_leaf_1"]:
        assert kernel_dict[shape_key]["type"] == "disk"

    # Kernel dict is valid
    assert KernelDict.new(kernel_dict).load()


# -- InstancedCanopyElement tests ----------------------------------------------


def test_instanced_canopy_element_create(mode_mono):
    """Unit testing for the :class:`InstancedCanopyElement` constructor."""
    cloud = LeafCloud(
        leaf_positions=[[0, 0, 0]], leaf_orientations=[[0, 0, 1]], leaf_radii=[0.1]
    )
    tree = AbstractTree(
        leaf_cloud=cloud, trunk_height=1.0, trunk_radius=0.1, trunk_reflectance=0.5
    )
    positions = np.array([[-5, -5, -5], [5, 5, 5]])
    assert InstancedCanopyElement(canopy_element=cloud, instance_positions=positions)
    assert InstancedCanopyElement(canopy_element=tree, instance_positions=positions)


def test_instanced_canopy_element_kernel_dict(mode_mono):
    """Unit testing for :meth:`InstancedLeafCloud.kernel_dict`."""
    ctx = KernelDictContext()

    cloud = LeafCloud(
        leaf_positions=[[0, 0, 0]],
        leaf_orientations=[[0, 0, 1]],
        leaf_radii=[0.1],
        id="leaf_cloud",
    )
    tree = AbstractTree(
        leaf_cloud=cloud,
        trunk_height=1.0,
        trunk_radius=0.1,
        trunk_reflectance=0.5,
        id="tree",
    )
    positions = np.array([[-5, -5, -5], [5, 5, 5]])
    kernel_dict = InstancedCanopyElement(
        canopy_element=tree, instance_positions=positions, id="element"
    ).kernel_dict(ctx=ctx)

    # The generated kernel dictionary can be instantiated
    assert KernelDict.new(kernel_dict).load()


def test_instanced_leaf_cloud_from_file(mode_mono, tempfile_spheres):
    """Unit testing for :meth:`InstancedLeafCloud.from_file`."""
    assert InstancedCanopyElement.from_file(
        filename=tempfile_spheres,
        canopy_element=LeafCloud.from_dict(
            {
                "leaf_positions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                "leaf_orientations": [[0, 0, 1], [1, 0, 0]],
                "leaf_radii": [0.1, 0.1],
            }
        ),
    )


def test_instanced_canopy_element_from_dict(mode_mono, tempfile_spheres):
    """Unit testing for :meth:`InstancedLeafCloud.from_dict`."""
    ctx = KernelDictContext()

    # We can instantiate from a full-dict spec
    instanced_leaf_cloud = InstancedCanopyElement.from_dict(
        {
            "canopy_element": {
                "type": "leaf_cloud",
                "leaf_positions": [[0, 0, 0], [1, 1, 1]],
                "leaf_orientations": [[0, 0, 1], [1, 0, 0]],
                "leaf_radii": [0.1, 0.1],
            },
            "instance_positions": np.array([[-5, -5, -5], [5, 5, 5]]),
        }
    )
    assert instanced_leaf_cloud

    # The generated kernel dictionary can be instantiated
    assert KernelDict.new(instanced_leaf_cloud.kernel_dict(ctx=ctx)).load()

    # We can access the from_file constructor from a dict
    assert InstancedCanopyElement.from_dict(
        {
            "construct": "from_file",
            "filename": tempfile_spheres,
            "canopy_element": {
                "type": "leaf_cloud",
                "leaf_positions": [[0, 0, 0], [1, 1, 1]],
                "leaf_orientations": [[0, 0, 1], [1, 0, 0]],
                "leaf_radii": [0.1, 0.1],
            },
        }
    )


# -- DiscreteCanopy tests ------------------------------------------------------


def test_discrete_canopy_instantiate(mode_mono):
    # Constructing an empty canopy is possible
    assert DiscreteCanopy()


def test_discrete_canopy_homogeneous(mode_mono):
    ctx = KernelDictContext()

    # The generate_homogeneous() constructor returns a valid canopy object
    canopy = DiscreteCanopy.homogeneous(
        n_leaves=1, leaf_radius=0.1, l_horizontal=10, l_vertical=3
    )
    assert KernelDict.new(canopy, ctx=ctx).load()

    # We can reproduce this behaviour using the dict-based API
    canopy = DiscreteCanopy.from_dict(
        {
            "construct": "homogeneous",
            "n_leaves": 1,
            "leaf_radius": 0.1,
            "l_horizontal": 10,
            "l_vertical": 3,
        }
    )
    assert KernelDict.new(canopy, ctx=ctx).load()


def test_discrete_canopy_from_files(mode_mono, tempfile_spheres, tempfile_leaves):
    ctx = KernelDictContext()

    # The from_files() constructor returns a valid canopy object
    canopy = DiscreteCanopy.leaf_cloud_from_files(
        size=[1, 1, 1],
        leaf_cloud_dicts=[
            {
                "sub_id": "spheres_1",
                "instance_filename": tempfile_spheres,
                "leaf_cloud_filename": tempfile_leaves,
            },
            {
                "sub_id": "spheres_2",
                "instance_filename": tempfile_spheres,
                "leaf_cloud_filename": tempfile_leaves,
            },
        ],
    )
    assert KernelDict.new(canopy, ctx=ctx).load()

    # We can reproduce this behaviour using the dict-based API
    canopy = DiscreteCanopy.from_dict(
        {
            "construct": "leaf_cloud_from_files",
            "size": [1, 1, 1],
            "leaf_cloud_dicts": [
                {
                    "sub_id": "spheres_1",
                    "instance_filename": tempfile_spheres,
                    "leaf_cloud_filename": tempfile_leaves,
                },
                {
                    "sub_id": "spheres_2",
                    "instance_filename": tempfile_spheres,
                    "leaf_cloud_filename": tempfile_leaves,
                },
            ],
        }
    )
    assert KernelDict.new(canopy, ctx=ctx).load()


def test_discrete_canopy_advanced(mode_mono, tempfile_spheres, tempfile_leaves):
    """
    A more advanced test where we load a pre-computed canopy consisting of a
    generated cuboid leaf cloud and a series of instanced pre-computed leaf
    clouds.
    """
    ctx = KernelDictContext()

    # First use the regular Python API
    canopy = DiscreteCanopy(
        size=[1.0, 1.0, 1.0] * ureg.m,
        instanced_canopy_elements=[
            InstancedCanopyElement(
                instance_positions=[[0, 0, 0]],
                canopy_element=LeafCloud.cuboid(
                    n_leaves=1,
                    leaf_radius=0.1,
                    l_horizontal=10,
                    l_vertical=3,
                    id="leaf_cloud_cuboid",
                ),
            ),
            InstancedCanopyElement.from_file(
                filename=tempfile_spheres,
                canopy_element=LeafCloud.from_file(
                    filename=tempfile_leaves, id="leaf_cloud_precomputed"
                ),
            ),
        ],
    )
    assert KernelDict.new(canopy, ctx=ctx).load()

    # Reproduce previous example with dict API
    canopy = DiscreteCanopy.from_dict(
        {
            "size": [1.0, 1.0, 1.0] * ureg.m,
            "instanced_canopy_elements": [
                {
                    "instance_positions": [[0, 0, 0]],
                    "canopy_element": {
                        "type": "leaf_cloud",
                        "construct": "cuboid",
                        "n_leaves": 1,
                        "leaf_radius": 0.1,
                        "l_horizontal": 10,
                        "l_vertical": 3,
                        "id": "leaf_cloud_cuboid",
                    },
                },
                {
                    "construct": "from_file",
                    "filename": tempfile_spheres,
                    "canopy_element": {
                        "type": "leaf_cloud",
                        "construct": "from_file",
                        "filename": tempfile_leaves,
                        "id": "leaf_cloud_precomputed",
                    },
                },
            ],
        }
    )
    assert KernelDict.new(canopy, ctx=ctx).load()


def test_discrete_canopy_padded(mode_mono, tempfile_leaves, tempfile_spheres):
    """Unit tests for :meth:`.DiscreteCanopy.padded`"""
    ctx = KernelDictContext()

    canopy = DiscreteCanopy.leaf_cloud_from_files(
        id="canopy",
        size=[100, 100, 30],
        leaf_cloud_dicts=[
            {
                "instance_filename": tempfile_spheres,
                "leaf_cloud_filename": tempfile_leaves,
                "sub_id": "leaf_cloud",
            }
        ],
    )
    # The padded canopy object is valid and instantiable
    padded_canopy = canopy.padded(2)
    assert padded_canopy
    assert KernelDict.new(padded_canopy, ctx=ctx).load()
    # Padded canopy has (2*padding + 1) ** 2 times more instances than original
    assert len(padded_canopy.instances()) == len(canopy.instances()) * 25
    # Padded canopy has 2*padding + 1 times larger horizontal size than original
    assert np.allclose(padded_canopy.size[:2], 5 * canopy.size[:2])
    # Padded canopy has same vertical size as original
    assert padded_canopy.size[2] == canopy.size[2]
