import os
import tempfile

import attr
import numpy as np
import pytest

from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.scenes.biosphere._leaf_cloud import (
    LeafCloud,
    _leaf_cloud_orientations,
    _leaf_cloud_positions_cuboid,
    _leaf_cloud_positions_cuboid_avoid_overlap,
    _leaf_cloud_positions_cylinder,
    _leaf_cloud_positions_ellipsoid,
    _leaf_cloud_radii,
    _sample_lad,
)
from eradiate.scenes.core import traverse
from eradiate.test_tools.types import check_scene_element


@pytest.fixture(scope="function")
def rng():
    yield np.random.default_rng(seed=12345)


@pytest.fixture(scope="function")
def rng_mock():
    @attr.s
    class RngMock:
        values = attr.ib(converter=np.atleast_1d)

        def random(self, n=1):
            selector = np.array([i % len(self.values) for i in range(n)])
            return self.values[selector].squeeze()

    yield RngMock


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


# -- Basic function tests ------------------------------------------------------


def test_sample_lad(rng_mock):
    """
    Unit tests for :func:`_sample_lad`.
    The RNG Mock lets us test assert the probability of sampling a certain
    value from the Goel and Strebel LAD function.

    The a member of the mock represents the theta sample and is mapped
    to the [0, pi/2.] range.
    The b member represents the rejection sample value.
    """

    rng = rng_mock([0.1, 0.45])
    assert np.allclose(_sample_lad(2.531, 1.096, rng), np.pi / 20.0)

    rng = rng_mock([0.9, 0.45])
    assert np.allclose(_sample_lad(1.096, 2.531, rng), np.pi / 2.0 * 0.9)


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


def test_leaf_cloud_orientations(rng_mock):
    """Unit tests for :func:`_leaf_cloud_orientations`."""
    rng = rng_mock([0.1, 0.45])

    orientations = _leaf_cloud_orientations(1, 2.531, 1.096, rng)
    assert orientations.shape == (1, 3)
    assert np.allclose(orientations, [0.1265581, 0.09194987, 0.98768834])


def test_leaf_cloud_radii():
    radii = _leaf_cloud_radii(1, 10.0 * ureg.cm)
    assert radii.shape == (1,)
    assert np.allclose(radii, 10.0 * ureg.cm)


# ------------------------------------------------------------------------------
#                                 LeafCloud tests
# ------------------------------------------------------------------------------


def test_leaf_cloud_instantiate(mode_mono):
    """Unit tests for :class:`LeafCloud`'s default constructor."""

    # Empty constructor does not raise
    cloud = LeafCloud()
    check_scene_element(cloud)

    # Now with more sensible values, also check template and parameter map
    cloud = LeafCloud(
        id="cloud",
        leaf_positions=[[0, 0, 0]],
        leaf_orientations=[[0, 0, 1]],
        leaf_radii=[0.1],
    )
    check_scene_element(cloud)
    template, params = traverse(cloud)
    assert set(template.keys()) == {
        "bsdf_cloud.type",
        "bsdf_cloud.reflectance.type",
        "bsdf_cloud.reflectance.value",
        "bsdf_cloud.transmittance.type",
        "bsdf_cloud.transmittance.value",
        "cloud_leaf_0.type",
        "cloud_leaf_0.bsdf.type",
        "cloud_leaf_0.bsdf.id",
        "cloud_leaf_0.to_world",
    }
    assert set(params.keys()) == {
        "bsdf_cloud.reflectance.value",
        "bsdf_cloud.transmittance.value",
    }

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

    cloud = LeafCloud(
        leaf_positions=leaf_positions,
        leaf_orientations=leaf_orientations,
        leaf_radii=leaf_radii,
    )

    # Translated object is a copy
    translated_leaf_cloud = cloud.translated(xyz * ureg.m)
    assert translated_leaf_cloud is not cloud

    # Translated object has offset leaf positions
    assert np.allclose(
        translated_leaf_cloud.leaf_positions,
        np.array(expected_leaf_positions) * ureg.m,
    )

    # Shapes do not match: translated() raises
    with pytest.raises(ValueError):
        cloud.translated([[1, 0, 0]] * 2 * ureg.m)


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
    check_scene_element(cloud)
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
    check_scene_element(cloud)
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
    check_scene_element(cloud)
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
    check_scene_element(cloud)
    assert cloud.n_leaves() == 1000
    assert np.allclose(
        np.min(cloud.leaf_positions.m_as("m"), axis=0), [-15, -15, 0], atol=0.05
    )
    assert np.allclose(
        np.max(cloud.leaf_positions.m_as("m"), axis=0), [15, 15, 3], atol=0.05
    )


def test_leaf_cloud_from_file(mode_mono, tempfile_leaves):
    """Unit testing for :meth:`LeafCloud.from_file`."""

    # A LeafCloud instance can be loaded from a file on the hard drive
    leaf_cloud = LeafCloud.from_file(tempfile_leaves)
    assert len(leaf_cloud.leaf_positions) == 5
    assert np.allclose(leaf_cloud.leaf_radii, 0.1 * ureg.m)

    # Produced kernel dict is valid
    check_scene_element(leaf_cloud)


def test_leaf_cloud_kernel_dict(mode_mono):
    """Partial unit testing for :meth:`LeafCloud.kernel_dict`."""

    leaf_cloud_id = "leaf_cloud"
    leaf_cloud = LeafCloud(
        id=leaf_cloud_id,
        leaf_positions=[[0, 0, 0], [1, 1, 1]],
        leaf_orientations=[[1, 0, 0], [0, 1, 0]],
        leaf_radii=[0.1, 0.1],
        leaf_reflectance=0.5,
        leaf_transmittance=0.5,
    )

    template, params = traverse(leaf_cloud)

    # The BSDF is bilambertian with the parameters we initially set
    kernel_dict = template.render(ctx=KernelContext())
    assert kernel_dict[f"bsdf_{leaf_cloud_id}"] == {
        "type": "bilambertian",
        "reflectance": {"type": "uniform", "value": 0.5},
        "transmittance": {"type": "uniform", "value": 0.5},
    }

    # Leaves are disks
    for shape_key in [f"{leaf_cloud_id}_leaf_0", f"{leaf_cloud_id}_leaf_1"]:
        assert kernel_dict[shape_key]["type"] == "disk"

    # Kernel dict is valid
    check_scene_element(leaf_cloud)


def test_surface_area(mode_mono):
    """Unit testing for :meth:`LeafCloud.surface_area`."""
    cloud = LeafCloud(
        leaf_positions=[[0, 0, 0], [1, 1, 1]],
        leaf_orientations=[[1, 0, 0], [0, 1, 0]],
        leaf_radii=[0.1, 0.1],
    )

    assert cloud.surface_area() == (2.0 * np.pi * 0.1**2) * (ureg.m**2)
