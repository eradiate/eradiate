import numpy as np
import os
import pytest

from eradiate.scenes.biosphere import HomogeneousDiscreteCanopy, LeafCloud, FloatingSpheresCanopy, RealZoomInCanopy
from eradiate import unit_registry as ureg


@pytest.fixture(scope="module")
def tempfile_leaves():
    with open("./tempfile_leaves.txt", "w") as tf:
        tf.write("0.100 8.864 9.040 1.878 -0.314 0.025 0.949\n")
        tf.write("0.100 9.539 -10.463 0.627 0.489 -0.276 0.828\n")
        tf.write("0.100 -2.274 -9.204 0.797 0.618 0.184 0.764\n")
        tf.write("0.100 -9.957 -4.971 0.719 -0.066 0.100 0.993\n")
        tf.write("0.100 5.339 9.153 0.500 0.073 -0.294 0.953\n")
    yield
    os.remove("./tempfile_leaves.txt")


@pytest.fixture(scope="module")
def tempfile_spheres():
    with open("./tempfile_spheres.txt", "w") as tf:
        tf.write("8.864 9.040 1.878\n")
        tf.write("0.100 9.539 -10.463\n")
        tf.write("0.100 -2.274 -9.204\n")
        tf.write("0.100 -9.957 -4.971\n")
        tf.write("0.100 5.339 9.153\n")
    yield
    os.remove("./tempfile_spheres.txt")


@pytest.mark.slow
def test_leaf_cloud_instantiate_parameters(mode_mono):
    """Test the instantiation of the leaf cloud class from parameters"""

    cloud = LeafCloud.from_parameters()

    # create a valid kernel dict by default
    assert cloud.kernel_dict() is not None

    # default parameters are used correctly
    cloud2 = LeafCloud.from_parameters(size=[0, 0, 0], leaf_radius=0.25)
    assert cloud2.n_leaves() == 4000
    max_x = 0,
    max_y = 0,
    max_z = 0
    for pos in cloud2.leaf_positions:
        pos = pos.m_as(ureg.m)
        if pos[0] > max_x:
            max_x = pos[0]
        if pos[1] > max_y:
            max_y = pos[1]
        if pos[2] > max_z:
            max_z = pos[2]

    # these values are empirical
    expected_size = [8.08, 8.08, 15.28]

    assert np.allclose([max_x, max_y, max_z], expected_size, atol=1.e-1)

    # size takes precedence over number of leaves
    cloud3 = LeafCloud.from_parameters(size=[10, 10, 10]*ureg.m, lai=0.5)
    assert cloud3.n_leaves() == 1591
    max_x = 0,
    max_y = 0,
    max_z = 0
    for pos in cloud3.leaf_positions:
        pos = pos.m_as(ureg.m)
        if pos[0] > max_x:
            max_x = pos[0]
        if pos[1] > max_y:
            max_y = pos[1]
        if pos[2] > max_z:
            max_z = pos[2]

    # the canopy extends from -size/2. to size/2. for the x and y dimensions.
    expected_size = [5., 5., 10.]
    assert np.allclose([max_x, max_y, max_z], expected_size, atol=1.e-1)


def test_leaf_cloud_instantiate_file(mode_mono, tempfile_leaves):
    """Test instantiation of a leaf cloud from a file"""

    cloud = LeafCloud.from_file("./tempfile_leaves.txt", 0.5, 0.5)

    assert len(cloud.leaf_positions) == 5
    assert cloud.leaf_radius == 0.1*ureg.m


@pytest.mark.slow
def test_homogeneous_discrete_canopy_instantiate(mode_mono, tempfile_leaves):
    """Test dictionary instantiation of the canopy."""
    canopy_dict = {
        "size": [1, 1, 1] * ureg.m,
        "leaf_cloud": {
            "fname_leaves": "./tempfile_leaves.txt",
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5
        }
    }

    canopy = HomogeneousDiscreteCanopy.from_dict(canopy_dict)

    # this canopy should only have one leaf cloud element inside it
    assert len(canopy.leaf_cloud_specs) == 1
    spec, positions = canopy.leaf_cloud_specs[0]

    # there must be a LeafCloud object, positioned at the coordinate origin
    assert isinstance(spec, LeafCloud)
    assert np.allclose(positions[0], [0, 0, 0])

    # instantiate the canopy from parameters
    canopy_dict = {
        "size": [1, 1, 1] * ureg.m,
        "leaf_cloud": {
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
            "shape_type": "cube",
            "lai": 3.0,
            "n_leaves": 50000
        }
    }

    canopy = HomogeneousDiscreteCanopy.from_dict(canopy_dict)

    # this canopy should only have one leaf cloud element inside it
    assert len(canopy.leaf_cloud_specs) == 1
    spec, positions = canopy.leaf_cloud_specs[0]

    # there must be a LeafCloud object, positioned at the coordinate origin
    assert isinstance(spec, LeafCloud)
    assert np.allclose(positions[0], [0, 0, 0])

    # parameter instantiation must require 'cube' type
    broken_dict = {
        "size": [1, 1, 1] * ureg.m,
        "leaf_cloud": {
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
            "shape_type": "sphere",
            "lai": 3.0,
            "n_leaves": 50000
        }
    }
    with pytest.raises(ValueError):
        canopy = HomogeneousDiscreteCanopy.from_dict(broken_dict)


@pytest.mark.slow
def test_floating_sphere_canopy_instantiate(mode_mono, tempfile_leaves, tempfile_spheres):
    canopy_dict = {
        "leaf_cloud": {
            "fname_leaves": "./tempfile_leaves.txt",
            "fname_positions": "./tempfile_spheres.txt",
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
        },
        "size": [1, 1, 1] * ureg.m
    }

    canopy = FloatingSpheresCanopy.from_dict(canopy_dict)

    # this canopy should have one leaf cloud element with 5 positions inside it
    assert len(canopy.leaf_cloud_specs) == 1

    spec, positions = canopy.leaf_cloud_specs[0]
    assert isinstance(spec, LeafCloud)
    assert len(positions) == 5

    # instantiate the canopy from parameters
    canopy_dict = {
        "size": [1, 1, 1] * ureg.m,
        "leaf_cloud": {
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
            "fname_positions": "./tempfile_spheres.txt",
            "shape_type": "sphere",
            "sphere_radius": 1.0,
            "lai": 3.0,
            "n_leaves": 5000
        }
    }

    canopy = FloatingSpheresCanopy.from_dict(canopy_dict)

    # this canopy should have one leaf cloud element with 5 positions inside it
    assert len(canopy.leaf_cloud_specs) == 1

    spec, positions = canopy.leaf_cloud_specs[0]
    assert isinstance(spec, LeafCloud)
    assert len(positions) == 5

    # parameter instantiation must require 'cube' type
    broken_dict = {
        "size": [1, 1, 1] * ureg.m,
        "leaf_cloud": {
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
            "fname_positions": "./tempfile_spheres.txt",
            "shape_type": "cube",
            "lai": 3.0,
            "n_leaves": 50000
        }
    }
    with pytest.raises(ValueError):
        canopy = FloatingSpheresCanopy.from_dict(broken_dict)


@pytest.mark.slow
def test_real_zoom_in_canopy_instantiate(mode_mono, tempfile_leaves, tempfile_spheres):
    # instantiate the canopy from files and see it does not fail
    canopy_dict = {
        "spherical_leaf_cloud": {
            "fname_leaves": "./tempfile_leaves.txt",
            "fname_positions": "./tempfile_spheres.txt",
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
        },
        "cylindrical_leaf_cloud": {
            "fname_leaves": "./tempfile_leaves.txt",
            "fname_positions": "./tempfile_spheres.txt",
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
        },
        "size": [1, 1, 1] * ureg.m
    }

    canopy = RealZoomInCanopy.from_dict(canopy_dict)

    # this canopy should have one leaf cloud element with 5 positions inside it
    assert len(canopy.leaf_cloud_specs) == 2

    for (spec, positions)in  canopy.leaf_cloud_specs:
        assert isinstance(spec, LeafCloud)
        assert len(positions) == 5

    # instantiate the canopy from parameters
    canopy_dict = {
        "size": [1, 1, 1] * ureg.m,
        "spherical_leaf_cloud": {
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
            "fname_positions": "./tempfile_spheres.txt",
            "shape_type": "sphere",
            "sphere_radius": 1.0,
            "lai": 3.0,
            "n_leaves": 5000
        },
        "cylindrical_leaf_cloud": {
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
            "fname_positions": "./tempfile_spheres.txt",
            "shape_type": "cylinder",
            "cylinder_radius": 1.0,
            "cylinder_height": 2.0,
            "lai": 3.0,
            "n_leaves": 5000
        }
    }

    canopy = RealZoomInCanopy.from_dict(canopy_dict)

    # this canopy should have one leaf cloud element with 5 positions inside it
    assert len(canopy.leaf_cloud_specs) == 2

    for (spec, positions)in  canopy.leaf_cloud_specs:
        assert isinstance(spec, LeafCloud)
        assert len(positions) == 5

    # parameter instantiation must require 'cube' type
    broken_dict1 = {
        "size": [1, 1, 1] * ureg.m,
        "spherical_leaf_cloud": {
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
            "fname_positions": "./tempfile_spheres.txt",
            "shape_type": "cube",
            "lai": 3.0,
            "n_leaves": 50000
        },
        "cylindrical_leaf_cloud": {
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
            "fname_positions": "./tempfile_spheres.txt",
            "shape_type": "cylinder",
            "lai": 3.0,
            "n_leaves": 50000
        }
    }
    broken_dict2 = {
        "size": [1, 1, 1] * ureg.m,
        "spherical_leaf_cloud": {
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
            "fname_positions": "./tempfile_spheres.txt",
            "shape_type": "sphere",
            "lai": 3.0,
            "n_leaves": 50000
        },
        "cylindrical_leaf_cloud": {
            "leaf_transmittance": 0.5,
            "leaf_reflectance": 0.5,
            "fname_positions": "./tempfile_spheres.txt",
            "shape_type": "cube",
            "lai": 3.0,
            "n_leaves": 50000
        }
    }
    with pytest.raises(ValueError):
        canopy = RealZoomInCanopy.from_dict(broken_dict1)
    with pytest.raises(ValueError):
        canopy = RealZoomInCanopy.from_dict(broken_dict2)
