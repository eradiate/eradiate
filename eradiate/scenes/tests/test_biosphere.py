import numpy as np

from eradiate.scenes.biosphere import HomogeneousDiscreteCanopy
from eradiate import unit_registry as ureg


def test_homogeneous_discrete_canopy_instantiate(mode_mono):
    """Test instantiation for the homogeneous discrete canopy
    in a cuboid volume."""

    cloud = HomogeneousDiscreteCanopy.from_parameters()

    # create a valid kernel dict by default
    assert cloud.kernel_dict() is not None

    # default parameters are used correctly
    cloud2 = HomogeneousDiscreteCanopy.from_parameters(size=[0, 0, 0], leaf_radius=0.25)
    assert cloud2._n_leaves == 4000
    assert np.allclose(cloud2.size,
                       [16.17, 16.15, 15.28] * ureg.m, rtol=1.e-2)

    # size takes precedence over number of leaves
    cloud3 = HomogeneousDiscreteCanopy.from_parameters(size=[10, 10, 10])
    assert cloud3._n_leaves == 9549
    assert np.allclose(cloud3.size, [10, 10, 10] * ureg.m, rtol=1.e-3)
