import numpy as np
import pint

from eradiate.scenes.biosphere import HomogeneousDiscreteCanopy
ureg = pint.UnitRegistry()


def test_homogeneous_discrete_canopy_instantiate(mode_mono):
    """Test instantiation for the homogeneous discrete canopy
    in a cuboid volume."""

    # assert that the values that are possibly computed during init
    # have the expected value
    cloud = HomogeneousDiscreteCanopy.from_parameters()
    assert cloud.n_leaves == 4000
    assert np.allclose(cloud.size.to(ureg.m).magnitude,
                       [16, 16, 15], rtol=1.e-6)
    assert cloud.hdo == 1 * ureg.m

    cloud2 = HomogeneousDiscreteCanopy.from_parameters(size=[10, 10, 10])
    assert cloud2.n_leaves == 1527
    assert np.allclose(cloud2.size.to(ureg.m).magnitude,
                       [10, 10, 10], rtol=1.e-6)
    assert np.allclose(cloud2.hdo.to(ureg.m).magnitude,
                       0.868232846, rtol=1.e-6)

#TODO: Add a render test, that renders a scene with a leaf cloud
# and compares with a reference.
# Needed for that: The radiancemeter with tunable target area