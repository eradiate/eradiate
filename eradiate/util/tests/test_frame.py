from eradiate.util.frame import *

def test_spherical_to_cartesian():
    r = 2.
    theta = np.deg2rad(30)
    phi = np.deg2rad(0)
    x, y, z = spherical_to_cartesian(r, theta, phi)
    assert np.allclose([x, y, z], [1, 0, np.sqrt(3)])

    r = 2.
    theta = np.deg2rad(60)
    phi = np.deg2rad(30)
    x, y, z = spherical_to_cartesian(r, theta, phi)
    assert np.allclose([x, y, z], [3. / 2., np.sqrt(3) / 2., 1])
