from pprint import pprint

import eradiate.kernel
from eradiate.solvers.onedim import *
from eradiate.solvers.onedim import _make_distant, _make_default_scene


def test_make_sensor(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Without units (check if created dict is valid)
    dict_sensor = _make_distant(45., 0., 32)
    assert load_dict(dict_sensor) is not None

    # With degrees
    assert _make_distant(45. * ureg.deg, 0., 32) == dict_sensor

    # With radian
    assert _make_distant(0.25 * np.pi * ureg.rad, 0., 32) == dict_sensor


def test_onedimsolver(variant_scalar_mono):
    # Construct
    solver = OneDimSolver()
    assert solver.dict_scene == _make_default_scene()

    # Run simulation with default parameters (and check if result array is cast to scalar)
    assert solver.run() == 0.1591796875

    # Run simulation with array of vzas (and check if result array is squeezed)
    result = solver.run(vza=np.linspace(0, 90, 91), spp=32)
    assert result.shape == (91,)
    assert np.all(result == 0.1591796875)

    # Run simulation with array of vzas and vaas
    result = solver.run(vza=np.linspace(0, 90, 11),
                        vaa=np.linspace(0, 180, 11),
                        spp=32)
    assert result.shape == (11, 11)
    assert np.all(result == 0.1591796875)


def test_add_rayleigh_atmosphere(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    dict_scene = _make_default_scene()
    assert load_dict(add_rayleigh_atmosphere(dict_scene)) is not None
