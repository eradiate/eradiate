import numpy as np
import pytest

from eradiate.solvers.onedim import OneDimSolver


def test_onedimsolver(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Construct
    solver = OneDimSolver()
    assert solver.scene_dict == OneDimSolver.DEFAULT_SCENE_DICT

    # Check if default scene is valid
    assert load_dict(solver.scene_dict) is not None

    # Run simulation with default parameters (and check if result array is cast to scalar)
    assert np.allclose(solver.run(), 1. / (2. * np.pi), rtol=1e-3)

    # Run simulation with array of vzas (and check if result array is squeezed)
    result = solver.run(vza=np.linspace(0, 90, 91), spp=32)
    assert result.shape == (91,)
    assert np.allclose(result, 1. / (2. * np.pi), rtol=1e-3)

    # Run simulation with array of vzas and vaas
    result = solver.run(vza=np.linspace(0, 90, 11),
                        vaa=np.linspace(0, 180, 11),
                        spp=32)
    assert result.shape == (11, 11)
    assert np.allclose(result, 1. / (2. * np.pi), rtol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("scene_size", [10 ** i for i in range(1, 9)])
def test_onedimsolver_large_size(variant_scalar_mono_double, scene_size):
    from eradiate.scenes import SceneDict
    from eradiate.kernel.core import ScalarTransform4f, ScalarVector3f

    scene_dict = SceneDict({
        "type": "scene",
        "bsdf_surface": {
            "type": "diffuse",
            "reflectance": {"type": "uniform", "value": 0.5}
        },
        "surface": {
            "type": "rectangle",
            "to_world": ScalarTransform4f.scale(ScalarVector3f(scene_size, scene_size, 1)),
            "bsdf": {"type": "ref", "id": "bsdf_surface"}
        },
        "illumination": {
            "type": "directional",
            "direction": [0, 0, -1],
            "irradiance": {"type": "uniform", "value": 1.0}
        },
        "integrator": {"type": "path"}
    })

    solver = OneDimSolver(scene_dict)

    vza = np.linspace(0, 90, 10)
    vaa = np.linspace(0, 360, 37)

    result = solver.run(vza=vza, vaa=vaa, spp=1024)
    assert np.allclose(result, 1. / (2. * np.pi), rtol=1e-3)
