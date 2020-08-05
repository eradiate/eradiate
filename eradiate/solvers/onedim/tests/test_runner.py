import numpy as np

from eradiate.solvers.onedim.runner import OneDimRunner


def test_onedimsolver(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # Construct
    solver = OneDimRunner()
    assert solver.kernel_dict == OneDimRunner.DEFAULT_KERNEL_DICT

    # Check if default scene is valid
    assert load_dict(solver.kernel_dict) is not None

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
