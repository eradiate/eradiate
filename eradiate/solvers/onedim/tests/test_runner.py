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
    assert np.allclose(solver.run()["measure"], 1. / (2. * np.pi), rtol=1e-3)

    # Run simulation with array of vzas and vaas
    result = solver.run()["measure"]
    assert result.shape == (1, 4, 1)
    assert np.allclose(result, 1. / (2. * np.pi), rtol=1e-3)
