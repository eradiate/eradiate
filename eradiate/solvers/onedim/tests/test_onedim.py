from decimal import Decimal
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