import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate._util import onedict_value
from eradiate.contexts import KernelDictContext
from eradiate.scenes.phase._blend import BlendPhaseFunction


def test_blend_construct_basic():
    """
    Test instantiation with simple weight values.
    """
    # Constructing with reasonable settings succeeds
    assert BlendPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
        weights=[0.25, 0.25, 0.5],
    )

    # Improper number of components raises
    with pytest.raises(ValueError):
        BlendPhaseFunction(
            components=[{"type": "isotropic"}],
            weights=[1],
        )

    # Weights are normalised
    phase = BlendPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
        weights=[1, 1, 2],
    )
    assert np.allclose([0.25, 0.25, 0.5], phase.weights)

    # Improper weight array shape raises
    with pytest.raises(ValueError):
        BlendPhaseFunction(
            components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
            weights=[0.25, 0.25],
        )
    with pytest.raises(ValueError):
        BlendPhaseFunction(
            components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
            weights=[[0.25, 0.25, 0.5]],
        )


def test_blend_bbox(mode_mono):
    """
    Test bounding box and grid volume transform.
    """
    # By default, no bounding box is defined
    phase = BlendPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
        weights=[0.25, 0.25, 0.5],
    )
    assert phase.bbox is None
    with pytest.raises(ValueError):
        phase._gridvolume_transform()

    # Appropriate bounding box setup works
    phase = BlendPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
        weights=[0.25, 0.25, 0.5],
        bbox=([0, 0, 0] * ureg.m, [1, 1, 1] * ureg.m),
    )
    assert np.allclose(
        [
            [1, 0, 0, -0.5],
            [0, 1, 0, -0.5],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        np.array(phase._gridvolume_transform().matrix),
    )

    # Nested BlendPhaseFunction objects must have the same bbox
    for comp in phase.components:
        if isinstance(comp, BlendPhaseFunction):
            assert np.allclose(
                [
                    [1, 0, 0, -0.5],
                    [0, 1, 0, -0.5],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                np.array(comp._gridvolume_transform().matrix),
            )

def test_blend_kernel_dict(mode_mono):
    ctx = KernelDictContext()

    # With 2 components
    phase = BlendPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}],
        weights=[0.3, 0.7],
    )
    kernel_dict = phase.kernel_dict(ctx)
    phase_dict = onedict_value(kernel_dict.data)
    assert phase_dict["type"] == "blendphase"
    assert phase_dict["phase1"] == {"type": "isotropic"}
    assert phase_dict["phase2"] == {"type": "rayleigh"}
    assert phase_dict["weight"] == 0.3
    assert kernel_dict.load()

    # With 3 components
    phase = BlendPhaseFunction(
        components=[
            {"type": "isotropic"},
            {"type": "rayleigh"},
            {"type": "hg", "g": 0.1},
        ],
        weights=[0.3, 0.3, 0.4],
    )
    kernel_dict = phase.kernel_dict(ctx)
    phase_dict = onedict_value(kernel_dict.data)
    assert phase_dict["type"] == "blendphase"
    assert phase_dict["weight"] == 0.3
    assert phase_dict["phase1"] == {"type": "isotropic"}
    assert phase_dict["phase2"]["type"] == "blendphase"
    assert phase_dict["phase2"]["weight"] == 0.3 / 0.7
    assert phase_dict["phase2"]["phase1"] == {"type": "rayleigh"}
    assert phase_dict["phase2"]["phase2"] == {"type": "hg", "g": 0.1}
    assert kernel_dict.load()


def test_blend_array(modes_all):
    """
    Test instantiation and kernel dict generation with an array of weights.
    """
    # Constructing using 1D arrays for weights succeeds
    phase = BlendPhaseFunction(
        components=[
            {"type": "isotropic"},
            {"type": "isotropic"},
            {"type": "isotropic"},
        ],
        weights=[np.ones((10,)), np.ones((10,)), np.ones((10,))],
        bbox=[[0, 0, 0], [1, 1, 1]],
    )

    # Weights are normalised
    assert np.allclose(1 / 3, phase.weights)

    # Kernel dict generation succeeds
    ctx = KernelDictContext()
    assert phase.kernel_dict(ctx).load()
