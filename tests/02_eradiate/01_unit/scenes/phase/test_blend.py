import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.kernel.gridvolume import read_binary_grid3d
from eradiate.scenes.phase._blend import BlendPhaseFunction
from eradiate.util.misc import onedict_value


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
    assert np.allclose(phase.weights, [0.25, 0.25, 0.5])

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
        np.array(phase._gridvolume_transform().matrix),
        [
            [1, 0, 0, -0.5],
            [0, 1, 0, -0.5],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )

    # Nested BlendPhaseFunction objects must have the same bbox
    for comp in phase.components:
        if isinstance(comp, BlendPhaseFunction):
            assert np.allclose(
                np.array(comp._gridvolume_transform().matrix),
                [
                    [1, 0, 0, -0.5],
                    [0, 1, 0, -0.5],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            )


@pytest.mark.parametrize(
    "weights",
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.3, 0.7),
    ],
)
def test_blend_kernel_dict_2_components(mode_mono, weights):
    """
    Blendphase with 2 components produces correct kernel dict and can be loaded.
    """
    ctx = KernelDictContext()
    weight1, weight2 = weights
    phase = BlendPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}],
        weights=[weight1, weight2],
    )
    kernel_dict = phase.kernel_dict(ctx)
    phase_dict = onedict_value(kernel_dict.data)
    assert phase_dict["type"] == "blendphase"
    assert phase_dict["phase1"] == {"type": "isotropic"}
    assert phase_dict["phase2"] == {"type": "rayleigh"}
    assert phase_dict["weight"] == weight2
    assert isinstance(kernel_dict.load(), mi.PhaseFunction)


@pytest.mark.parametrize(
    "weights",
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.3, 0.3, 0.4),
    ],
)
def test_blend_kernel_dict_3_components(mode_mono, weights):
    """
    Blendphase with 3 components produces correct kernel dict and can be loaded.
    """
    ctx = KernelDictContext()
    weight1, weight2, weight3 = weights
    phase = BlendPhaseFunction(
        components=[
            {"type": "isotropic"},
            {"type": "rayleigh"},
            {"type": "hg", "g": 0.1},
        ],
        weights=[weight1, weight2, weight3],
    )
    kernel_dict = phase.kernel_dict(ctx)
    phase_dict = onedict_value(kernel_dict.data)
    assert phase_dict["type"] == "blendphase"
    assert phase_dict["weight"] == weight2 + weight3
    assert phase_dict["phase1"] == {"type": "isotropic"}
    assert phase_dict["phase2"]["type"] == "blendphase"
    nested_weight = float(
        np.divide(
            weight3,
            weight2 + weight3,
            where=weight2 + weight3 != 0.0,
            out=np.zeros_like(weight3),
        )
    )
    assert phase_dict["phase2"]["weight"] == nested_weight
    assert phase_dict["phase2"]["phase1"] == {"type": "rayleigh"}
    assert phase_dict["phase2"]["phase2"] == {"type": "hg", "g": 0.1}
    assert isinstance(kernel_dict.load(), mi.PhaseFunction)


def test_blend_array(modes_all_double):
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
    assert np.allclose(phase.weights, 1 / 3)

    # Kernel dict generation succeeds
    ctx = KernelDictContext()
    assert isinstance(phase.kernel_dict(ctx).load(), mi.PhaseFunction)


def test_blend_array_2_components(modes_all_double):
    """
    Test instantiation and kernel dict generation with an array of weights.
    """
    # Constructing using 1D arrays for weights succeeds
    weight1 = np.array([1.0, 0.7, 0.5, 0.3, 0.0])
    weight2 = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    phase = BlendPhaseFunction(
        components=[
            {"type": "isotropic"},
            {"type": "rayleigh"},
        ],
        weights=[weight1, weight2],
        bbox=[[0, 0, 0], [1, 1, 1]],
    )

    # Kernel dict generation succeeds
    ctx = KernelDictContext()
    phase_plugin = phase.kernel_dict(ctx).load()
    assert isinstance(phase_plugin, mi.PhaseFunction)

    # Weights are correct
    weight_plugin = mi.traverse(phase_plugin)["weight.data"]
    assert np.allclose(weight_plugin, weight2.reshape((-1, 1, 1, 1)))


def test_blend_array_3_components(modes_all_double):
    """
    Test instantiation and kernel dict generation with an array of weights.
    """
    # Constructing using 1D arrays for weights succeeds
    weight1 = np.array([1.0, 0.6, 0.2, 0.2, 0.0])
    weight2 = np.array([0.0, 0.2, 0.6, 0.2, 0.5])
    weight3 = np.array([0.0, 0.2, 0.2, 0.6, 0.5])
    phase = BlendPhaseFunction(
        components=[
            {"type": "rayleigh"},
            {"type": "isotropic"},
            {"type": "isotropic"},
        ],
        weights=[weight1, weight2, weight3],
        bbox=[[0, 0, 0], [1, 1, 1]],
    )

    # Kernel dict generation succeeds
    ctx = KernelDictContext()
    phase_plugin = phase.kernel_dict(ctx).load()
    assert isinstance(phase_plugin, mi.PhaseFunction)

    # Top-level weights are correct
    params = mi.traverse(phase_plugin)
    weight_plugin = params["weight.data"]
    assert np.allclose(weight_plugin, (weight2 + weight3).reshape((-1, 1, 1, 1)))

    # Nested weights are also correct
    weight_nested_plugin = params["phase_1.weight.data"]
    expected = np.divide(
        weight3, weight2 + weight3, where=weight2 + weight3 != 0.0, out=weight3
    )
    assert np.allclose(weight_nested_plugin, expected.reshape((-1, 1, 1, 1)))
