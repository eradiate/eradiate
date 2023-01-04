import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.scenes.core import traverse
from eradiate.scenes.phase import BlendPhaseFunction
from eradiate.test_tools.types import check_scene_element


def mi_to_numpy(d):
    result = {}

    for k, v in d.items():
        if isinstance(v, mi.VolumeGrid):
            result[k] = np.array(v)
        elif isinstance(v, mi.Transform4f):
            result[k] = np.array(v.matrix)
        else:
            result[k] = v

    return result


def assert_cmp_dict(value, expected):
    """
    Helper function to compare two dictionaries.
    """
    # Keys should be identical
    assert set(value.keys()) == set(expected.keys())

    for key in value.keys():
        first_element = value[key]
        second_element = expected[key]

        if isinstance(first_element, np.ndarray) or isinstance(
            second_element, np.ndarray
        ):
            # Handle special case of NumPy arrays
            np.testing.assert_allclose(
                first_element, second_element, err_msg=f"Failing key: '{key}'"
            )

        else:
            # Other types are compared for proper equality
            np.testing.assert_equal(
                first_element, second_element, err_msg=f"failing key: '{key}'"
            )

        assert True


def test_blend_phase_construct_basic():
    """
    Test instantiation with simple weight values.
    """
    # Constructing with reasonable settings succeeds
    phase = BlendPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
        weights=[0.25, 0.25, 0.5],
    )
    assert phase.weights.shape == (3,)  # Scalar weights are stored as a 1D array

    # Improper number of components raises
    with pytest.raises(ValueError):
        BlendPhaseFunction(
            components=[{"type": "isotropic"}],
            weights=[1],
        )

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


def test_blend_phase_construct_array(modes_all_double):
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
    assert phase.weights.shape == (3, 10)  # Array weights are stored as a 2D array


@pytest.mark.parametrize(
    "weights, expected",
    [
        (
            [[0, 1, 2], [1, 0, 2]],
            [[1, 0, 0.5]],
        ),
        (
            [[0, 1, 2], [1, 0, 2], [0.5, 0.5, 2]],
            [[1, 1 / 3, 2 / 3], [1 / 3, 1, 0.5]],
        ),
    ],
    ids=[
        "array_2cmp",
        "array_3cmp",
    ],
)
def test_blend_phase_weights(mode_mono, weights, expected):
    phase = BlendPhaseFunction(
        components=[{"type": "isotropic"}] * len(weights),
        weights=weights,
        bbox=[[0, 0, 0], [1, 1, 1]],
    )
    np.testing.assert_allclose(
        phase.eval_conditional_weights(SpectralContext.new()), expected
    )


def test_blend_phase_bbox(mode_mono):
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
    np.testing.assert_allclose(
        np.array(phase._gridvolume_transform().matrix),
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )

    # Nested BlendPhaseFunction objects must have the same bbox
    phase = BlendPhaseFunction(
        components=[
            {"type": "isotropic"},
            {
                "type": "blend_phase",
                "components": [{"type": "rayleigh"}, {"type": "hg"}],
                "weights": [0.25, 0.75],
            },
        ],
        weights=[0.25, 0.75],
        bbox=([0, 0, 0] * ureg.m, [1, 1, 1] * ureg.m),
    )

    for comp in phase.components:
        if isinstance(comp, BlendPhaseFunction):
            np.testing.assert_allclose(
                np.array(comp._gridvolume_transform().matrix),
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": [0.0, 0.0]},
        {"weights": [1.0, 0.0]},
        {"weights": [0.0, 1.0]},
        {"weights": [0.5, 0.5]},
        {"weights": [0.3, 0.7]},
        {
            "weights": [
                np.array([1.0, 0.7, 0.5, 0.3, 0.0]),
                np.array([0.0, 0.3, 0.5, 0.7, 1.0]),
            ],
            "bbox": [[0, 0, 0], [1, 1, 1]],
        },
    ],
    ids=[
        "scalar0",
        "scalar1",
        "scalar2",
        "scalar3",
        "scalar4",
        "array",
    ],
)
def test_blend_phase_kernel_dict_2_components(mode_mono, kwargs):
    """
    Blendphase with 2 components produces correct kernel dict and can be loaded.
    """
    phase = BlendPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}],
        **kwargs,
    )
    check_scene_element(phase, mi.PhaseFunction)

    template, params = traverse(phase)
    ctx = KernelDictContext()
    kernel_dict = mi_to_numpy(template.render(ctx, nested=False))

    # Check that the kernel dict is correct
    expected = {
        "type": "blendphase",
        "phase_0.type": "isotropic",
        "phase_1.type": "rayleigh",
        "weight.type": "gridvolume",
        "weight.grid": np.reshape(
            phase.eval_conditional_weights(ctx.spectral_ctx), (-1, 1, 1)
        ),
    }
    if "bbox" in kwargs:
        expected["weight.to_world"] = np.identity(4)
    assert_cmp_dict(kernel_dict, expected)

    # Check that the parameter map is correct
    assert set(params.keys()) == {"weight.data"}


@pytest.mark.parametrize(
    "kwargs, expected_mi_weights",
    [
        ({"weights": [0.0, 0.0, 0.0]}, [0.0, 0.0]),
        ({"weights": [1.0, 0.0, 0.0]}, [0.0, 0.0]),
        ({"weights": [0.0, 1.0, 0.0]}, [1.0, 0.0]),
        ({"weights": [0.0, 0.0, 1.0]}, [1.0, 1.0]),
        ({"weights": [0.5, 0.5, 0.0]}, [0.5, 0.0]),
        ({"weights": [0.0, 0.5, 0.5]}, [1.0, 0.5]),
        ({"weights": [0.5, 0.0, 0.5]}, [0.5, 1.0]),
        ({"weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]}, [2.0 / 3.0, 0.5]),
        ({"weights": [0.3, 0.3, 0.4]}, [0.7, 4.0 / 7.0]),
        (
            {
                "weights": [
                    np.array([1.0, 0.6, 0.2, 0.2, 0.0]),
                    np.array([0.0, 0.2, 0.6, 0.2, 0.5]),
                    np.array([0.0, 0.2, 0.2, 0.6, 0.5]),
                ],
                "bbox": [[0, 0, 0], [1, 1, 1]],
            },
            [
                [0.0, 0.4, 0.8, 0.8, 1.0],
                [0.0, 0.5, 0.25, 0.75, 0.5],
            ],
        ),
    ],
    ids=[
        "scalar0",
        "scalar1",
        "scalar2",
        "scalar3",
        "scalar4",
        "scalar5",
        "scalar6",
        "scalar7",
        "scalar8",
        "array",
    ],
)
def test_blend_phase_kernel_dict_3_components(mode_mono, kwargs, expected_mi_weights):
    """
    Blendphase with 3 components produces correct kernel dict and can be loaded.
    """
    phase = BlendPhaseFunction(
        components=[
            {"type": "hg", "g": -0.1},
            {"type": "rayleigh"},
            {"type": "hg", "g": 0.1},
        ],
        **kwargs,
    )
    check_scene_element(phase, mi.PhaseFunction)

    template, params = traverse(phase)
    kernel_dict = mi_to_numpy(template.render(KernelDictContext(), nested=False))

    # Array weights
    expected = {
        "type": "blendphase",
        "weight.type": "gridvolume",
        "weight.grid": np.reshape(expected_mi_weights[0], (-1, 1, 1)),
        "phase_0.type": "hg",
        "phase_0.g": -0.1,
        "phase_1.type": "blendphase",
        "phase_1.weight.type": "gridvolume",
        "phase_1.weight.grid": np.reshape(expected_mi_weights[1], (-1, 1, 1)),
        "phase_1.phase_0.type": "rayleigh",
        "phase_1.phase_1.type": "hg",
        "phase_1.phase_1.g": 0.1,
    }

    if "bbox" in kwargs:
        expected["weight.to_world"] = np.identity(4)
        expected["phase_1.weight.to_world"] = np.identity(4)

    assert_cmp_dict(expected, kernel_dict)

    # Check that the parameter map is correct
    assert set(params.keys()) == {
        "weight.data",
        "phase_0.g",
        "phase_1.weight.data",
        "phase_1.phase_1.g",
    }
