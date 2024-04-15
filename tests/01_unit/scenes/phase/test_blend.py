import mitsuba as mi
import numpy as np
import pytest

from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.scenes.core import traverse
from eradiate.scenes.geometry import SceneGeometry
from eradiate.scenes.phase import BlendPhaseFunction
from eradiate.spectral.index import SpectralIndex
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
        geometry="plane_parallel",
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
        geometry="plane_parallel",
    )
    np.testing.assert_allclose(
        phase.eval_conditional_weights(SpectralIndex.new()), expected
    )


@pytest.mark.parametrize(
    "geometry",
    [
        {
            "type": "plane_parallel",
            "width": 1.0 * ureg.m,
            "ground_altitude": 0.0 * ureg.m,
            "toa_altitude": 1.0 * ureg.m,
        },
        {
            "type": "spherical_shell",
            "planet_radius": 1.0 * ureg.m,
            "ground_altitude": 0.0 * ureg.m,
            "toa_altitude": 1.0 * ureg.m,
        },
    ],
    ids=["plane_parallel", "spherical_shell"],
)
def test_blend_phase_geometry(mode_mono, geometry):
    """
    Test geometry and grid volume transform.
    """
    # Defining no geometry is accepted; in that case, no transform is applied
    # to the volume data
    phase = BlendPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
        weights=[0.25, 0.25, 0.5],
    )
    assert phase.geometry is None
    template = traverse(phase)[0]
    assert "weight.to_world" not in template

    # Appropriate geometry setup works
    geometry = SceneGeometry.convert(geometry)
    expected = np.array(geometry.atmosphere_volume_to_world.matrix)
    phase = BlendPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
        weights=[0.25, 0.25, 0.5],
        geometry=geometry,
    )
    template = traverse(phase)[0]
    to_world = np.array(template["weight.to_world"].matrix)
    np.testing.assert_allclose(to_world, expected)
    check_scene_element(phase, mi.PhaseFunction)

    # Nested BlendPhaseFunction objects must have the same geometry
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
        geometry=geometry,
    )

    for comp in phase.components:
        if isinstance(comp, BlendPhaseFunction):
            template = traverse(comp)[0]
            to_world = np.array(template["weight.to_world"].matrix)
            np.testing.assert_allclose(to_world, expected)

    check_scene_element(phase, mi.PhaseFunction)


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
            "geometry": "plane_parallel",
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
    ctx = KernelContext()
    kernel_dict = mi_to_numpy(template.render(ctx, nested=False))

    # Check that the kernel dict is correct
    expected = {
        "type": "blendphase",
        "phase_0.type": "isotropic",
        "phase_1.type": "rayleigh",
        "weight.type": "gridvolume",
        "weight.grid": np.reshape(phase.eval_conditional_weights(ctx.si), (-1, 1, 1)),
    }
    if "geometry" in kwargs:
        geometry = SceneGeometry.convert(kwargs["geometry"])
        expected["weight.to_world"] = np.array(
            geometry.atmosphere_volume_to_world.matrix
        )
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
                "geometry": "plane_parallel",
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
    kernel_dict = mi_to_numpy(template.render(KernelContext(), nested=False))

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

    if "geometry" in kwargs:
        geometry = SceneGeometry.convert(kwargs["geometry"])
        expected["weight.to_world"] = np.array(
            geometry.atmosphere_volume_to_world.matrix
        )
        expected["phase_1.weight.to_world"] = expected["weight.to_world"].copy()

    assert_cmp_dict(expected, kernel_dict)

    # Check that the parameter map is correct
    assert set(params.keys()) == {
        "weight.data",
        "phase_0.g",
        "phase_1.weight.data",
        "phase_1.phase_1.g",
    }
