import typing as t

import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import NodeSceneElement, traverse
from eradiate.scenes.phase._blend import BlendPhaseFunction
from eradiate.test_tools.types import check_type
from eradiate.util.misc import flatten


def assert_cmp_dict(first, second):
    """
    Helper function to compare two dictionaries.
    """
    if first.keys() != second.keys():
        assert False, (
            f"Keys are not the same: expected {set(first.keys())}, "
            f"got {set(second.keys())} "
            f"(diff: {set(first.keys()) - set(second.keys())})"
        )

    for key in first.keys():
        first_element = first[key]
        second_element = second[key]

        # Convert to numpy arrays if necessary
        if isinstance(first_element, mi.VolumeGrid):
            first_element = np.array(first_element)
        elif isinstance(first_element, mi.Transform4f):
            first_element = np.array(first_element.matrix)

        if isinstance(second_element, mi.VolumeGrid):
            second_element = np.array(second_element)
        elif isinstance(second_element, mi.Transform4f):
            second_element = np.array(second_element.matrix)

        if isinstance(first_element, np.ndarray) or isinstance(
            second_element, np.ndarray
        ):
            if not np.allclose(first_element, second_element):
                assert (
                    False
                ), f"Different values for {key}: expected {first_element}, got {second_element}"
        else:
            if first_element != second_element:
                assert False, f"Different values for {key}"

        assert True


def test_blend_phase_type():
    check_type(
        BlendPhaseFunction,
        expected_mro=[BlendPhaseFunction, NodeSceneElement],
        expected_slots=[],
    )


def test_blend_phase_construct_basic():
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

    # Weights are normalised
    assert np.allclose(phase.weights, 1 / 3)


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
    assert np.allclose(
        np.array(phase._gridvolume_transform().matrix),
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
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
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            )


@pytest.mark.parametrize(
    "weights, bbox",
    [
        ((0.0, 0.0), None),
        ((1.0, 0.0), None),
        ((0.0, 1.0), None),
        ((0.5, 0.5), None),
        ((0.3, 0.7), None),
        (
            (
                np.array([1.0, 0.7, 0.5, 0.3, 0.0]),
                np.array([0.0, 0.3, 0.5, 0.7, 1.0]),
            ),
            ([0, 0, 0], [1, 1, 1]),
        ),
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
def test_blend_phase_kernel_dict_2_components(
    mode_mono, weights: t.Tuple[float, float], bbox
):
    """
    Blendphase with 2 components produces correct kernel dict and can be loaded.
    """
    phase = BlendPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}],
        weights=weights,
        bbox=bbox,
    )
    template, _ = traverse(phase)

    if bbox is None:
        # Scalar weights
        assert_cmp_dict(
            template.data,
            {
                "type": "blendphase",
                "phase1.type": "isotropic",
                "phase2.type": "rayleigh",
                "weight": weights[1],
            },
        )
    else:
        # Array weights
        assert_cmp_dict(
            template.data,
            {
                "type": "blendphase",
                "phase1.type": "isotropic",
                "phase2.type": "rayleigh",
                "weight.type": "gridvolume",
                "weight.grid": np.reshape(weights[1], (-1, 1, 1)),
                "weight.to_world": np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                ),
            },
        )

    # Check plugin weights
    phase_plugin = mi.load_dict(template.render(KernelDictContext()))
    assert isinstance(phase_plugin, mi.PhaseFunction)
    if isinstance(weights[0], np.ndarray):
        weight_plugin = mi.traverse(phase_plugin)["weight.data"]
        assert np.allclose(weight_plugin, np.reshape(weights[1], (-1, 1, 1, 1)))


@pytest.mark.parametrize(
    "weights, bbox",
    [
        ((0.0, 0.0, 0.0), None),
        ((1.0, 0.0, 0.0), None),
        ((0.0, 1.0, 0.0), None),
        ((0.0, 0.0, 1.0), None),
        ((0.5, 0.5, 0.0), None),
        ((0.0, 0.5, 0.5), None),
        ((0.5, 0.0, 0.5), None),
        ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), None),
        ((0.3, 0.3, 0.4), None),
        (
            (
                np.array([1.0, 0.6, 0.2, 0.2, 0.0]),
                np.array([0.0, 0.2, 0.6, 0.2, 0.5]),
                np.array([0.0, 0.2, 0.2, 0.6, 0.5]),
            ),
            ([0, 0, 0], [1, 1, 1]),
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
def test_blend_phase_kernel_dict_3_components(
    mode_mono, weights: t.Tuple[float, float, float], bbox
):
    """
    Blendphase with 3 components produces correct kernel dict and can be loaded.
    """
    phase = BlendPhaseFunction(
        components=[
            {"type": "isotropic"},
            {"type": "rayleigh"},
            {"type": "hg", "g": 0.1},
        ],
        weights=weights,
        bbox=bbox,
    )
    nested_weight = np.divide(
        weights[2],
        weights[1] + weights[2],
        where=weights[1] + weights[2] != 0.0,
        out=np.zeros_like(weights[2]),
    )  # This is the predicted weight of the nested blended component
    if bbox is None:  # No bbox means scalar weights
        nested_weight = float(nested_weight)

    template, _ = traverse(phase)
    kernel_dict = template.render(KernelDictContext())

    if bbox is None:
        # Scalar weights
        assert_cmp_dict(
            flatten(kernel_dict),
            {
                "type": "blendphase",
                "weight": weights[1] + weights[2],
                "phase1.type": "isotropic",
                "phase2.type": "blendphase",
                "phase2.weight": nested_weight,
                "phase2.phase1.type": "rayleigh",
                "phase2.phase2.type": "hg",
                "phase2.phase2.g": 0.1,
            },
        )
    else:
        # Array weights
        assert_cmp_dict(
            flatten(kernel_dict),
            {
                "type": "blendphase",
                "weight.type": "gridvolume",
                "weight.grid": np.reshape(weights[1] + weights[2], (-1, 1, 1)),
                "weight.to_world": np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                ),
                "phase1.type": "isotropic",
                "phase2.type": "blendphase",
                "phase2.weight.type": "gridvolume",
                "phase2.weight.grid": np.reshape(nested_weight, (-1, 1, 1)),
                "phase2.weight.to_world": np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                ),
                "phase2.phase1.type": "rayleigh",
                "phase2.phase2.type": "hg",
                "phase2.phase2.g": 0.1,
            },
        )

    # Check plugin weights
    phase_plugin = mi.load_dict(template.render(KernelDictContext()))
    assert isinstance(phase_plugin, mi.PhaseFunction)
    if isinstance(weights[0], np.ndarray):
        weight_plugin = mi.traverse(phase_plugin)["weight.data"]
        print(weight_plugin)
        assert np.allclose(
            weight_plugin, np.reshape(weights[1] + weights[2], (-1, 1, 1, 1))
        )

    # Also check the parameter table: g must be present (we already checked that
    # it evaluates to 0.1 above)
    assert set(phase.params.keys()) == {"phase2.phase2.g"}
