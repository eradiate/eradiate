import mitsuba as mi
import numpy as np
import pytest

from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.grid import PlaneParallelGridCoords
from eradiate.scenes.core import traverse
from eradiate.scenes.geometry import (
    PlaneParallelGeometry,
    SceneGeometry,
    SphericalShellGeometry,
)
from eradiate.scenes.phase import MultiPhaseFunction
from eradiate.spectral.index import MonoSpectralIndex
from eradiate.test_tools.types import check_scene_element


def mi_to_numpy(d):
    result = {}

    for k, v in d.items():
        if isinstance(v, mi.VolumeGrid):
            result[k] = np.array(v)
        elif isinstance(v, (mi.Transform4f, mi.ScalarTransform4f)):
            result[k] = np.array(v.matrix)
        else:
            result[k] = v

    return result


def assert_cmp_dict(value, expected):
    assert set(value.keys()) == set(expected.keys())

    for key in value:
        first_element = value[key]
        second_element = expected[key]

        if isinstance(first_element, np.ndarray) or isinstance(
            second_element, np.ndarray
        ):
            np.testing.assert_allclose(
                first_element, second_element, err_msg=f"Failing key: '{key}'"
            )
        else:
            np.testing.assert_equal(
                first_element, second_element, err_msg=f"failing key: '{key}'"
            )


def test_multi_construct_basic():
    """Scalar weights are stored as a 1D array; component/weight count mismatches raise."""
    phase = MultiPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
        weights=[0.25, 0.25, 0.5],
    )
    assert phase.weights.shape == (3,)  # Scalar weights are stored as a 1D array

    with pytest.raises(ValueError):
        MultiPhaseFunction(components=[{"type": "isotropic"}], weights=[1])

    with pytest.raises(ValueError):
        MultiPhaseFunction(
            components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
            weights=[0.25, 0.25],
        )
    with pytest.raises(ValueError):
        MultiPhaseFunction(
            components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
            weights=[[0.25, 0.25, 0.5]],
        )


def test_multi_construct_array(modes_all_double):
    """Per-component 1D array weights are stacked into a 2D (n_components, n_z) array."""
    phase = MultiPhaseFunction(
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
def test_multi_geometry(mode_mono, geometry):
    """Weight grids only get a to_world transform when a geometry is set, and it matches the geometry's."""
    weight_keys = ["weight_0.to_world", "weight_1.to_world", "weight_2.to_world"]

    phase = MultiPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
        weights=[0.25, 0.25, 0.5],
    )
    assert phase.geometry is None
    template = traverse(phase)[0]
    for weight_key in weight_keys:
        assert weight_key not in template

    geometry = SceneGeometry.convert(geometry)
    is_spherical = isinstance(geometry, SphericalShellGeometry)
    expected = np.array(
        (
            geometry.atmosphere_volume_to_world
            if is_spherical
            else geometry.grid.to_world
        ).matrix
    )
    phase = MultiPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}],
        weights=[0.25, 0.25, 0.5],
        geometry=geometry,
    )
    template = traverse(phase)[0]
    for weight_key in weight_keys:
        to_world = np.array(template[weight_key].matrix)
        np.testing.assert_allclose(to_world, expected)

        if is_spherical:
            inner_to_world = np.array(
                template[weight_key.replace(".to_world", ".volume.to_world")].matrix
            )
            np.testing.assert_allclose(
                inner_to_world, np.array(geometry.grid.to_world.matrix)
            )

    check_scene_element(phase, mi.PhaseFunction)


@pytest.mark.parametrize(
    "weights, use_small_grid",
    [
        ([0.25, 0.75], False),
        ([0.0, 1.0], False),
        ([1.0, 0.0], False),
        ([np.linspace(1.0, 0.0, 4), np.linspace(0.0, 1.0, 4)], True),
    ],
    ids=["scalar_even", "scalar_all_second", "scalar_all_first", "array"],
)
def test_multi_kernel_dict_2_components(mode_mono, weights, use_small_grid):
    """
    A 2-component blend produces the expected flat multiphase kernel dict,
    with one independent weight grid per component.
    """
    kwargs = {}
    if use_small_grid:
        grid = PlaneParallelGridCoords(
            edges_x=np.array([-1.0, 1.0]) * ureg.km,
            edges_y=np.array([-1.0, 1.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )
        kwargs["geometry"] = PlaneParallelGeometry(
            grid=grid, toa_altitude=grid.levels[-1]
        )

    phase = MultiPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}],
        weights=weights,
        **kwargs,
    )
    check_scene_element(phase, mi.PhaseFunction)

    template, params = traverse(phase)
    ctx = KernelContext()
    kernel_dict = mi_to_numpy(template.render(ctx, nested=False))

    expected = {
        "type": "multiphase",
        "use_mis": True,
        "phase_0.type": "isotropic",
        "phase_1.type": "rayleigh",
    }
    for i, w in enumerate(weights):
        expected[f"weight_{i}.type"] = "gridvolume"
        expected[f"weight_{i}.grid"] = np.reshape(w, (-1, 1, 1)).astype(np.float32)
        expected[f"weight_{i}.use_grid_bbox"] = True
        expected[f"weight_{i}.filter_type"] = "nearest"
        expected[f"weight_{i}.wrap_mode"] = "clamp"

    if "geometry" in kwargs:
        to_world = np.array(kwargs["geometry"].grid.to_world.matrix)
        expected["weight_0.to_world"] = to_world
        expected["weight_1.to_world"] = to_world

    assert_cmp_dict(kernel_dict, expected)
    assert set(params.keys()) == {"weight_0.data", "weight_1.data"}


def test_multi_kernel_dict_3_components(mode_mono):
    """
    A flat 3-component blend produces one independent weight grid per
    component, equal to the raw input weights directly (no pairwise
    conversion, unlike the former nested BlendPhaseFunction design).
    """
    weights = [0.3, 0.3, 0.4]
    phase = MultiPhaseFunction(
        components=[
            {"type": "hg", "g": -0.1},
            {"type": "rayleigh"},
            {"type": "hg", "g": 0.1},
        ],
        weights=weights,
    )
    check_scene_element(phase, mi.PhaseFunction)

    template, params = traverse(phase)
    kernel_dict = mi_to_numpy(template.render(KernelContext(), nested=False))

    expected = {
        "type": "multiphase",
        "use_mis": True,
        "phase_0.type": "hg",
        "phase_0.g": -0.1,
        "phase_1.type": "rayleigh",
        "phase_2.type": "hg",
        "phase_2.g": 0.1,
    }
    for i, w in enumerate(weights):
        expected[f"weight_{i}.type"] = "gridvolume"
        expected[f"weight_{i}.grid"] = np.reshape(w, (-1, 1, 1)).astype(np.float32)
        expected[f"weight_{i}.use_grid_bbox"] = True
        expected[f"weight_{i}.filter_type"] = "nearest"
        expected[f"weight_{i}.wrap_mode"] = "clamp"

    assert_cmp_dict(expected, kernel_dict)
    assert set(params.keys()) == {
        "weight_0.data",
        "weight_1.data",
        "weight_2.data",
        "phase_0.g",
        "phase_2.g",
    }


def _make_small_geometry():
    grid = PlaneParallelGridCoords(
        edges_x=np.array([-1.0, 0.0, 1.0]) * ureg.km,
        edges_y=np.array([-1.0, 0.0, 1.0]) * ureg.km,
        levels=np.array([0.0, 1.0, 2.0]) * ureg.km,
    )
    return PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])


def test_multi_full_3d_weights(mode_mono):
    """
    A genuine full 3D weight array (n_components, n_x, n_y, n_z) matching the
    geometry's grid exactly is passed through to the kernel dict unchanged.
    """
    geometry = _make_small_geometry()
    n_x, n_y, n_z = (
        geometry.grid.n_cells_x,
        geometry.grid.n_cells_y,
        geometry.grid.n_cells_z,
    )

    w0 = np.arange(n_x * n_y * n_z, dtype=np.float64).reshape(n_x, n_y, n_z)
    w1 = 10.0 - w0
    weights = np.stack([w0, w1], axis=0)

    phase = MultiPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}],
        weights=weights,
        geometry=geometry,
    )
    assert phase.weights.shape == (2, n_x, n_y, n_z)
    check_scene_element(phase, mi.PhaseFunction)

    template, _ = traverse(phase)
    rendered = mi_to_numpy(template.render(KernelContext(), nested=False))

    for i, w in enumerate([w0, w1]):
        np.testing.assert_allclose(rendered[f"weight_{i}.grid"], w.T)


def test_multi_partial_broadcast_weights_raises(mode_mono):
    """
    A per-component weight shape that only partially matches the grid (e.g.
    broadcasting along x only) is rejected: generate_gridvolume only accepts
    a scalar, an exact (n_z,) vertical profile, or an exact (n_x,n_y,n_z)
    match -- not arbitrary partial reductions.
    """
    geometry = _make_small_geometry()
    n_y, n_z = geometry.grid.n_cells_y, geometry.grid.n_cells_z

    w0 = np.arange(n_y * n_z, dtype=np.float64).reshape(1, n_y, n_z)
    w1 = 5.0 - w0
    weights = np.stack([w0, w1], axis=0)

    phase = MultiPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}],
        weights=weights,
        geometry=geometry,
    )
    template, _ = traverse(phase)
    with pytest.raises(ValueError, match="Invalid grid shape"):
        template.render(KernelContext(), nested=False)


@pytest.mark.parametrize(
    "weights, expected",
    [
        ([1.0, 1.0, 2.0], [0.25, 0.25, 0.5]),
        ([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]),
        (
            [np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])],
            np.array([[0.25, 0.5, 0.75], [0.75, 0.5, 0.25]]),
        ),
    ],
    ids=["unnormalized_scalar", "already_normalized_scalar", "array"],
)
def test_multi_normalized(weights, expected):
    """with_normalized_weights() rescales weights to sum to 1 per cell, keeping components/geometry."""
    components = [{"type": "isotropic"}, {"type": "rayleigh"}, {"type": "hg"}][
        : len(weights)
    ]
    phase = MultiPhaseFunction(components=components, weights=weights)
    ctx = KernelContext(si=MonoSpectralIndex(w=ureg.Quantity(1.0, "micron")))

    result = phase.with_normalized_weights(ctx)

    assert isinstance(result, MultiPhaseFunction)
    assert all(a is b for a, b in zip(result.components, phase.components))
    assert result.geometry is phase.geometry
    np.testing.assert_allclose(result.weights.sum(axis=0), 1.0)
    np.testing.assert_allclose(result.weights, expected, rtol=1e-6)


def test_multi_normalized_callable_weights():
    """with_normalized_weights() evaluates callable weights at the given context before normalizing."""
    phase = MultiPhaseFunction(
        components=[{"type": "isotropic"}, {"type": "rayleigh"}],
        weights=[lambda si: np.array([1.0, 3.0]), lambda si: np.array([3.0, 1.0])],
    )
    ctx = KernelContext(si=MonoSpectralIndex(w=ureg.Quantity(1.0, "micron")))

    result = phase.with_normalized_weights(ctx)

    np.testing.assert_allclose(result.weights.sum(axis=0), 1.0)
    np.testing.assert_allclose(result.weights, [[0.25, 0.75], [0.75, 0.25]], rtol=1e-6)
