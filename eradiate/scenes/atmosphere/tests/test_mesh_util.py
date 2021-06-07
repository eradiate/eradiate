import numpy as np

from eradiate.scenes.atmosphere._mesh_util import find_closest, merge


def test_find_closest() -> None:
    """Find individual targets."""
    x = np.arange(0, 11)
    targets = [2.2, 7.8, 0.0, 10.0]
    expected_values = [2, 8, 0, 10]
    for target, expected_value in zip(targets, expected_values):
        assert find_closest(x, target) == expected_value


def test_find_closest_multiple_targets() -> None:
    """Finds multiple targets."""
    x = np.arange(0, 11)
    targets = [2.2, 7.8, 0.0, 10.0]
    expected_values = [2, 8, 0, 10]
    assert all(find_closest(x, targets) == expected_values)


def test_merge_one_layer() -> None:
    """Returns expected mesh."""
    z_mol = np.linspace(0.0, 4.0, 3)
    z_par = [np.linspace(0.8, 1.2, 5)]
    expected_z_merged = np.linspace(0.0, 4.0, 41)

    z_merged, _ = merge(z_mol, z_par)
    assert np.allclose(z_merged, expected_z_merged)


def test_merge_layer_mesh_is_not_degraded() -> None:
    """Particle altitude mesh is not degraded."""
    z_mol = np.linspace(0.0, 4.0, 3)
    z_par = [np.linspace(0.0, 2.001, 21)]
    _, new_z_par = merge(z_mol, z_par)
    assert len(new_z_par[0]) >= len(z_par[0])


def test_merge_layers_meshes_are_not_degraded() -> None:
    """Particle layers altitude meshes are not degraded."""
    z_mol = np.linspace(0.0, 4.0, 3)
    z_par = [np.linspace(0.0, 2.001, 21), np.linspace(1.999, 2.5, 12)]
    _, new_z_par = merge(z_mol, z_par)
    for i in range(len(z_par)):
        assert len(new_z_par[i]) >= len(z_par[i])


def test_merge_atol() -> None:
    """Tolerance parameter is applied."""
    z_mol = np.linspace(0.0, 4.0, 3)
    bottom = 0.0
    top = 2.01
    atol = 0.005
    z_par = [np.linspace(bottom, top, 21)]
    z_merged, new_z_par = merge(z_mol, z_par, atol=atol)
    new_bottom = new_z_par[0].min()
    new_top = new_z_par[0].max()
    assert np.isclose(bottom, new_bottom, atol=atol)
    assert np.isclose(top, new_top, atol=atol)


def test_merge_atol_multiple_layers() -> None:
    """Tolerance parameter is applied to all layers."""
    z_mol = np.linspace(0.0, 4.0, 3)
    bottoms = [0.0, 0.95]
    tops = [2.01, 2.651]
    atol = 0.005
    z_par = [np.linspace(bottom, top, 21) for bottom, top in zip(bottoms, tops)]
    z_merged, new_z_par = merge(z_mol, z_par, atol=atol)
    new_bottoms = [z.min() for z in new_z_par]
    new_tops = [z.max() for z in new_z_par]
    assert np.allclose(bottoms, new_bottoms, atol=atol)
    assert np.allclose(tops, new_tops, atol=atol)
