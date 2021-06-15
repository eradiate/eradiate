import numpy as np
import pytest

from eradiate.scenes.atmosphere._mesh_util import (
    conciliate,
    extract_layer_mesh,
    find_closest,
)


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


def test_conciliate_one_layer() -> None:
    """Returns expected mesh."""
    z_mol = np.linspace(0.0, 4.0, 3)
    z_par = [np.linspace(0.8, 1.2, 5)]
    expected_z = np.linspace(0.0, 4.0, 41)

    z = conciliate(z_mol, z_par)
    assert np.allclose(z, expected_z)


def test_conciliate_one_layer_2() -> None:
    """Returns expected mesh."""
    z_mol = np.linspace(0.0, 86.0, 87)
    z_par = [np.linspace(0.5, 1.5, 2)]
    expected_z = np.linspace(0.0, 86.0, 173)

    z = conciliate(z_mol, z_par)
    assert np.allclose(z, expected_z)


def test_conciliate_layer_mesh_is_not_degraded() -> None:
    """Particle altitude mesh is not degraded."""
    z_mol = np.linspace(0.0, 4.0, 3)
    z_par = [np.linspace(0.0, 2.001, 21)]
    z = conciliate(z_mol, z_par)
    new_z_par = extract_layer_mesh(z=z, bottom=z_par[0].min(), top=z_par[0].max())
    assert len(new_z_par) >= len(z_par[0])


def test_conciliate_layers_meshes_are_not_degraded() -> None:
    """Particle layers altitude meshes are not degraded."""
    z_mol = np.linspace(0.0, 4.0, 3)
    z_par = [np.linspace(0.0, 2.001, 21), np.linspace(1.999, 2.5, 12)]
    z = conciliate(z_mol, z_par)
    new_z_par = [
        extract_layer_mesh(z=z, bottom=z_par[i].min(), top=z_par[i].max())
        for i in range(len(z_par))
    ]
    for i in range(len(z_par)):
        assert len(new_z_par[i]) >= len(z_par[i])


def test_conciliate_atol() -> None:
    """Tolerance parameter is applied."""
    z_mol = np.linspace(0.0, 4.0, 3)
    bottom = 0.0
    top = 2.01
    atol = 0.005
    z_par = [np.linspace(bottom, top, 21)]
    z = conciliate(z_mol, z_par, atol=atol)
    new_z_par = extract_layer_mesh(z=z, bottom=z_par[0].min(), top=z_par[0].max())
    new_bottom = new_z_par.min()
    new_top = new_z_par.max()
    assert np.isclose(bottom, new_bottom, atol=atol)
    assert np.isclose(top, new_top, atol=atol)


def test_conciliate_atol_multiple_layers() -> None:
    """Tolerance parameter is applied to all layers."""
    z_mol = np.linspace(0.0, 4.0, 3)
    bottoms = [0.0, 0.95]
    tops = [2.01, 2.651]
    atol = 0.005
    z_par = [np.linspace(bottom, top, 21) for bottom, top in zip(bottoms, tops)]
    z = conciliate(z_mol, z_par, atol=atol)
    new_z_par = [
        extract_layer_mesh(z=z, bottom=z_par[i].min(), top=z_par[i].max())
        for i in range(len(z_par))
    ]
    new_bottoms = [z.min() for z in new_z_par]
    new_tops = [z.max() for z in new_z_par]
    assert np.allclose(bottoms, new_bottoms, atol=atol)
    assert np.allclose(tops, new_tops, atol=atol)


def test_extract_layer_mesh() -> None:
    """Returns expected sub mesh."""
    z = np.linspace(0, 10, 11)
    z_sub = extract_layer_mesh(z=z, bottom=2.0, top=4.0)
    assert np.allclose(z_sub, [2.0, 3.0, 4.0])


def test_extract_layer_mesh_find_closest() -> None:
    """Find closes values when exact match does not exists."""
    z = np.linspace(0, 10, 11)
    z_sub = extract_layer_mesh(z=z, bottom=2.1, top=3.5)
    assert np.allclose(z_sub, [2.0, 3.0, 4.0])


def test_extract_layer_mesh_invalid_bottom() -> None:
    """Raises when invalid bottom is provided."""
    z = np.linspace(0, 10, 11)
    bottom_invalid = [-1, 11]
    for bottom_invalid_value in bottom_invalid:
        with pytest.raises(ValueError):
            extract_layer_mesh(z=z, bottom=bottom_invalid_value, top=5.0)


def test_extract_layer_mesh_invalid_top() -> None:
    """Raises when invalid top is provided."""
    z = np.linspace(0, 10, 11)
    top_invalid = [-1, 11]
    for top_invalid_value in top_invalid:
        with pytest.raises(ValueError):
            extract_layer_mesh(z=z, bottom=5.0, top=top_invalid_value)


def test_extract_layer_mesh_bottom_larger_than_top() -> None:
    """Raises when bottom is larger than top."""
    z = np.linspace(0, 10, 11)
    with pytest.raises(ValueError):
        extract_layer_mesh(z=z, bottom=2.0, top=1.0)
