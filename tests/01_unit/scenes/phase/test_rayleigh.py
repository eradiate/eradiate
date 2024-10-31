import mitsuba as mi
import numpy as np

import eradiate
from eradiate.scenes.core import traverse
from eradiate.scenes.geometry import PlaneParallelGeometry, SphericalShellGeometry
from eradiate.scenes.phase import RayleighPhaseFunction
from eradiate.spectral import CKDSpectralIndex
from eradiate.test_tools.types import check_scene_element
from eradiate.units import unit_registry as ureg


def test_rayleigh(modes_all_double):
    phase = RayleighPhaseFunction()
    check_scene_element(phase, mi.PhaseFunction)

    kdict, _ = traverse(phase)
    if eradiate.mode().is_polarized:
        assert kdict.data["type"] == "rayleigh_polarized"
        assert "depolarization.type" in kdict.data


def test_rayleigh_plane_parallel(modes_all_double):
    phase = RayleighPhaseFunction(geometry=PlaneParallelGeometry())
    check_scene_element(phase, mi.PhaseFunction)


def test_rayleigh_spherical_shell(modes_all_double):
    phase = RayleighPhaseFunction(geometry=SphericalShellGeometry())
    check_scene_element(phase, mi.PhaseFunction)


def test_depolarization(mode_ckd_polarized):
    si = CKDSpectralIndex(w=550 * ureg.nm)

    phase = RayleighPhaseFunction()
    depol = phase.eval_depolarization_factor(si)
    kdict, _ = traverse(phase)
    check_scene_element(phase, mi.PhaseFunction)
    assert isinstance(depol, np.ndarray)
    assert np.shape(depol) == (1,)

    phase = RayleighPhaseFunction(depolarization=0.1)
    depol = phase.eval_depolarization_factor(si)
    kdict, _ = traverse(phase)
    check_scene_element(phase, mi.PhaseFunction)
    assert isinstance(depol, np.ndarray)
    assert np.shape(depol) == (1,)

    phase = RayleighPhaseFunction(depolarization=[0.5, 0.1, 0.3])
    depol = phase.eval_depolarization_factor(si)
    kdict, _ = traverse(phase)
    check_scene_element(phase, mi.PhaseFunction)
    assert isinstance(depol, np.ndarray)
    assert np.shape(depol) == (3,)
