import numpy as np
import pytest

from eradiate.grid import CartesianGrid
from eradiate.scenes.atmosphere import ParticleStateComponent
from eradiate.scenes.geometry import PlaneParallelGeometry
from eradiate.scenes.phase import GriddedParticlePhaseFunction
from eradiate.spectral.index import MonoSpectralIndex
from eradiate.units import unit_registry as ureg

from ._psd_helpers import make_state_profile_dataset, make_state_properties_dataset

MASS_CONCENTRATION = 2.0 * ureg("g/m^3")
SSA = 0.8

# Each case: properties state dimensions, profile state variables (excluding
# the amount), state_variables mapping
CASES = {
    "hum": (
        {"hum": np.array([0.0, 50.0, 90.0]) * ureg.percent},
        {"hum": 45.0 * ureg.percent},
        {},
    ),
    "two_state": (
        {
            "a": np.array([1.0, 2.0]) * ureg.m,
            "b": np.array([0.1, 0.2, 0.3]) * ureg.s,
        },
        {"a": 1.5 * ureg.m, "b": 0.25 * ureg.s},
        {},
    ),
    "renamed": (
        {"hum": np.array([0.0, 50.0, 90.0]) * ureg.percent},
        {"rh": 45.0 * ureg.percent},
        {"hum": "rh"},
    ),
}


def make_geometry() -> PlaneParallelGeometry:
    grid = CartesianGrid(
        edges_x=np.array([-1.0, 0.0, 1.0]) * ureg.km,
        edges_y=np.array([-1.0, 1.0]) * ureg.km,
        levels=np.array([0.0, 1.0, 2.0]) * ureg.km,
    )
    return PlaneParallelGeometry(grid=grid, toa_altitude=grid.levels[-1])


def make_component(state_dims, profile_vars, state_variables, **kwargs):
    geometry = make_geometry()
    profile = make_state_profile_dataset(
        geometry.grid, {**profile_vars, "mass_concentration": MASS_CONCENTRATION}
    )
    return ParticleStateComponent(
        profile=profile,
        properties=make_state_properties_dataset(state_dims, ssa_value=SSA),
        state_variables=state_variables,
        geometry=geometry,
        **kwargs,
    )


@pytest.fixture(params=list(CASES), ids=list(CASES))
def case(request):
    state_dims, profile_vars, state_variables = CASES[request.param]
    component = make_component(state_dims, profile_vars, state_variables)
    return component, state_dims, profile_vars


class TestParticleStateComponent:
    def test_state_dims(self, case):
        component, state_dims, _ = case
        assert component.state_dims == tuple(state_dims)

    def test_amount_selected_automatically(self, case):
        component, _, _ = case
        assert component._amount == "mass_concentration"

    def test_eval_sigma_t_albedo(self, case):
        """Extinction and albedo match the properties interpolated at the
        profile's state values; extinction is linear in the state variables
        by construction, so interpolation is exact."""
        component, _, profile_vars = case
        si = MonoSpectralIndex(w=1.0 * ureg.micron)
        ext = 1.0 + sum(q.m for q in profile_vars.values())  # km^-1 / (g/m^3)

        np.testing.assert_allclose(
            component.eval_sigma_t(si).m_as("1/km"),
            ext * MASS_CONCENTRATION.m_as("g/m^3"),
        )
        np.testing.assert_allclose(component.eval_albedo(si).m_as(""), SSA)

    def test_phase(self, case):
        """The phase function is a GriddedParticlePhaseFunction with one
        interpolation slot per state dimension."""
        component, state_dims, profile_vars = case
        phase = component.phase
        assert isinstance(phase, GriddedParticlePhaseFunction)

        grids = list(state_dims.values())
        values = list(profile_vars.values())
        np.testing.assert_allclose(phase.grid_1.m_as(grids[0].u), grids[0].m)
        np.testing.assert_allclose(phase.volume_1.m_as(values[0].u), values[0].m)
        if len(grids) == 1:
            assert phase.grid_2 is None and phase.volume_2 is None
        else:
            np.testing.assert_allclose(phase.grid_2.m_as(grids[1].u), grids[1].m)
            np.testing.assert_allclose(phase.volume_2.m_as(values[1].u), values[1].m)

    def test_eval_phase_data(self, case):
        """_eval_phase_data has one row per state-dimension grid point."""
        component, _, _ = case
        data = component._eval_phase_data(MonoSpectralIndex(w=1.0 * ureg.micron))
        state_shape = component.properties.state_shape
        n_pts = int(np.prod(state_shape))

        assert data["mu"].shape[0] == n_pts
        assert data["phase"].shape[:2] == (n_pts, 4)
        assert data["ext"].shape == state_shape
        assert data["ssa"].shape == state_shape

    def test_template_phase(self, case, mode_mono):
        """The kernel dict maps state variables to the plugin's slots."""
        component, _, _ = case
        template = component._template_phase
        assert template["type"] == "particlephase"
        for key in [
            "r_eff_volume.grid",
            "v_eff_volume.grid",
            "r_eff_grid",
            "v_eff_grid",
        ]:
            assert key in template

    @pytest.mark.parametrize(
        "state_dims",
        [
            {},
            {
                "a": np.array([1.0, 2.0]) * ureg.m,
                "b": np.array([1.0, 2.0]) * ureg.m,
                "c": np.array([1.0, 2.0]) * ureg.m,
            },
        ],
        ids=["zero", "three"],
    )
    def test_wrong_number_of_state_dims_raises(self, state_dims):
        profile_vars = dict.fromkeys(state_dims, 1.0 * ureg.m)
        with pytest.raises(ValueError, match="exactly 1 or 2 state dimensions"):
            make_component(state_dims, profile_vars, {})

    def test_missing_profile_variable_raises(self):
        state_dims, profile_vars, _ = CASES["renamed"]
        with pytest.raises(
            ValueError, match=r"no variable for state dimension\(s\) \['hum'\]"
        ):
            make_component(state_dims, profile_vars, {})
