import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.grid import CartesianGrid, SphericalGrid
from eradiate.scenes.atmosphere import ParticleProfile


def make_profile_dataset(
    reff, veff, mass_concentration, x_edges, y_edges, z_edges, xy_units="km"
) -> xr.Dataset:
    """
    Build a dense Ppr-Core v1 profile dataset. ``reff``/``veff``/
    ``mass_concentration`` must be broadcastable to shape
    ``(len(z_edges) - 1, len(y_edges) - 1, len(x_edges) - 1)``.
    """
    shape = (len(z_edges) - 1, len(y_edges) - 1, len(x_edges) - 1)
    return xr.Dataset(
        data_vars={
            "reff": (
                ["z", "y", "x"],
                np.broadcast_to(reff, shape).copy(),
                {"units": "micron"},
            ),
            "veff": (
                ["z", "y", "x"],
                np.broadcast_to(veff, shape).copy(),
                {"units": "dimensionless"},
            ),
            "mass_concentration": (
                ["z", "y", "x"],
                np.broadcast_to(mass_concentration, shape).copy(),
                {"units": "g/m^3"},
            ),
        },
        coords={
            "x_edges": (["x_edges"], np.asarray(x_edges), {"units": xy_units}),
            "y_edges": (["y_edges"], np.asarray(y_edges), {"units": xy_units}),
            "z_edges": (["z_edges"], np.asarray(z_edges), {"units": "km"}),
        },
    )


class TestConstruction:
    def test_valid_dataset(self):
        """A dataset with all required variables constructs successfully."""
        ds = make_profile_dataset(
            reff=10.0,
            veff=0.1,
            mass_concentration=1e-3,
            x_edges=[0.0, 1.0],
            y_edges=[0.0, 1.0],
            z_edges=[0.0, 1.0],
        )
        ParticleProfile(data=ds)

    def test_arbitrary_variables(self):
        """State variables are not prescribed."""
        ds = make_profile_dataset(
            reff=10.0,
            veff=0.1,
            mass_concentration=1e-3,
            x_edges=[0.0, 1.0],
            y_edges=[0.0, 1.0],
            z_edges=[0.0, 1.0],
        ).rename({"mass_concentration": "number_concentration"})
        ds = ds.drop_vars(["reff", "veff"])
        profile = ParticleProfile(data=ds)
        assert profile.variables == ("number_concentration",)

    @pytest.mark.parametrize("missing", ["x_edges", "y_edges", "z_edges"])
    def test_missing_variable_raises(self, missing):
        """Constructing from a dataset missing an edge coordinate raises."""
        ds = make_profile_dataset(
            reff=10.0,
            veff=0.1,
            mass_concentration=1e-3,
            x_edges=[0.0, 1.0],
            y_edges=[0.0, 1.0],
            z_edges=[0.0, 1.0],
        ).drop_vars(missing)
        with pytest.raises(ValueError, match="missing required coordinate"):
            ParticleProfile(data=ds)

    def test_no_state_variable_raises(self):
        """Constructing from a dataset with only edge coordinates raises."""
        ds = make_profile_dataset(
            reff=10.0,
            veff=0.1,
            mass_concentration=1e-3,
            x_edges=[0.0, 1.0],
            y_edges=[0.0, 1.0],
            z_edges=[0.0, 1.0],
        ).drop_vars(["reff", "veff", "mass_concentration"])
        with pytest.raises(ValueError, match="no state variable"):
            ParticleProfile(data=ds)

    def test_wrong_dims_raises(self):
        """Constructing from a dataset with a state variable not on (z, y, x) raises."""
        ds = make_profile_dataset(
            reff=10.0,
            veff=0.1,
            mass_concentration=1e-3,
            x_edges=[0.0, 1.0],
            y_edges=[0.0, 1.0],
            z_edges=[0.0, 1.0],
        )
        ds["reff"] = ds["reff"].transpose("x", "y", "z")
        with pytest.raises(ValueError, match=r"offending variable.*reff"):
            ParticleProfile(data=ds)

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_wrong_edges_size_raises(self, axis):
        """Constructing from a dataset whose edges don't have len(dim) + 1 values raises."""
        ds = make_profile_dataset(
            reff=10.0,
            veff=0.1,
            mass_concentration=1e-3,
            x_edges=[0.0, 1.0],
            y_edges=[0.0, 1.0],
            z_edges=[0.0, 1.0],
        )
        name = f"{axis}_edges"
        ds = ds.drop_vars(name)
        ds[name] = ([name], np.array([0.0, 1.0, 2.0]), {"units": "km"})
        with pytest.raises(ValueError, match=f"'{name}' must have dims"):
            ParticleProfile(data=ds)

    @pytest.mark.parametrize(
        "name, values, attrs, match",
        [
            ("x_edges", [0.0, 1.0], {}, "'x_edges' has no units"),
            ("z_edges", [1.0, 0.0], {"units": "km"}, "'z_edges' must be strictly"),
            ("z_edges", [0.0, 1.0], {"units": "rad"}, "'z_edges' must have length"),
            ("y_edges", [0.0, 1.0], {"units": "rad"}, "both have length units"),
            ("x_edges", [0.0, 1.0], {"units": "s"}, "both have length units"),
        ],
        ids=["no_units", "decreasing", "z_angle", "xy_mixed", "xy_time"],
    )
    def test_invalid_edges_raises(self, name, values, attrs, match):
        """Constructing from a dataset with invalid edge coordinates raises."""
        ds = make_profile_dataset(
            reff=10.0,
            veff=0.1,
            mass_concentration=1e-3,
            x_edges=[0.0, 1.0],
            y_edges=[0.0, 1.0],
            z_edges=[0.0, 1.0],
        )
        ds = ds.assign_coords({name: ([name], np.array(values), attrs)})
        with pytest.raises(ValueError, match=match):
            ParticleProfile(data=ds)


class TestVariablesAndGetitem:
    @pytest.fixture
    def profile(self) -> ParticleProfile:
        return ParticleProfile(
            data=make_profile_dataset(
                reff=10.0,
                veff=0.1,
                mass_concentration=1e-3,
                x_edges=[0.0, 1.0],
                y_edges=[0.0, 1.0],
                z_edges=[0.0, 1.0],
            )
        )

    def test_variables_excludes_edges(self, profile):
        """variables lists state variables only, not edge coordinates."""
        assert set(profile.variables) == {"reff", "veff", "mass_concentration"}

    def test_getitem_cached(self, profile):
        """profile[name] returns the same cached quantity on every access."""
        for name in profile.variables:
            assert profile[name] is profile[name]
        assert profile["reff"].units == ureg.micron

    def test_getitem_rejects_edge_coordinate(self, profile):
        """profile[name] raises for an edge (x/y/z_edges) coordinate."""
        with pytest.raises(KeyError, match="x_edges"):
            profile["x_edges"]

    def test_getitem_rejects_unknown_variable(self, profile):
        """profile[name] raises for a name that isn't in the profile at all."""
        with pytest.raises(KeyError, match="nonexistent"):
            profile["nonexistent"]


class TestResample:
    @pytest.fixture
    def profile(self) -> ParticleProfile:
        reff = [[[10.0, 11.0], [0.0, 0.0]], [[0.0, 0.0], [12.0, 13.0]]]
        mass_concentration = [
            [[1e-3, 2e-3], [0.0, 0.0]],
            [[0.0, 0.0], [3e-3, 4e-3]],
        ]
        return ParticleProfile(
            data=make_profile_dataset(
                reff=reff,
                veff=0.1,
                mass_concentration=mass_concentration,
                x_edges=[0.0, 1.0, 2.0],
                y_edges=[0.0, 1.0, 2.0],
                z_edges=[0.0, 1.0, 4.0],
            )
        )

    @pytest.fixture
    def grid(self) -> CartesianGrid:
        return CartesianGrid(
            x_edges=np.array([0.0, 1.0, 2.0]) * ureg.km,
            y_edges=np.array([0.0, 1.0, 2.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )

    def test_containing_layer_z(self, profile, grid):
        """resample picks the source layer containing each target layer centre."""
        result = profile.resample(grid)
        reff = result.data["reff"].values
        mass_concentration = result.data["mass_concentration"].values

        np.testing.assert_allclose(reff[0, 0, 0], 10.0)
        assert mass_concentration[0, 0, 0] > 0
        assert mass_concentration[1, 0, 0] == 0
        assert mass_concentration[2, 0, 0] == 0
        assert mass_concentration[3, 0, 0] == 0

        assert mass_concentration[0, 1, 0] == 0
        np.testing.assert_allclose(reff[1, 1, 0], 12.0)
        assert mass_concentration[1, 1, 0] > 0
        np.testing.assert_allclose(reff[2, 1, 0], 12.0)
        assert mass_concentration[2, 1, 0] > 0
        np.testing.assert_allclose(reff[3, 1, 0], 12.0)
        assert mass_concentration[3, 1, 0] > 0

    def test_z_extent(self):
        """Target layers inside a single-layer profile are filled, others zeroed."""
        profile = ParticleProfile(
            data=make_profile_dataset(
                reff=10.0,
                veff=0.1,
                mass_concentration=1e-3,
                x_edges=[0.0, 1.0],
                y_edges=[0.0, 1.0],
                z_edges=[1.0, 2.0],
            )
        )
        grid = CartesianGrid(
            x_edges=np.array([0.0, 1.0]) * ureg.km,
            y_edges=np.array([0.0, 1.0]) * ureg.km,
            levels=np.linspace(0.0, 3.0, 7) * ureg.km,
        )
        result = profile.resample(grid)
        np.testing.assert_array_equal(
            result.data["mass_concentration"].values[:, 0, 0],
            [0.0, 0.0, 1e-3, 1e-3, 0.0, 0.0],
        )

    def test_extra_variable(self, profile, grid):
        """Variables beyond reff/veff/mass_concentration are accessible and resampled."""
        ds = profile.data.copy()
        ds["number_concentration"] = (
            ["z", "y", "x"],
            np.full((2, 2, 2), 5.0),
            {"units": "cm^-3"},
        )
        extended = ParticleProfile(data=ds)
        assert extended["number_concentration"].units == ureg("cm^-3").units

        result = extended.resample(grid)
        np.testing.assert_allclose(
            result["number_concentration"].m_as("cm^-3"), np.full((4, 2, 2), 5.0)
        )

    def test_output_is_regular_on_all_axes(self, profile, grid):
        """The resampled profile's x/y/z edges are all evenly spaced."""
        result = profile.resample(grid)
        for name in ("x_edges", "y_edges", "z_edges"):
            values = result.data[name].values
            step = np.diff(values)
            np.testing.assert_allclose(step, step[0])

    def test_output_dataset_valid(self, profile, grid):
        """The resampled dataset is itself a valid ParticleProfile input."""
        result = profile.resample(grid)
        ParticleProfile(data=result.data)

    @pytest.mark.parametrize("axis", ["x_edges", "y_edges"])
    def test_irregular_source_raises(self, profile, grid, axis):
        """resample rejects a profile with an irregular x_edges/y_edges axis."""
        ds = profile.data.copy()
        ds[axis] = ([axis], np.array([0.0, 1.0, 3.0]), {"units": "km"})
        irregular = ParticleProfile(data=ds)
        with pytest.raises(ValueError, match="do not match the target"):
            irregular.resample(grid)

    def test_extent_mismatch_raises(self, profile):
        """resample rejects a target grid whose extent doesn't match the profile's."""
        mismatched_grid = CartesianGrid(
            x_edges=np.array([0.0, 1.0, 2.0, 3.0]) * ureg.km,
            y_edges=np.array([0.0, 1.0, 2.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )
        with pytest.raises(ValueError, match="do not match the target"):
            profile.resample(mismatched_grid)

    def test_resolution_mismatch_raises(self, profile):
        """resample rejects a target grid whose resolution doesn't match the profile's."""
        mismatched_grid = CartesianGrid(
            x_edges=np.array([0.0, 2.0]) * ureg.km,
            y_edges=np.array([0.0, 1.0, 2.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )
        with pytest.raises(ValueError, match="do not match the target"):
            profile.resample(mismatched_grid)

    def test_spherical_shell_dispatch(self):
        """resample dispatches to the SphericalGrid implementation."""
        profile = ParticleProfile(
            data=make_profile_dataset(
                reff=10.0,
                veff=0.1,
                mass_concentration=1e-3,
                x_edges=np.linspace(0.0, 2 * np.pi, 3),
                y_edges=np.linspace(0.0, np.pi, 3),
                z_edges=[0.0, 1.0],
                xy_units="radian",
            )
        )

        grid = SphericalGrid(
            levels=np.array([0.0, 1.0]) * ureg.km,
            azimuths=np.linspace(0.0, 2 * np.pi, 3) * ureg.radian,
            colatitudes=np.linspace(0.0, np.pi, 3) * ureg.radian,
        )
        result = profile.resample(grid)
        assert result.data["reff"].shape == (1, 2, 2)
