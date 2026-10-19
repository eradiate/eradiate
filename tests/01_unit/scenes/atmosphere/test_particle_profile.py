import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.grid import CartesianGrid, SphericalGrid
from eradiate.scenes.atmosphere import ParticleProfile


def make_profile_dataset(
    reff, veff, mass_concentration, x_edges, y_edges, z_edges
) -> xr.Dataset:
    """
    Build a dense Pfd-Core v1 profile dataset. ``reff``/``veff``/
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
            "x_edges": (["x_edges"], np.asarray(x_edges), {"units": "km"}),
            "y_edges": (["y_edges"], np.asarray(y_edges), {"units": "km"}),
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
        """Field variables are not prescribed."""
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
        """Constructing from a dataset missing a structural variable raises."""
        ds = make_profile_dataset(
            reff=10.0,
            veff=0.1,
            mass_concentration=1e-3,
            x_edges=[0.0, 1.0],
            y_edges=[0.0, 1.0],
            z_edges=[0.0, 1.0],
        ).drop_vars(missing)
        with pytest.raises(ValueError, match="missing required variable"):
            ParticleProfile(data=ds)

    def test_no_field_variable_raises(self):
        """Constructing from a dataset with only structural variables raises."""
        ds = make_profile_dataset(
            reff=10.0,
            veff=0.1,
            mass_concentration=1e-3,
            x_edges=[0.0, 1.0],
            y_edges=[0.0, 1.0],
            z_edges=[0.0, 1.0],
        ).drop_vars(["reff", "veff", "mass_concentration"])
        with pytest.raises(ValueError, match="no variable besides"):
            ParticleProfile(data=ds)

    def test_wrong_dims_raises(self):
        """Constructing from a dataset with a field variable not on (z, y, x) raises."""
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

    def test_variables_excludes_structural(self, profile):
        """variables lists non-structural data_vars only."""
        assert set(profile.variables) == {"reff", "veff", "mass_concentration"}

    def test_getitem_cached(self, profile):
        """profile[name] returns the same cached quantity on every access."""
        for name in profile.variables:
            assert profile[name] is profile[name]
        assert profile["reff"].units == ureg.micron

    def test_getitem_rejects_structural_variable(self, profile):
        """profile[name] raises for a structural (x/y/z_edges) variable."""
        with pytest.raises(KeyError, match="x_edges"):
            profile["x_edges"]

    def test_getitem_rejects_unknown_variable(self, profile):
        """profile[name] raises for a name that isn't a profile variable at all."""
        with pytest.raises(KeyError, match="nonexistent"):
            profile["nonexistent"]


class TestResampleRegular:
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
            edges_x=np.array([0.0, 1.0, 2.0]) * ureg.km,
            edges_y=np.array([0.0, 1.0, 2.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )

    def test_containing_layer_z(self, profile, grid):
        """resample_regular picks the source layer containing each target layer centre."""
        result = profile.resample_regular(grid)
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
            edges_x=np.array([0.0, 1.0]) * ureg.km,
            edges_y=np.array([0.0, 1.0]) * ureg.km,
            levels=np.linspace(0.0, 3.0, 7) * ureg.km,
        )
        result = profile.resample_regular(grid)
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

        result = extended.resample_regular(grid)
        np.testing.assert_allclose(
            result["number_concentration"].m_as("cm^-3"), np.full((4, 2, 2), 5.0)
        )

    def test_output_is_regular_on_all_axes(self, profile, grid):
        """The resampled profile's x/y/z edges are all evenly spaced."""
        result = profile.resample_regular(grid)
        for name in ("x_edges", "y_edges", "z_edges"):
            values = result.data[name].values
            step = np.diff(values)
            np.testing.assert_allclose(step, step[0])

    def test_output_dataset_valid(self, profile, grid):
        """The resampled dataset is itself a valid ParticleProfile input."""
        result = profile.resample_regular(grid)
        ParticleProfile(data=result.data)

    @pytest.mark.parametrize("axis", ["x_edges", "y_edges"])
    def test_irregular_source_raises(self, profile, grid, axis):
        """resample_regular rejects a profile with an irregular x_edges/y_edges axis."""
        ds = profile.data.copy()
        ds[axis] = ([axis], np.array([0.0, 1.0, 3.0]), {"units": "km"})
        irregular = ParticleProfile(data=ds)
        with pytest.raises(ValueError, match=f"'{axis}' must be regularly spaced"):
            irregular.resample_regular(grid)

    def test_extent_mismatch_raises(self, profile):
        """resample_regular rejects a target grid whose extent doesn't match the profile's."""
        mismatched_grid = CartesianGrid(
            edges_x=np.array([0.0, 1.0, 2.0, 3.0]) * ureg.km,
            edges_y=np.array([0.0, 1.0, 2.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )
        with pytest.raises(ValueError, match="do not match the target"):
            profile.resample_regular(mismatched_grid)

    def test_resolution_mismatch_raises(self, profile):
        """resample_regular rejects a target grid whose resolution doesn't match the profile's."""
        mismatched_grid = CartesianGrid(
            edges_x=np.array([0.0, 2.0]) * ureg.km,
            edges_y=np.array([0.0, 1.0, 2.0]) * ureg.km,
            levels=np.linspace(0.0, 4.0, 5) * ureg.km,
        )
        with pytest.raises(ValueError, match="do not match the target"):
            profile.resample_regular(mismatched_grid)

    def test_spherical_shell_dispatch(self):
        """resample_regular dispatches to the SphericalGrid implementation."""
        profile = ParticleProfile(
            data=make_profile_dataset(
                reff=10.0,
                veff=0.1,
                mass_concentration=1e-3,
                x_edges=np.linspace(0.0, 2 * np.pi, 3),
                y_edges=np.linspace(0.0, np.pi, 3),
                z_edges=[0.0, 1.0],
            )
        )
        profile.data["x_edges"].attrs["units"] = "radian"
        profile.data["y_edges"].attrs["units"] = "radian"

        grid = SphericalGrid(
            levels=np.array([0.0, 1.0]) * ureg.km,
            azimuths=np.linspace(0.0, 2 * np.pi, 3) * ureg.radian,
            colatitudes=np.linspace(0.0, np.pi, 3) * ureg.radian,
        )
        result = profile.resample_regular(grid)
        assert result.data["reff"].shape == (1, 2, 2)
