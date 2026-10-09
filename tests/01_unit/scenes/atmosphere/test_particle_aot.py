import numpy as np
import pint
import pytest
import xarray as xr

from eradiate import KernelContext, fresolver
from eradiate import unit_registry as ureg
from eradiate.grid import SpatialGrid, CartesianGrid, SphericalGrid
from eradiate.data.convert import make_prt_psd_v1
from eradiate.radprops import ParticleProperties
from eradiate.scenes.atmosphere import (
    ArrayParticleDistribution,
    ParticleAOTComponent,
    ParticleComponent,
    UniformParticleDistribution,
)
from eradiate.scenes.core import traverse
from eradiate.spectral.index import SpectralIndex
from eradiate.test_tools.types import check_scene_element
from eradiate.units import to_quantity

# ------------------------------------------------------------------------------
#                                   Fixtures
# ------------------------------------------------------------------------------

DS_ID_TO_FNAME = {
    "unpolarized_data": "govaerts_2021-desert-aer_core_v2",
    "polarized_data": "aeronet_sahara_spherical_RAMIA_GENERIC_extrapolated-aer_core_v2",
}


@pytest.fixture(scope="module", params=list(DS_ID_TO_FNAME.keys()))
def particle_dataset_path(request):
    ds_id = request.param
    fname = DS_ID_TO_FNAME[ds_id]
    yield f"tests/aerosol/{fname}.nc"


@pytest.fixture(scope="module")
def unpolarized_particle_dataset():
    fname = DS_ID_TO_FNAME["unpolarized_data"]
    fname = f"tests/aerosol/{fname}.nc"
    yield fresolver.load_dataset(fname)


@pytest.fixture(scope="module")
def particle_dataset(particle_dataset_path):
    yield fresolver.load_dataset(particle_dataset_path)


@pytest.fixture(scope="module")
def particle_properties(particle_dataset):
    yield ParticleProperties(particle_dataset)


# ------------------------------------------------------------------------------
#                                   Tests
# ------------------------------------------------------------------------------


def _make_3d_grid():
    return CartesianGrid(
        edges_x=np.array([0.0, 1.0, 2.0]) * ureg.km,
        edges_y=np.array([0.0, 1.0, 2.0]) * ureg.km,
        levels=np.linspace(0.0, 5.0, 11) * ureg.km,
    )


class TestParticleAOTComponent:
    @pytest.mark.parametrize("geometry", ["plane_parallel", "spherical_shell"])
    def test_basic(self, mode_mono, geometry):
        """Basic constructor pattern"""
        # Construct from path
        layer = ParticleAOTComponent(
            particle_properties="tests/aerosol/govaerts_2021-desert-aer_core_v2.nc",
            geometry=geometry,
        )

        # Check if kernel dictionary is well-formed
        check_scene_element(layer)

    def test_construct_noargs(self):
        """Construction succeeds with basic parameters"""

        assert (
            ParticleAOTComponent()
        )  # should pick up the default dataset from shipped assets

    def test_convert_passthrough(self):
        """ParticleComponent.convert returns an existing instance unchanged."""
        component = ParticleAOTComponent()
        assert ParticleAOTComponent.convert(component) is component
        assert ParticleComponent.convert(component) is component

    def test_convert_from_dict_defaults_to_particle_aot(self):
        """
        Called on the abstract ParticleComponent base, a dict without 'type'
        defaults to 'particle_aot' and is routed through atmosphere_factory.
        """
        result = ParticleComponent.convert({})
        assert isinstance(result, ParticleAOTComponent)

    def test_convert_rejects_wrong_type(self):
        """A value that is neither a dict nor a ParticleComponent instance is rejected."""
        with pytest.raises(TypeError):
            ParticleComponent.convert(object())

    def test_construct_attrs(self, particle_properties_test):
        """Assigns parameters to expected values."""
        bottom = 1.2 * ureg.km
        top = 1.8 * ureg.km
        tau_ref = 0.3 * ureg.dimensionless
        layer = ParticleAOTComponent(
            geometry={
                "type": "plane_parallel",
                "ground_altitude": bottom,
                "toa_altitude": top,
            },
            bottom=bottom,
            top=top,
            distribution={"type": "uniform"},
            tau_ref=tau_ref,
            particle_properties=particle_properties_test,
        )
        assert layer.bottom_altitude == bottom  # inherited from Atmosphere
        assert layer.top_altitude == top  # inherited from Atmosphere
        assert isinstance(layer.distribution, UniformParticleDistribution)
        assert layer.tau_ref == tau_ref

    def test_altitude_units(self, particle_dataset_path):
        """Accept different units for bottom and top altitudes."""
        assert ParticleAOTComponent(
            particle_properties=particle_dataset_path,
            bottom=1.0 * ureg.km,
            top=2000.0 * ureg.m,
        )

    def test_invalid_bottom_top(self, particle_dataset_path):
        """Raises when 'bottom' is greater that 'top'."""
        with pytest.raises(
            ValueError, match="bottom altitude must be lower than top altitude"
        ):
            ParticleAOTComponent(
                particle_properties=particle_dataset_path,
                top=1.2 * ureg.km,
                bottom=1.8 * ureg.km,
            )

    def test_invalid_tau_ref(self, particle_dataset_path):
        """Raises when 'tau_ref' is invalid."""
        with pytest.raises(ValueError, match="tau_ref must be all positive or zero"):
            ParticleAOTComponent(
                particle_properties=particle_dataset_path,
                bottom=1.2 * ureg.km,
                top=1.8 * ureg.km,
                tau_ref=-0.1 * ureg.dimensionless,
            )

    def test_particle_properties_rejects_size_distribution(self):
        """Raises when 'particle_properties' has 'reff'/'veff' dimensions."""
        mu = np.linspace(-1.0, 1.0, 5)
        theta = np.rad2deg(np.arccos(mu))
        properties_with_size_dist = make_prt_psd_v1(
            w=ureg.Quantity([1.0], "micron"),
            phamat=["11"],
            mu=ureg.Quantity(
                mu[np.newaxis, np.newaxis, np.newaxis, :], "dimensionless"
            ),
            theta=ureg.Quantity(theta[np.newaxis, np.newaxis, np.newaxis, :], "degree"),
            ext=ureg.Quantity([[[1.0]]], "1/km"),
            ssa=ureg.Quantity([[[0.6]]], "dimensionless"),
            phase=ureg.Quantity(np.ones((1, 1, 1, 1, 5)), "1/sr"),
            pmom=np.zeros((1, 1, 1, 1)),
            reff=ureg.Quantity([10.0], "micron"),
            veff=ureg.Quantity([0.1], "dimensionless"),
        )

        with pytest.raises(ValueError, match="reff.*veff"):
            ParticleAOTComponent(
                particle_properties=properties_with_size_dist,
                w_ref=ureg.Quantity(1.0, "micron"),
            )

    @pytest.mark.parametrize("w", list([280.0, 550.0, 1600.0, 2400.0] * ureg.nm))
    def test_eval_absorbing_only(
        self, modes_all_unpolarized_double, particle_properties_absorbing_only, w
    ):
        """
        eval methods return expected values for an absorbing-only layer
        """
        bottom = 0.0 * ureg.km
        top = 1.0 * ureg.km
        layer = ParticleAOTComponent(
            geometry={
                "type": "plane_parallel",
                "ground_altitude": bottom,
                "toa_altitude": top,
            },
            particle_properties=particle_properties_absorbing_only,
            tau_ref=0.2,
            w_ref=550.0,
            bottom=bottom,
            top=top,
        )

        si = SpectralIndex.new(w=w)
        np.testing.assert_allclose(layer.eval_sigma_s(si).m_as("1/km"), 0.0)
        np.testing.assert_allclose(layer.eval_sigma_a(si).m_as("1/km"), 0.2)
        np.testing.assert_allclose(layer.eval_albedo(si).m, 0.0)

        ctx = KernelContext(si=si)
        assert layer.eval_mfp(ctx).magnitude == np.inf

    @pytest.mark.parametrize("w", list([280.0, 550.0, 1600.0, 2400.0] * ureg.nm))
    def test_eval_mono_scattering_only(
        self, modes_all_unpolarized_double, particle_properties_scattering_only, w
    ):
        """
        eval methods return expected values for an scattering-only layer
        """
        bottom = 0.0 * ureg.km
        top = 1.0 * ureg.km
        layer = ParticleAOTComponent(
            geometry={
                "type": "plane_parallel",
                "ground_altitude": bottom,
                "toa_altitude": top,
            },
            particle_properties=particle_properties_scattering_only,
            tau_ref=0.2,
            w_ref=550.0,
            bottom=bottom,
            top=top,
        )

        si = SpectralIndex.new(w=w)
        np.testing.assert_allclose(layer.eval_sigma_s(si).m_as("1/km"), 0.2)
        np.testing.assert_allclose(layer.eval_sigma_a(si).m_as("1/km"), 0.0)
        np.testing.assert_allclose(layer.eval_albedo(si).m, 1.0)

        ctx = KernelContext(si=si)
        assert np.isclose(layer.eval_mfp(ctx), 5.0 * ureg.km)

    @pytest.mark.parametrize("w", list([280.0, 550.0, 1600.0, 2400.0] * ureg.nm))
    def test_eval_general(
        self, modes_all_unpolarized_double, particle_properties_test, w
    ):
        """
        eval_* methods return expected values for a scattering and absorbing layer
        """
        bottom = 0.0 * ureg.km
        top = 1.0 * ureg.km
        layer = ParticleAOTComponent(
            geometry={
                "type": "plane_parallel",
                "ground_altitude": bottom,
                "toa_altitude": top,
            },
            particle_properties=particle_properties_test,
            tau_ref=1.0,
            w_ref=550.0,
            bottom=bottom,
            top=top,
        )

        si = SpectralIndex.new(w=w)
        np.testing.assert_allclose(layer.eval_sigma_t(si).m_as("1/km"), 1.0)
        np.testing.assert_allclose(layer.eval_sigma_s(si).m_as("1/km"), 0.8)
        np.testing.assert_allclose(layer.eval_sigma_a(si).m_as("1/km"), 0.2)
        np.testing.assert_allclose(layer.eval_albedo(si).m, 0.8)

        ctx = KernelContext(si=si)
        assert np.isclose(layer.eval_mfp(ctx), 1.25 * ureg.km)

    def test_kernel_scale(self, modes_all_single, particle_properties_test):
        """Scale parameter propagates to kernel dict and latter can be loaded."""
        particles = ParticleAOTComponent(
            id="atmosphere", particle_properties=particle_properties_test, scale=2.0
        )
        template, _ = traverse(particles)
        assert template["medium_atmosphere.scale"] == 2.0

    def test_eval_radprops_format(self, modes_all_single, particle_dataset_path):
        """
        Method 'eval_radprops' returns dataset with expected datavars and coords.
        """
        layer = ParticleAOTComponent(particle_properties=particle_dataset_path)
        si = SpectralIndex.new()
        ds = layer.eval_radprops(si)
        expected_data_vars = ["sigma_t", "albedo"]
        expected_coords = ["z_layer"]
        assert all(coord in ds.coords for coord in expected_coords) and all(
            var in ds.data_vars for var in expected_data_vars
        )

    @pytest.mark.parametrize(
        "tau_ref", list(np.array([0.6, 1.0, 2.5]) * ureg.dimensionless)
    )
    def test_eval_radprops(self, mode_mono, particle_dataset_path, tau_ref):
        layer = ParticleAOTComponent(
            particle_properties=particle_dataset_path,
            bottom=0.5 * ureg.km,  # arbitrary
            top=3.0 * ureg.km,  # arbitrary
            distribution={"type": "uniform"},
            tau_ref=tau_ref,
        )

        # compute optical thickness at reference wavelength from layer's radprops
        # and check it matches the input tau_ref
        si = SpectralIndex.new(w=layer.w_ref)
        radprops = layer.eval_radprops(si)
        delta_z = layer.geometry.grid.layer_height

        with xr.set_options(keep_attrs=True):
            tau = to_quantity(radprops.sigma_t.sum()) * delta_z

        assert np.isclose(tau, tau_ref)

    @pytest.mark.parametrize("distribution", ["uniform", "gaussian", "exponential"])
    @pytest.mark.parametrize(
        "tau_ref", list([0.1, 0.5, 1.0, 2.0, 5.0] * ureg.dimensionless)
    )
    @pytest.mark.parametrize(
        "singlewavelength", [True, False], ids=["singlewavelength", "multiwavelength"]
    )
    def test_eval_sigma_t_impl(
        self,
        mode_mono,
        tau_ref,
        distribution,
        singlewavelength,
        unpolarized_particle_dataset,
    ):
        r"""
        Check correct handling of spectral dependency of extinction.

        If σ_t(λ) denotes the extinction coefficient at the wavelength λ, then the
        optical thickness of a uniform ParticleAOTComponent is τ(λ) = σ_t(λ) Δz, where Δz
        is the layer's thickness. From that follows:
        τ(λ) / τ(λ_ref) = σ_t(λ) / σ_t(λ_ref).

        This is what this test asserts.
        """
        ds = unpolarized_particle_dataset
        if singlewavelength:
            ds = ds.sel(w=[550], method="nearest")

        w_ref = 550 * ureg.nm
        bottom = 1.0 * ureg.km  # arbitrary
        top = 4.0 * ureg.km  # arbitrary
        n_wavelengths = 3
        n_layers = 10
        wavelengths = np.linspace(500.0, 1500.0, n_wavelengths) * ureg.nm
        grid = SpatialGrid.make_1d_from_levels(
            np.linspace(0, 5, n_layers + 1) * ureg.km
        )

        layer = ParticleAOTComponent(
            particle_properties=ds,
            geometry={
                "type": "plane_parallel",
                "toa_altitude": grid.levels[-1],
                "grid": grid,
            },
            bottom=bottom,
            top=top,
            distribution=distribution,
            w_ref=w_ref,
            tau_ref=tau_ref,
        )

        # Compute layer optical thickness at current wavelengths based on sigma_t
        # evaluation routine
        sigma_t = layer._eval_sigma_t_impl(wavelengths, layer.geometry.grid)
        assert sigma_t.units.is_compatible_with(ureg("m**-1"))
        assert sigma_t.shape == (n_wavelengths, 1, 1, n_layers)
        # -- Integrate sigma_t * dz vs space coordinate using rectangle method
        # (bottom/top are scalar here, so the x, y axes are trivial)
        tau = np.sum(sigma_t * layer.geometry.grid.layer_height, axis=-1).squeeze(
            axis=(1, 2)
        )

        # Manually compute extinction at running and reference wavelengths
        w_units = ureg(ds["w"].attrs["units"])

        if singlewavelength:
            sigma_t = to_quantity(
                ds["ext"].sel(w=wavelengths.m_as(w_units), method="nearest")
            )
            sigma_t_ref = to_quantity(
                ds["ext"].sel(w=w_ref.m_as(w_units), method="nearest")
            )
        else:
            sigma_t = to_quantity(ds["ext"].interp(w=wavelengths.m_as(w_units)))
            sigma_t_ref = to_quantity(ds["ext"].interp(w=w_ref.m_as(w_units)))

        # The spectral dependence of the optical thickness and extinction
        # coefficient match, so the ratios below must match
        result = (tau / tau_ref).m_as(ureg.dimensionless)
        expected = (sigma_t / sigma_t_ref).m_as(ureg.dimensionless)
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("distribution", ["uniform", "gaussian", "exponential"])
    @pytest.mark.parametrize(
        "bottom, top",
        [
            (1.0 * ureg.km, 4.0 * ureg.km),
            (
                np.array([[1.0, 1.5], [2.0, 2.5]]) * ureg.km,
                np.array([[3.0, 3.5], [4.0, 4.5]]) * ureg.km,
            ),
        ],
        ids=["scalar", "varying"],
    )
    def test_eval_fractions_shape_and_normalization(self, distribution, bottom, top):
        grid = _make_3d_grid()
        layer = ParticleAOTComponent(
            geometry={
                "type": "plane_parallel",
                "toa_altitude": grid.levels[-1],
                "grid": grid,
            },
            bottom=bottom,
            top=top,
            distribution=distribution,
        )

        fractions = layer.eval_fractions(grid)
        spatial_shape = fractions.shape[:-1]

        z_centers = grid.layers.m_as(ureg.km)
        bottom_grid = np.broadcast_to(bottom.m_as(ureg.km), spatial_shape)
        top_grid = np.broadcast_to(top.m_as(ureg.km), spatial_shape)

        for idx in np.ndindex(spatial_shape):
            in_range = (z_centers >= bottom_grid[idx]) & (z_centers <= top_grid[idx])
            col = fractions[idx]
            assert np.isclose(col.sum(), 1.0)
            assert np.all(col[in_range] > 0.0)
            assert np.allclose(col[~in_range], 0.0)

    def test_eval_fractions_with_3d_distribution(self):
        grid = _make_3d_grid()
        n_z = grid.layers.size
        values = np.zeros((2, 2, n_z))
        peak = np.arange(4).reshape(2, 2) % n_z
        np.put_along_axis(values, peak[..., np.newaxis], 1.0, axis=-1)

        # bottom/top must share the distribution's own (x, y) grid shape
        bottom = np.full((2, 2), grid.levels[0].m) * grid.levels.units
        top = np.full((2, 2), grid.levels[-1].m) * grid.levels.units

        particles = ParticleAOTComponent(
            geometry={
                "type": "plane_parallel",
                "toa_altitude": grid.levels[-1],
                "grid": grid,
            },
            bottom=bottom,
            top=top,
            distribution=ArrayParticleDistribution(values=values, method="nearest"),
        )

        fractions = particles.eval_fractions(grid)
        assert fractions.shape == (2, 2, n_z)
        assert np.allclose(fractions.sum(axis=-1), 1.0)
        assert not np.allclose(fractions[0, 0], fractions[0, 1])
        assert not np.allclose(fractions[0, 0], fractions[1, 0])

    def test_eval_fractions_uniform_is_flat(self):
        grid = _make_3d_grid()
        layer = ParticleAOTComponent(
            bottom=1.0 * ureg.km, top=4.0 * ureg.km, distribution="uniform"
        )

        fractions = layer.eval_fractions(grid)
        nonzero = fractions[fractions > 0.0]
        assert np.allclose(nonzero, nonzero[0])

    @pytest.mark.parametrize(
        "bottom, top",
        [
            (1.0 * ureg.km, 4.0 * ureg.km),
            (
                np.array([[1.0, 1.5], [2.0, 2.5]]) * ureg.km,
                np.array([[3.0, 3.5], [4.0, 4.5]]) * ureg.km,
            ),
        ],
        ids=["scalar_extent", "varying_extent"],
    )
    @pytest.mark.parametrize(
        "tau_ref",
        [
            0.5 * ureg.dimensionless,
            np.array([[0.2, 0.5], [1.0, 2.0]]) * ureg.dimensionless,
        ],
        ids=["scalar_tau_ref", "varying_tau_ref"],
    )
    def test_eval_sigma_t_broadcast(
        self, mode_mono, particle_dataset_path, bottom, top, tau_ref
    ):
        grid = _make_3d_grid()
        w_ref = 550 * ureg.nm

        layer = ParticleAOTComponent(
            geometry={
                "type": "plane_parallel",
                "toa_altitude": grid.levels[-1],
                "grid": grid,
            },
            particle_properties=particle_dataset_path,
            bottom=bottom,
            top=top,
            tau_ref=tau_ref,
            w_ref=w_ref,
        )

        si = SpectralIndex.new(w=w_ref)
        sigma_t = layer.eval_sigma_t(si, grid).m_as("km^-1")
        albedo = layer.eval_albedo(si, grid).m_as(ureg.dimensionless)
        sigma_s = layer.eval_sigma_s(si, grid).m_as("km^-1")
        sigma_a = layer.eval_sigma_a(si, grid).m_as("km^-1")
        fractions = layer.eval_fractions(grid)

        sigma_t, albedo, sigma_s, sigma_a, fractions = np.broadcast_arrays(
            sigma_t, albedo, sigma_s, sigma_a, fractions
        )
        spatial_shape = sigma_t.shape[:-1]

        tau = np.sum(sigma_t * grid.layer_height.m_as("km"), axis=-1)
        expected_tau = np.broadcast_to(tau_ref.m_as(ureg.dimensionless), spatial_shape)
        np.testing.assert_allclose(tau, expected_tau)

        np.testing.assert_allclose(sigma_s, sigma_t * albedo, rtol=1e-6)
        np.testing.assert_allclose(sigma_a, sigma_t * (1.0 - albedo), rtol=1e-6)

        for idx in np.ndindex(spatial_shape):
            nonzero = fractions[idx] > 0.0
            ratios = sigma_t[idx][nonzero] / fractions[idx][nonzero]
            np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6)

        if np.prod(spatial_shape, dtype=int) > 1 and np.size(tau_ref) > 1:
            assert not np.allclose(tau, tau.flat[0])

    @pytest.mark.parametrize(
        "has_absorption, has_scattering, expected",
        [
            (
                True,
                True,
                {
                    "albedo": "in_01",
                    "sigma_t": 1.0 / ureg.km,
                    "sigma_a": "positive",
                    "sigma_s": "positive",
                },
            ),
            (
                False,
                True,
                {
                    "albedo": 1.0 * ureg.dimensionless,
                    "sigma_t": "positive",
                    "sigma_a": 0.0 / ureg.km,
                    "sigma_s": "sigma_t",
                },
            ),
            (
                True,
                False,
                {
                    "albedo": 0.0 * ureg.dimensionless,
                    "sigma_t": "positive",
                    "sigma_a": "sigma_t",
                    "sigma_s": 0.0 / ureg.km,
                },
            ),
            (False, False, "raise"),
        ],
        ids=["both", "noabs", "noscat", "none"],
    )
    def test_switches(
        self, mode_mono, has_absorption, has_scattering, expected, particle_dataset_path
    ):
        try:
            bottom = 0.0 * ureg.km
            top = 1.0 * ureg.km
            particles = ParticleAOTComponent(
                particle_properties=particle_dataset_path,
                geometry={
                    "type": "plane_parallel",
                    "ground_altitude": bottom,
                    "toa_altitude": top,
                },
                tau_ref=1.0,
                has_absorption=has_absorption,
                has_scattering=has_scattering,
            )
            grid = particles.geometry.grid
            si = SpectralIndex.new(w=550.0 * ureg.nm)

            for field in ["albedo", "sigma_t", "sigma_a", "sigma_s"]:
                expected_value = expected[field]
                method = f"eval_{field}_mono"
                result = getattr(particles, method)(si, grid)

                if isinstance(expected_value, pint.Quantity):
                    np.testing.assert_allclose(
                        result.m_as(expected_value.units),
                        expected_value,
                        err_msg=f"Check of {method}() failed",
                    )

                elif expected_value == "in_01":
                    assert np.all((result > 0.0) & (result < 1.0)), (
                        f"Check of {method}() failed, some values are not in ]0, 1[ "
                        f"(got {result})"
                    )

                elif expected_value == "positive":
                    assert np.all(result > 0.0), (
                        f"Check of {method}() failed, some values are ≤0 (got {result})"
                    )

                elif expected_value == "sigma_t":
                    expected_value = particles.eval_sigma_t_mono(si, grid)
                    np.testing.assert_allclose(
                        result.m_as(expected_value.units),
                        expected_value.m,
                        err_msg=f"Check of {method}() failed, {field} and sigma_t should be identical",
                    )

                else:
                    raise RuntimeError

        except ValueError as e:
            assert expected == "raise"
            assert "while validating has_absorption" in str(e)

    def test_wavelength_alignment(self, particle_properties_test):
        """
        The single-scattering property dataset should contain the reference wavelength.
        """
        particle_properties = particle_properties_test
        w1, w2 = particle_properties.w[[1, 2]]

        assert ParticleAOTComponent(
            particle_properties=particle_properties, tau_ref=1.0, w_ref=w1
        )

        with pytest.warns(
            UserWarning,
            match="dataset does not contain the selected reference wavelength",
        ):
            ParticleAOTComponent(
                particle_properties=particle_properties,
                tau_ref=1.0,
                w_ref=0.5 * (w1 + w2),
            )
