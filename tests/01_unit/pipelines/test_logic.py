"""
This module tests the post-processing pipeline logic. It uses parametrized
fixtures to make the process of generating the same data multiple times as easy
and comprehensive as possible. The processing part of testing is implemented by
fixtures; test bodies perform only data checks.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import eradiate
import eradiate.pipelines.logic as logic
from eradiate.experiments import AtmosphereExperiment
from eradiate.scenes.illumination import ConstantIllumination, DirectionalIllumination
from eradiate.spectral import CKDSpectralGrid, MonoSpectralGrid
from eradiate.units import unit_registry as ureg

# ------------------------------------------------------------------------------
#                                    Fixtures
# ------------------------------------------------------------------------------

# In order to ensure that each test step gets the data formatted as it expects,
# we provide here a fixture-based implementation of a "mini-pipeline" that
# chains the operations that are unit-tested.


@pytest.fixture(params=["mono", "ckd", "mono_polarized", "ckd_polarized"])
def mode_id(request):
    yield request.param


@pytest.fixture
def mode(mode_id):
    eradiate.set_mode(mode_id)
    yield eradiate.get_mode()


@pytest.fixture(params=["hdistant", "distantflux"])
def measure(request):
    if request.param in {"hdistant", "distantflux"}:
        yield request.param
    else:
        raise NotImplementedError(request.param)


@pytest.fixture(params=["delta", "sentinel_2a-msi-3"])
def srf(request):
    if request.param in {"delta", "sentinel_2a-msi-3"}:
        yield request.param
    else:
        raise NotImplementedError(request.param)


def get_experiment(mode_id, srf, measure):
    experiment_setup = (mode_id, srf, measure)
    eradiate.set_mode(mode_id)
    mode = eradiate.get_mode()

    if experiment_setup not in EXPERIMENTS:
        if srf == "delta":
            srf_dict = {"type": "delta", "wavelengths": 550.0 * ureg.nm}
            spp = 256
        elif srf == "sentinel_2a-msi-3":
            srf_dict = "sentinel_2a-msi-3"
            spp = 32
        else:
            raise NotImplementedError(srf)

        exp = AtmosphereExperiment(
            atmosphere=None,
            surface={"type": "lambertian", "reflectance": 1.0},
            illumination={"type": "directional", "irradiance": 2.0},
            measures={
                "type": measure,
                "film_resolution": (32, 32),
                "spp": spp,
                "srf": srf_dict,
            },
        )

        var_name, _ = exp.measures[0].var
        exp.integrator.stokes = mode.is_polarized and var_name == "radiance"
        exp.integrator.moment = True

        exp.process()
        EXPERIMENTS[experiment_setup] = exp

    return EXPERIMENTS[experiment_setup]


@pytest.fixture
def experiment(mode_id, srf, measure):
    # Create an experiment and run it (parametrized by above fixtures)
    yield get_experiment(mode_id, srf, measure)


@pytest.fixture
def irradiance(mode_id, experiment):
    return logic.extract_irradiance(
        mode_id, experiment.illumination, experiment.spectral_grids[0]
    )["irradiance"]


@pytest.fixture
def solar_angles(mode_id, experiment):
    return logic.extract_irradiance(
        mode_id, experiment.illumination, experiment.spectral_grids[0]
    )["solar_angles"]


@pytest.fixture
def viewing_angles(experiment):
    return logic.viewing_angles(experiment.measures[0].viewing_angles.m_as("deg"))


@pytest.fixture
def raw_results(experiment):
    # Convenience: Return the raw results associated to an experiment
    return experiment.measures[0].mi_results


@pytest.fixture
def gather_bitmaps(mode, experiment, raw_results, viewing_angles, solar_angles):
    # Apply the `gather_bitmaps` pipeline step and return the results
    var_name, var_metadata = experiment.measures[0].var
    calculate_stokes = mode.is_polarized and var_name == "radiance"
    gather_variance = experiment.integrator.moment

    return logic.gather_bitmaps(
        mode_id=mode.id,
        var_name=var_name,
        var_metadata=var_metadata,
        gather_variance=gather_variance,
        calculate_stokes=calculate_stokes,
        bitmaps=raw_results,
        viewing_angles=viewing_angles,
        solar_angles=solar_angles,
    )


@pytest.fixture
def moment2_to_variance(mode, experiment, gather_bitmaps):
    var_name, _ = experiment.measures[0].var
    calculate_stokes = mode.is_polarized and var_name == "radiance"

    results_raw = gather_bitmaps[f"{var_name}_raw"]
    m2_raw = gather_bitmaps[f"{var_name}_m2_raw"]
    spp = gather_bitmaps["spp"]

    return logic.moment2_to_variance(results_raw, m2_raw, spp, calculate_stokes)


@pytest.fixture
def aggregate_ckd_quad(mode, experiment, gather_bitmaps):
    var_name = experiment.measures[0].var[0]
    results_raw = gather_bitmaps[f"{var_name}_raw"]
    calculate_variance = False

    return logic.aggregate_ckd_quad(
        mode_id=mode.id,
        raw_data=results_raw,
        spectral_grid=experiment.spectral_grids[0],
        ckd_quads=experiment.ckd_quads[0],
        is_variance=calculate_variance,
    )


@pytest.fixture
def aggregate_ckd_quad_var(mode, experiment, moment2_to_variance):
    calculate_variance = True

    return logic.aggregate_ckd_quad(
        mode_id=mode.id,
        raw_data=moment2_to_variance,
        spectral_grid=experiment.spectral_grids[0],
        ckd_quads=experiment.ckd_quads[0],
        is_variance=calculate_variance,
    )


@pytest.fixture
def apply_spectral_response(mode, measure, experiment, aggregate_ckd_quad):
    spectral_data = aggregate_ckd_quad
    return logic.apply_spectral_response(spectral_data, experiment.measures[0].srf)


# ------------------------------------------------------------------------------
#                                     Tests
# ------------------------------------------------------------------------------

EXPERIMENTS = {}


# def test_00_run_experiment(mode_id, srf, measure):
#     """
#     This test exists for debugging purposes. In case the setup of a test takes
#     too long, this one can be used to further investigate issued related to
#     experiment execution.
#     """
#     assert get_experiment(mode_id, srf, measure)


@pytest.mark.parametrize("srf", ["delta"])
@pytest.mark.parametrize("measure", ["hdistant"])
def test_01_gather_bitmaps(mode, experiment, gather_bitmaps):
    # Routine creates the variables we expect
    expected_variables = {"spp", "radiance_raw", "radiance_m2_raw", "weights_raw"}
    assert set(gather_bitmaps.keys()) == expected_variables

    # Each variable has the dimensions we expect
    # Note: the 'channel' film dimension is dropped in mono mode
    if mode.is_mono:
        spectral_sizes = {"w": 1}
    elif mode.is_ckd:
        spectral_sizes = {"w": 1, "g": 16}  # 16 is the default value
    else:
        raise RuntimeError("Selected mode is not handled")

    solar_angle_sizes = {"sza": 1, "saa": 1}
    film_sizes = {"y_index": 32, "x_index": 32}
    all_sizes = {**spectral_sizes, **film_sizes, **solar_angle_sizes}

    expected_sizes = {
        "spp": spectral_sizes,
        "radiance_raw": all_sizes
        if not mode.is_polarized
        else {**all_sizes, "stokes": 4},
        "radiance_m2_raw": all_sizes
        if not mode.is_polarized
        else {**all_sizes, "stokes": 1},
    }

    for var, da in gather_bitmaps.items():  # noqa: F402
        if da is None:
            continue
        assert da.sizes == expected_sizes[var], (
            f"While checking variable '{var}', expected sizes '{expected_sizes[var]}', "
            f"got '{da.sizes}'"
        )

    if mode.is_polarized:
        # Check radiance values
        assert np.allclose(
            gather_bitmaps["radiance_raw"].sel(stokes="I").values, 2.0 / np.pi
        )
    else:
        # Check radiance values
        assert np.allclose(gather_bitmaps["radiance_raw"].values, 2.0 / np.pi)


@pytest.mark.parametrize("mode_id", ["ckd", "ckd_polarized"])
@pytest.mark.parametrize("measure", ["hdistant"])
def test_02_aggregate_ckd_quad(mode, experiment, gather_bitmaps, aggregate_ckd_quad):
    var_name = experiment.measures[0].var[0]
    raw = gather_bitmaps[f"{var_name}_raw"]
    result = aggregate_ckd_quad

    # Dimension checks
    expected_dims = set(raw.dims) - {"g"}
    assert set(result.dims) == expected_dims

    # Coordinate checks
    expected_coords = (set(raw.coords) - {"g"}) | {"bin_wmin", "bin_wmax"}
    assert set(result.coords) == expected_coords
    assert result.bin_wmin.dims == ("w",)
    assert result.bin_wmax.dims == ("w",)

    # Variable checks
    # -- In the present case, the quadrature evaluates to 2/π
    if not mode.is_polarized:
        assert np.allclose(result.values, 2.0 / np.pi)
    else:
        assert np.allclose(result.sel(stokes="I").values, 2.0 / np.pi)

    # -- Metadata of the variable for which aggregation is performed are copied
    assert result.attrs == raw.attrs


@pytest.mark.parametrize("mode_id", ["ckd", "ckd_polarized"])
@pytest.mark.parametrize("measure", ["hdistant"])
def test_03_aggregate_ckd_quad_var(experiment, gather_bitmaps, aggregate_ckd_quad_var):
    var_name = experiment.measures[0].var[0]
    raw = gather_bitmaps[f"{var_name}_m2_raw"]
    result = aggregate_ckd_quad_var

    # Dimension checks
    expected_dims = set(raw.dims) - {"g"}
    assert set(result.dims) == expected_dims

    # Coordinate checks
    expected_coords = (set(raw.coords) - {"g"}) | {"bin_wmin", "bin_wmax"}
    assert set(result.coords) == expected_coords
    assert result.bin_wmin.dims == ("w",)
    assert result.bin_wmax.dims == ("w",)

    # Variable checks
    # -- Metadata of the variable for which aggregation is performed are copied
    assert result.attrs == raw.attrs


@pytest.mark.parametrize("mode_id", ["ckd"])
@pytest.mark.parametrize("srf", ["sentinel_2a-msi-3"])
def test_04_apply_spectral_response_main(
    experiment, aggregate_ckd_quad, apply_spectral_response
):
    """
    Unit test for :func:`.apply_spectral_response` when used on the main
    variable.
    """
    var_name = experiment.measures[0].var[0]
    raw = aggregate_ckd_quad
    result = apply_spectral_response

    # Dimension checks
    expected_dims = set(raw.dims) - {"w"}
    assert set(result.dims) == expected_dims

    # Coordinate checks
    expected_coords = set(raw.coords) - ({"w"} | {"bin_wmax", "bin_wmin"})
    assert set(result.coords) == expected_coords

    # The step adds an SRF-weighted variable
    assert apply_spectral_response.name == f"{var_name}_srf"
    assert np.all(apply_spectral_response.values > 0.0)


@pytest.mark.parametrize("mode_id", ["ckd"])
@pytest.mark.parametrize("srf", ["sentinel_2a-msi-3"])
def test_05_apply_spectral_response_irradiance(irradiance, experiment):
    """
    Unit test for :func:`.apply_spectral_response` when used on the irradiance
    variable.
    """
    result = logic.apply_spectral_response(irradiance, experiment.measures[0].srf)

    # The step adds an SRF-weighted variable
    assert result.name == "irradiance_srf"
    assert np.all(result.values > 0.0)


@pytest.mark.parametrize(
    "illumination_type, expected_dims, expect_solar_angles",
    [
        ("directional", ["sza", "saa"], True),
        ("constant", [], False),
    ],
)
def test_06_extract_irradiance(
    mode, illumination_type, expected_dims, expect_solar_angles
):
    if illumination_type == "directional":
        illumination = DirectionalIllumination(zenith=30.0, azimuth=45.0)
    elif illumination_type == "constant":
        illumination = ConstantIllumination()
    else:
        raise ValueError

    # Computation succeeds
    if mode.is_mono:
        spectral_grid = MonoSpectralGrid(np.linspace(400.0, 500.0) * ureg.nm)
    elif mode.is_ckd:
        spectral_grid = CKDSpectralGrid.arange(
            400.0 * ureg.nm, 500.0 * ureg.nm, 10.0 * ureg.nm
        )
    else:
        raise NotImplementedError

    # Return value is a dictionary holding a data array and an optional dataset
    result = logic.extract_irradiance(mode.id, illumination, spectral_grid)
    assert set(result.keys()) == {"irradiance", "solar_angles"}
    assert isinstance(result["irradiance"], xr.DataArray)

    if expect_solar_angles:
        assert isinstance(result["solar_angles"], xr.Dataset)
    else:
        assert result["solar_angles"] is None

    # Irradiance is indexed by solar angle and spectral coordinates
    assert set(result["irradiance"].dims) == {"w"}.union(set(expected_dims))

    # Irradiance data array also contains bin bounds as coordinates when relevant
    expected_coords = {"w"}.union(set(expected_dims))
    if eradiate.mode().is_ckd:
        expected_coords |= {"bin_wmin", "bin_wmax"}
    assert set(result["irradiance"].coords) == expected_coords


@pytest.mark.parametrize("measure", ["distantflux"])
def test_07_radiosity(mode, gather_bitmaps):
    # Initialize test data
    irradiance = 2.0
    sector_radiosity = gather_bitmaps["sector_radiosity_raw"]

    # Configure and apply step
    result = logic.radiosity(sector_radiosity=sector_radiosity)
    # Check that radiosity dimensions are correct
    assert not {"x_index", "y_index"}.issubset(result.dims)
    # This setup conserves energy
    assert np.allclose(irradiance, result, rtol=1e-3)


@pytest.mark.parametrize("mode_id", ["mono"])
@pytest.mark.parametrize("srf", ["delta"])
def test_08_viewing_angles(experiment):
    # This test is minimal and simply checks for returned type and variable names
    result = logic.viewing_angles(experiment.measures[0].viewing_angles)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"vaa", "vza"}


@pytest.mark.parametrize("mode_id", ["mono_polarized", "ckd_polarized"])
@pytest.mark.parametrize("measure", ["hdistant"])
@pytest.mark.parametrize("srf", ["delta"])
def test_09_degree_of_linear_polarization(mode, aggregate_ckd_quad):
    result = logic.degree_of_linear_polarization(aggregate_ckd_quad)

    # Each variable has the dimensions we expect
    spectral_sizes = {"w": 1}
    solar_angle_sizes = {"sza": 1, "saa": 1}
    film_sizes = {"y_index": 32, "x_index": 32}
    expected_size = {**spectral_sizes, **film_sizes, **solar_angle_sizes}
    assert isinstance(result, xr.DataArray)
    assert result.sizes == expected_size
