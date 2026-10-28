import numpy as np
import pytest

from eradiate.scenes.atmosphere import HeterogeneousAtmosphere, eval_transmittance_ckd
from eradiate.spectral import CKDSpectralGrid


# Adaptive policies are currently disabled and fall back to "fixed" with a warning
@pytest.mark.filterwarnings("ignore:An advanced spectral quadrature configuration")
@pytest.mark.parametrize("policy", ["fixed", "error_threshold"])
def test_eval_transmittance_ckd(mode_ckd, atmosphere_us_standard_ckd, policy):
    atmosphere = HeterogeneousAtmosphere(
        molecular_atmosphere=atmosphere_us_standard_ckd
    )
    spectral_grid = CKDSpectralGrid.from_absorption_database(
        atmosphere.absorption_data
    ).select({"type": "uniform", "wmin": 545.0, "wmax": 555.0})

    # Adaptive policies require the absorption database to build quadrature rules
    result = eval_transmittance_ckd(
        atmosphere, spectral_grid, ckd_quad_config={"policy": policy}
    )

    n_x_cells, n_y_cells, _ = atmosphere.geometry.grid.shape
    assert result.shape == (n_x_cells, n_y_cells, len(spectral_grid.wcenters))
    assert np.all((result.values >= 0.0) & (result.values <= 1.0))
