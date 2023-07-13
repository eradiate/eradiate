import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg

# ------------------------------------------------------------------------------
#                            Pre-process helpers
# ------------------------------------------------------------------------------

pytest.register_assert_rewrite("eradiate.test_tools.types.check_scene_element")

# ------------------------------------------------------------------------------
#                                 Mode fixtures
# ------------------------------------------------------------------------------


def generate_fixture(mode):
    @pytest.fixture()
    def fixture():
        import eradiate

        eradiate.set_mode(mode)

    globals()["mode_" + mode] = fixture


for mode in eradiate.modes():
    generate_fixture(mode)
del generate_fixture


def generate_fixture_group(name, modes):
    @pytest.fixture(params=modes)
    def fixture(request):
        mode = request.param
        import eradiate

        eradiate.set_mode(mode)

    globals()["modes_" + name] = fixture


variants = [x for x in eradiate.modes() if x not in {"mono", "ckd"}]  # Remove aliases
variant_groups = {
    "all_mono": [x for x in variants if x.startswith("mono")],
    "all_ckd": [x for x in variants if x.startswith("ckd")],
    "all_mono_ckd": [
        x for x in variants if (x.startswith("mono") or x.startswith("ckd"))
    ],
    "all_single": [x for x in variants if x.endswith("single")],
    "all_double": [x for x in variants if x.endswith("double")],
    "all": variants,
}

for name, variants in variant_groups.items():
    generate_fixture_group(name, variants)
del generate_fixture_group


# ------------------------------------------------------------------------------
#                              Other configuration
# ------------------------------------------------------------------------------


@pytest.fixture
def ert_seed_state():
    from eradiate.rng import SeedState

    return SeedState(0)


TEST_ERROR_HANDLER_CONFIG = {
    "x": {
        "missing": "ignore",
        "scalar": "ignore",
        "bounds": "raise",
    },
    "p": {"bounds": "ignore"},
    "t": {"bounds": "ignore"},
}


@pytest.fixture
def error_handler_config():
    """Error handler configuration for absorption coefficient interpolation.

    Notes
    -----
    This configuration is chosen to ignore all interpolation issues (except
    bounds error along the mole fraction dimension) because warnings are
    captured by pytest which will raise.
    Ignoring the bounds on pressure and temperature is safe because
    out-of-bounds values usually correspond to locations in the atmosphere
    that are so high that the contribution to the absorption coefficient
    are negligible at these heights.
    The bounds error for the 'x' (mole fraction) coordinate is considered
    fatal.
    """
    return TEST_ERROR_HANDLER_CONFIG


@pytest.fixture
def us_standard_mono():
    """
    AFGL (1986) U.S. Standard atmosphere with monochromatic absorption data.

    Notes
    -----
    Molecules included are H2O, CO2, O3, N2O, CO, CH4, O2.
    Specified absorption data covers the wavelength range [250, 3125] nm.
    Altitude grid is regular with a 1 km step, from 0 to 120 km.
    """
    return {
        "type": "molecular",
        "thermoprops": {
            "identifier": "afgl_1986-us_standard",
            "z": np.linspace(0.0, 120.0, 121) * ureg.km,
            "additional_molecules": False,
        },
        "absorption_data": ("komodo", [549.5, 550.5] * ureg.nm),
        "error_handler_config": TEST_ERROR_HANDLER_CONFIG,
    }


@pytest.fixture
def us_standard_ckd_550nm():
    """
    AFGL (1986) U.S. Standard atmosphere with CKD absorption data.

    Notes
    -----
    Molecules included are H2O, CO2, O3, N2O, CO, CH4, O2.
    Specified absorption data covers the CKD band associated with wavenumber
    interval [18100, 18200] cm^-1, i.e. the wavelenght range
    [549.45, 552.48] nm.
    Altitude grid is regular with a 1 km step, from 0 to 120 km.
    """
    return {
        "type": "molecular",
        "thermoprops": {
            "identifier": "afgl_1986-us_standard",
            "z": np.linspace(0.0, 120.0, 121) * ureg.km,
            "additional_molecules": False,
        },
        "absorption_data": ("monotropa", [549.5, 550.5] * ureg.nm),
        "error_handler_config": TEST_ERROR_HANDLER_CONFIG,
    }


@pytest.fixture
def cams_lybia4_ckd_550nm():
    """
    CAMS Lybia4 atmosphere with CKD absorption data.

    Notes
    -----
    Molecules included are H2O, CO2, O3, N2O, CO, CH4, O2, NO2, NO, SO2.
    Specified absorption data covers the CKD band associated with wavenumber
    interval [18100, 18200] cm^-1, i.e. the wavelenght range
    [549.45, 552.48] nm.
    Altitude grid is regular with a 1 km step, from 0 to 120 km.
    """
    thermoprops = eradiate.data.load_dataset(
        "tests/thermoprops/cams_lybia4_2005-04-01.nc"
    )

    return {
        "type": "molecular",
        "thermoprops": thermoprops,
        "absorption_data": ("monotropa", [549.5, 550.5] * ureg.nm),
        "error_handler_config": TEST_ERROR_HANDLER_CONFIG,
    }
