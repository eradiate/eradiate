import numpy as np
import pint
import pytest

from eradiate import unit_registry as ureg
from eradiate.radprops import AFGL1986RadProfile, ZGrid
from eradiate.spectral.index import SpectralIndex


@pytest.fixture
def test_ckd_si_1650():
    return SpectralIndex.new(w=1650 * ureg.nm)


@pytest.fixture
def test_ckd_si_550():
    return SpectralIndex.new(w=550.0 * ureg.nm)


@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_default_ckd(mode_ckd, model_id):
    """
    Collision coefficient evaluation methods return pint.Quantity objects.
    """
    profile = AFGL1986RadProfile(
        absorption_dataset=f"ckd/absorption/10nm/afgl_1986-{model_id}-10nm-v3.nc",
    )

    si = SpectralIndex.new()
    for field in ["albedo", "sigma_a", "sigma_s", "sigma_t"]:
        x = getattr(profile, f"eval_{field}")(si, profile.zgrid)
        assert isinstance(x, ureg.Quantity)


@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_has_absorption_default(
    mode_ckd, test_ckd_si_1650, model_id
):
    """
    Default value for 'has_absorption' is True, hence the absorption
    coefficient is computed and is not zero everywhere at 1650 nm.
    """
    profile = AFGL1986RadProfile(
        absorption_dataset=f"ckd/absorption/10nm/afgl_1986-{model_id}-10nm-v3.nc",
    )
    assert profile.has_absorption
    ds = profile.eval_dataset(test_ckd_si_1650, profile.zgrid)
    assert (ds.sigma_a.values != 0.0).any()


@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_has_absorption_true(
    mode_ckd, test_ckd_si_1650, model_id
):
    """
    When 'has_absorption' is True, the absorption coefficient is computed
    and is not zero everywhere at 1650 nm.
    """
    profile = AFGL1986RadProfile(
        thermoprops=dict(model_id=model_id),
        has_absorption=True,
        absorption_dataset=f"ckd/absorption/10nm/afgl_1986-{model_id}-10nm-v3.nc",
    )
    assert profile.has_absorption
    ds = profile.eval_dataset(test_ckd_si_1650, profile.zgrid)
    assert (ds.sigma_a.values != 0.0).any()


@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_has_absorption_false(
    mode_ckd, test_ckd_si_1650, model_id
):
    """
    When 'has_absorption' is False, the absorption coefficient is not
    computed and is zero everywhere.
    """
    profile = AFGL1986RadProfile(
        absorption_dataset=f"ckd/absorption/10nm/afgl_1986-{model_id}-10nm-v3.nc",
        thermoprops=dict(model_id=model_id),
        has_absorption=False,
    )
    assert not profile.has_absorption
    ds = profile.eval_dataset(test_ckd_si_1650, profile.zgrid)
    assert (ds.sigma_a.values == 0.0).all()


def test_afgl_1986_rad_profile_has_scattering_default(mode_ckd, test_ckd_si_550):
    """
    Default value for 'has_scattering' is True, hence the absorption
    coefficient is computed and is not zero everywhere at 550 nm.
    """
    profile = AFGL1986RadProfile(
        absorption_dataset="ckd/absorption/10nm/afgl_1986-us_standard-10nm-v3.nc",
    )
    assert profile.has_scattering
    ds = profile.eval_dataset(test_ckd_si_550, profile.zgrid)
    assert (ds.sigma_s.values != 0.0).any()


def test_afgl_1986_rad_profile_has_scattering_true(mode_ckd, test_ckd_si_550):
    """
    When 'has_scattering' is True, the scattering coefficient is computed
    and is not zero everywhere at 550 nm.
    """
    profile = AFGL1986RadProfile(
        absorption_dataset="ckd/absorption/10nm/afgl_1986-us_standard-10nm-v3.nc",
        has_scattering=True,
    )
    assert profile.has_scattering
    ds = profile.eval_dataset(test_ckd_si_550, profile.zgrid)
    assert (ds.sigma_s.values != 0.0).any()


def test_afgl_1986_rad_profile_has_scattering_false(mode_ckd, test_ckd_si_550):
    """
    When 'has_scattering' is False, the scattering coefficient is not
    computed and is zero everywhere.
    """
    profile = AFGL1986RadProfile(
        absorption_dataset="ckd/absorption/10nm/afgl_1986-us_standard-10nm-v3.nc",
        has_scattering=False,
    )
    assert not profile.has_scattering
    ds = profile.eval_dataset(test_ckd_si_550, profile.zgrid)
    assert (ds.sigma_s.values == 0.0).all()


@pytest.mark.parametrize(
    "model_id",
    [
        "tropical",
        "midlatitude_summer",
        "midlatitude_winter",
        "subarctic_summer",
        "subarctic_winter",
        "us_standard",
    ],
)
def test_afgl_1986_rad_profile_model_id(mode_ckd, model_id):
    """
    All models are supported in ckd mode.
    """
    AFGL1986RadProfile(
        absorption_dataset=f"ckd/absorption/10nm/afgl_1986-{model_id}-10nm-v3.nc",
        thermoprops=dict(model_id=model_id),
    )


@pytest.mark.parametrize(
    "molecule",
    ["N2O", "CO", "CH4", "O2"],
)
def test_afgl_1986_rad_profile_concentrations_ckd_not_implemented(mode_ckd, molecule):
    """
    Concentrations rescaling is not implemented for molecules other than
    H2O and O3 in CKD mode.
    """
    with pytest.raises(NotImplementedError):
        AFGL1986RadProfile(
            absorption_dataset="ckd/absorption/10nm/afgl_1986-us_standard-10nm-v3.nc",
            thermoprops=dict(concentrations={molecule: 0.0 * ureg.dimensionless}),
        )


@pytest.mark.parametrize(
    "w", [280.0, 550.0, 790.0, 1040.0, 1270.0, 1590.0, 2220.0, 2400.0] * ureg.nm
)
@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_ckd_10nm(mode_ckd, w, model_id):
    """
    Can evaluate absorption coefficient.
    """
    profile = AFGL1986RadProfile(
        absorption_dataset="ckd/absorption/10nm/afgl_1986-us_standard-10nm-v3.nc",
        thermoprops=dict(model_id=model_id),
    )
    si = SpectralIndex.new(w=w)
    sigma_a = profile.eval_sigma_a(si, profile.zgrid)
    assert isinstance(sigma_a, pint.Quantity)


def test_afgl_1986_eval_zgrid(mode_ckd):
    """
    Evaluation on an arbitrary altitude grid works.
    """
    profile = AFGL1986RadProfile(
        absorption_dataset="ckd/absorption/10nm/afgl_1986-us_standard-10nm-v3.nc"
    )
    si = SpectralIndex.new()
    zgrid = ZGrid(levels=np.linspace(0, 100, 1001) * ureg.km)
    ds = profile.eval_dataset(si, zgrid)
    assert len(ds.z_layer) == len(zgrid.layers)
