import numpy as np
import pint
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.contexts import SpectralContext
from eradiate.radprops import AFGL1986RadProfile, ZGrid


@pytest.fixture
def test_ckd_spectral_ctx_1650():
    spectral_cfg = eradiate.scenes.measure.MeasureSpectralConfig.new(bins="1650")
    bindex = eradiate.ckd.Bindex(bin=spectral_cfg.bins[0], index=3)
    return SpectralContext.new(bindex=bindex, bin_set="10nm")


@pytest.fixture
def test_ckd_spectral_ctx_550():
    spectral_cfg = eradiate.scenes.measure.MeasureSpectralConfig.new(bins="550")
    bindex = eradiate.ckd.Bindex(bin=spectral_cfg.bins[0], index=3)
    return SpectralContext.new(bindex=bindex, bin_set="10nm")


@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_default_ckd(mode_ckd, model_id):
    """
    Collision coefficient evaluation methods return pint.Quantity objects.
    """
    profile = AFGL1986RadProfile({"model_id": model_id})

    spectral_ctx = SpectralContext.new(bin_set="10nm")
    for field in ["albedo", "sigma_a", "sigma_t"]:
        x = getattr(profile, f"eval_{field}_ckd")([spectral_ctx.bindex], profile.zgrid)
        assert isinstance(x, ureg.Quantity)

    sigma_s = profile.eval_sigma_s_ckd([spectral_ctx.bindex], profile.zgrid)
    assert isinstance(sigma_s, ureg.Quantity)


@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_has_absorption_default(
    mode_ckd, test_ckd_spectral_ctx_1650, model_id
):
    """
    Default value for 'has_absorption' is True, hence the absorption
    coefficient is computed and is not zero everywhere at 1650 nm.
    """
    profile = AFGL1986RadProfile(dict(model_id=model_id))
    assert profile.has_absorption
    ds = profile.eval_dataset(test_ckd_spectral_ctx_1650)
    assert (ds.sigma_a.values != 0.0).any()


@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_has_absorption_true(
    mode_ckd, test_ckd_spectral_ctx_1650, model_id
):
    """
    When 'has_absorption' is True, the absorption coefficient is computed
    and is not zero everywhere at 1650 nm.
    """
    profile = AFGL1986RadProfile(
        thermoprops=dict(model_id=model_id), has_absorption=True
    )
    assert profile.has_absorption
    ds = profile.eval_dataset(test_ckd_spectral_ctx_1650)
    assert (ds.sigma_a.values != 0.0).any()


@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_has_absorption_false(
    mode_ckd, test_ckd_spectral_ctx_1650, model_id
):
    """
    When 'has_absorption' is False, the absorption coefficient is not
    computed and is zero everywhere.
    """
    profile = AFGL1986RadProfile(
        thermoprops=dict(model_id=model_id), has_absorption=False
    )
    assert not profile.has_absorption
    ds = profile.eval_dataset(test_ckd_spectral_ctx_1650)
    assert (ds.sigma_a.values == 0.0).all()


def test_afgl_1986_rad_profile_has_scattering_default(
    mode_ckd, test_ckd_spectral_ctx_550
):
    """
    Default value for 'has_scattering' is True, hence the absorption
    coefficient is computed and is not zero everywhere at 550 nm.
    """
    profile = AFGL1986RadProfile()
    assert profile.has_scattering
    ds = profile.eval_dataset(test_ckd_spectral_ctx_550)
    assert (ds.sigma_s.values != 0.0).any()


def test_afgl_1986_rad_profile_has_scattering_true(mode_ckd, test_ckd_spectral_ctx_550):
    """
    When 'has_scattering' is True, the scattering coefficient is computed
    and is not zero everywhere at 550 nm.
    """
    profile = AFGL1986RadProfile(has_scattering=True)
    assert profile.has_scattering
    ds = profile.eval_dataset(test_ckd_spectral_ctx_550)
    assert (ds.sigma_s.values != 0.0).any()


def test_afgl_1986_rad_profile_has_scattering_false(
    mode_ckd, test_ckd_spectral_ctx_550
):
    """
    When 'has_scattering' is False, the scattering coefficient is not
    computed and is zero everywhere.
    """
    profile = AFGL1986RadProfile(has_scattering=False)
    assert not profile.has_scattering
    ds = profile.eval_dataset(test_ckd_spectral_ctx_550)
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
    AFGL1986RadProfile(thermoprops=dict(model_id=model_id))


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
            thermoprops=dict(concentrations={molecule: 0.0 * ureg.dimensionless})
        )


@pytest.mark.parametrize(
    "bin", ["280", "550", "790", "1040", "1270", "1590", "2220", "2400"]
)
@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_ckd_10nm(mode_ckd, bin, model_id):
    """
    Can evaluate absorption coefficient.
    """
    profile = AFGL1986RadProfile(thermoprops=dict(model_id=model_id))
    bin = eradiate.scenes.measure.MeasureSpectralConfig.new(bins=bin).bins[0]
    bindex = eradiate.ckd.Bindex(bin=bin, index=3)
    spectral_ctx = SpectralContext.new(bindex=bindex, bin_set="10nm")
    assert isinstance(profile.eval_sigma_a(spectral_ctx, profile.zgrid), pint.Quantity)


def test_afgl_1986_eval_zgrid(mode_ckd):
    """
    Evaluation on an arbitrary altitude grid works.
    """
    profile = AFGL1986RadProfile()
    spectral_ctx = SpectralContext.new()
    zgrid = ZGrid(levels=np.linspace(0, 100, 1001) * ureg.km)
    ds = profile.eval_dataset(spectral_ctx, zgrid)
    assert len(ds.z_layer) == len(zgrid.layers)
