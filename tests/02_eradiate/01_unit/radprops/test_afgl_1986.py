import pint
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.contexts import SpectralContext
from eradiate.radprops import AFGL1986RadProfile


@pytest.fixture
def test_ckd_spectral_ctx_1650():
    spectral_cfg = eradiate.scenes.measure._core.CKDMeasureSpectralConfig(bins="1650")
    bindex = eradiate.ckd.Bindex(bin=spectral_cfg.bins[0], index=3)
    return eradiate.contexts.CKDSpectralContext(bindex=bindex, bin_set="10nm")


@pytest.fixture
def test_ckd_spectral_ctx_550():
    spectral_cfg = eradiate.scenes.measure._core.CKDMeasureSpectralConfig(bins="550")
    bindex = eradiate.ckd.Bindex(bin=spectral_cfg.bins[0], index=3)
    return eradiate.contexts.CKDSpectralContext(bindex=bindex, bin_set="10nm")


@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_default_ckd(mode_ckd, model_id):
    """
    Collision coefficient evaluation methods return pint.Quantity objects.
    """
    p = AFGL1986RadProfile(dict(model_id=model_id))

    spectral_ctx = SpectralContext.new(bin_set="10nm")
    for field in ["albedo", "sigma_a", "sigma_t"]:
        x = getattr(p, f"eval_{field}_ckd")(
            spectral_ctx.bindex, bin_set_id=spectral_ctx.bin_set.id
        )
        assert isinstance(x, ureg.Quantity)

    sigma_s = p.eval_sigma_s_ckd(spectral_ctx.bindex)
    assert isinstance(sigma_s, ureg.Quantity)


@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_has_absorption_default(
    mode_ckd, test_ckd_spectral_ctx_1650, model_id
):
    """
    Default value for 'has_absorption' is True, hence the absorption
    coefficient is computed and is not zero everywhere at 1650 nm.
    """
    p = AFGL1986RadProfile(dict(model_id=model_id))
    assert p.has_absorption
    ds = p.eval_dataset(test_ckd_spectral_ctx_1650)
    assert (ds.sigma_a.values != 0.0).any()


@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_has_absorption_true(
    mode_ckd, test_ckd_spectral_ctx_1650, model_id
):
    """
    When 'has_absorption' is True, the absorption coefficient is computed
    and is not zero everywhere at 1650 nm.
    """
    p = AFGL1986RadProfile(thermoprops=dict(model_id=model_id), has_absorption=True)
    assert p.has_absorption
    ds = p.eval_dataset(test_ckd_spectral_ctx_1650)
    assert (ds.sigma_a.values != 0.0).any()


@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_has_absorption_false(
    mode_ckd, test_ckd_spectral_ctx_1650, model_id
):
    """
    When 'has_absorption' is False, the absorption coefficient is not
    computed and is zero everywhere.
    """
    p = AFGL1986RadProfile(thermoprops=dict(model_id=model_id), has_absorption=False)
    assert not p.has_absorption
    ds = p.eval_dataset(test_ckd_spectral_ctx_1650)
    assert (ds.sigma_a.values == 0.0).all()


def test_afgl_1986_rad_profile_has_scattering_default(
    mode_ckd, test_ckd_spectral_ctx_550
):
    """
    Default value for 'has_scattering' is True, hence the absorption
    coefficient is computed and is not zero everywhere at 550 nm.
    """
    p = AFGL1986RadProfile()
    assert p.has_scattering
    ds = p.eval_dataset(test_ckd_spectral_ctx_550)
    assert (ds.sigma_s.values != 0.0).any()


def test_afgl_1986_rad_profile_has_scattering_true(mode_ckd, test_ckd_spectral_ctx_550):
    """
    When 'has_scattering' is True, the scattering coefficient is computed
    and is not zero everywhere at 550 nm.
    """
    p = AFGL1986RadProfile(has_scattering=True)
    assert p.has_scattering
    ds = p.eval_dataset(test_ckd_spectral_ctx_550)
    assert (ds.sigma_s.values != 0.0).any()


def test_afgl_1986_rad_profile_has_scattering_false(
    mode_ckd, test_ckd_spectral_ctx_550
):
    """
    When 'has_scattering' is False, the scattering coefficient is not
    computed and is zero everywhere.
    """
    p = AFGL1986RadProfile(has_scattering=False)
    assert not p.has_scattering
    ds = p.eval_dataset(test_ckd_spectral_ctx_550)
    assert (ds.sigma_s.values == 0.0).all()


@pytest.mark.parametrize(
    "model_id",
    [
        "midlatitude_winter",
        "subarctic_summer",
        "subarctic_winter",
        "tropical",
    ],
)
def test_afgl_1986_rad_profile_model_id_ckd_not_implemented(mode_ckd, model_id):
    """
    Models other than 'us_standard' or 'midlatitude_summer' are not implemented
    in ckd mode.
    """
    with pytest.raises(NotImplementedError):
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
            thermoprops=dict(concentrations={molecule: 1.0 * ureg.kg / ureg.m**2})
        )


@pytest.mark.parametrize(
    "bin", ["280", "550", "790", "1040", "1270", "1590", "2220", "2400"]
)
@pytest.mark.parametrize("model_id", ["midlatitude_summer", "us_standard"])
def test_afgl_1986_rad_profile_ckd_10nm(mode_ckd, bin, model_id):
    """
    Can evaluate absorption coefficient.
    """
    p = AFGL1986RadProfile(thermoprops=dict(model_id=model_id))
    bin = eradiate.scenes.measure._core.CKDMeasureSpectralConfig(bins=bin).bins[0]
    bindex = eradiate.ckd.Bindex(bin=bin, index=3)
    spectral_ctx = eradiate.contexts.CKDSpectralContext(bindex=bindex, bin_set="10nm")
    assert isinstance(p.eval_sigma_a(spectral_ctx), pint.Quantity)
