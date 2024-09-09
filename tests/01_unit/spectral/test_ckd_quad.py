from eradiate.quad import QuadType
from eradiate.spectral.ckd_quad import CKDQuadConfig, CKDQuadPolicy


def test_ckd_quad_config_construct():
    cqc = CKDQuadConfig()
    assert isinstance(cqc.type, QuadType)
    assert isinstance(cqc.ng_max, int)
    assert isinstance(cqc.policy, CKDQuadPolicy)


def test_ckd_quad_config_convert():
    cqc = CKDQuadConfig.convert(
        {"type": "gauss_lobatto", "ng_max": 8, "policy": "fixed"}
    )
    assert cqc.type is QuadType.GAUSS_LOBATTO
    assert cqc.ng_max == 8
    assert cqc.policy is CKDQuadPolicy.FIXED
