import drjit as dr
import mitsuba as mi
import pytest


def test_instantiation(variant_scalar_rgb):
    b = mi.load_dict({"type": "bilambertian"})
    assert b is not None
    assert b.component_count() == 2
    expected_flags_reflection = (
        mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
    )
    expected_flags_transmission = (
        mi.BSDFFlags.DiffuseTransmission
        | mi.BSDFFlags.FrontSide
        | mi.BSDFFlags.BackSide
    )
    assert expected_flags_reflection == b.flags(0)
    assert expected_flags_transmission == b.flags(1)


@pytest.mark.parametrize(
    "r, t",
    [
        (0.2, 0.4),
        (0.4, 0.2),
        (0.1, 0.9),
        (0.9, 0.1),
        (0.4, 0.6),
        (0.6, 0.4),
    ],
)
def test_eval_pdf(variant_scalar_rgb, r, t):
    albedo = r + t

    bsdf = mi.load_dict({"type": "bilambertian", "reflectance": r, "transmittance": t})

    ctx = mi.BSDFContext()

    si = mi.SurfaceInteraction3f()
    si.p = [0, 0, 0]
    si.n = [0, 0, 1]

    for wi in [
        mi.ScalarVector3f(x) for x in ([0, 0, 1], [0, 0, -1])
    ]:  # We try from both the front and back sides
        si.wi = wi
        si.sh_frame = mi.Frame3f(si.n)

        for i in range(20):
            theta = i / 19.0 * dr.Pi  # We cover the entire circle

            wo = [dr.sin(theta), 0, dr.cos(theta)]
            v_pdf = bsdf.pdf(ctx, si, wo=wo)
            v_eval = bsdf.eval(ctx, si, wo=wo)

            if dr.dot(wi, wo) > 0:
                # reflection
                assert dr.allclose(r * dr.abs(wo[2]) / dr.Pi, v_eval)
                assert dr.allclose(r / albedo * dr.abs(wo[2]) / dr.Pi, v_pdf)
            else:
                # transmission
                assert dr.allclose(t * dr.abs(wo[2]) / dr.Pi, v_eval)
                assert dr.allclose(t / albedo * dr.abs(wo[2]) / dr.Pi, v_pdf)


@pytest.mark.parametrize(
    "r, t",
    [
        [0.6, 0.2],
        [0.2, 0.6],
        [0.6, 0.4],
        [0.4, 0.6],
        [0.9, 0.1],
        [0.1, 0.9],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ],
)
def test_chi2(variant_llvm_rgb, r, t):
    from mitsuba.python.chi2 import BSDFAdapter, ChiSquareTest, SphericalDomain

    xml = f"""
        <spectrum name="reflectance" value="{r}"/>
        <spectrum name="transmittance" value="{t}"/>
    """

    sample_func, pdf_func = BSDFAdapter("bilambertian", xml)

    chi2 = ChiSquareTest(
        domain=SphericalDomain(),
        sample_func=sample_func,
        pdf_func=pdf_func,
        sample_dim=3,
    )

    assert chi2.run()
