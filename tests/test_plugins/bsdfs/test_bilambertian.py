import pytest
import enoki as ek


def test_instantiation(variant_scalar_rgb):
    from mitsuba.render import BSDFFlags
    from mitsuba.core import load_dict

    b = load_dict({"type": "bilambertian"})
    assert b is not None
    assert b.component_count() == 2
    expected_flags_reflection = (
        BSDFFlags.DiffuseReflection | BSDFFlags.FrontSide | BSDFFlags.BackSide
    )
    expected_flags_transmission = (
        BSDFFlags.DiffuseTransmission | BSDFFlags.FrontSide | BSDFFlags.BackSide
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
    from mitsuba.render import BSDFContext, SurfaceInteraction3f
    from mitsuba.core import load_dict, Frame3f, ScalarVector3f

    albedo = r + t

    bsdf = load_dict({"type": "bilambertian", "reflectance": r, "transmittance": t})

    ctx = BSDFContext()

    si = SurfaceInteraction3f()
    si.p = [0, 0, 0]
    si.n = [0, 0, 1]

    for wi in [
        ScalarVector3f(x) for x in ([0, 0, 1], [0, 0, -1])
    ]:  # We try from both the front and back sides
        si.wi = wi
        si.sh_frame = Frame3f(si.n)
        print(f"wi = {wi}")

        for i in range(20):
            theta = i / 19.0 * ek.Pi  # We cover the entire circle

            wo = [ek.sin(theta), 0, ek.cos(theta)]
            print(f"wo = {wo}")
            v_pdf = bsdf.pdf(ctx, si, wo=wo)
            v_eval = bsdf.eval(ctx, si, wo=wo)
            print(f"v_eval = {v_eval}")

            if ek.dot(wi, wo) > 0:
                # reflection
                print(r * ek.abs(wo[2]) / ek.Pi)
                print(v_eval)
                assert ek.allclose(r * ek.abs(wo[2]) / ek.Pi, v_eval)
                assert ek.allclose(r / albedo * ek.abs(wo[2]) / ek.Pi, v_pdf)
            else:
                # transmission
                assert ek.allclose(t * ek.abs(wo[2]) / ek.Pi, v_eval)
                assert ek.allclose(t / albedo * ek.abs(wo[2]) / ek.Pi, v_pdf)


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
