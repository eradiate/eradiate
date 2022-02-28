from eradiate.contexts import KernelDictContext
from eradiate.scenes.bsdfs import RPVBSDF


def test_rpv(modes_all_double):
    ctx = KernelDictContext()

    # Default constructor
    rpv_bsdf = RPVBSDF()
    assert rpv_bsdf.kernel_dict(ctx).load()

    # Construct from floats
    rpv_bsdf = RPVBSDF(rho_0=0.3, k=1.4, g=-0.23)
    assert rpv_bsdf.kernel_dict(ctx).load()

    # Construct from mixed spectrum types
    rpv_bsdf = RPVBSDF(
        rho_0=0.3,
        k={"type": "uniform", "value": 0.3},
        g={
            "type": "interpolated",
            "wavelengths": [300.0, 800.0],
            "values": [-0.23, 0.23],
        },
    )
    assert rpv_bsdf.kernel_dict(ctx).load()
