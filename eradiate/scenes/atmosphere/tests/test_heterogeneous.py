from eradiate.data import presolver
from eradiate.scenes.atmosphere.heterogeneous import HeterogeneousAtmosphere
from eradiate.scenes.core import KernelDict
from eradiate.util.collections import onedict_value
from eradiate.util.units import kernel_default_units, ureg


def test_heterogeneous(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # Test default constructor
    a = HeterogeneousAtmosphere(
        width=ureg.Quantity(100., ureg.km),
        height=ureg.Quantity(100., ureg.km),
        sigma_t=presolver.resolve("tests/textures/heterogeneous_atmosphere_mono/sigma_t.vol"),
        albedo=presolver.resolve("tests/textures/heterogeneous_atmosphere_mono/albedo.vol")
    )

    # Check if default output can be loaded
    p = a.phase()
    assert load_dict(onedict_value(p)) is not None

    m = a.media()
    assert load_dict(onedict_value(m)) is not None

    s = a.shapes()
    assert load_dict(onedict_value(s)) is not None

    # Load all elements at once (and use references)
    with kernel_default_units.override({"length": "km"}):
        kernel_dict = KernelDict.empty()
        kernel_dict.add(a)
        scene = kernel_dict.load()
        assert scene is not None
