import enoki as ek
import pytest

import eradiate.util.view as view


def test_bsdf_wrapper(variant_scalar_mono):
    """Test the Mitsuba BSDF-Plugin wrapper, by instantiating the wrapper class
    and calling its evaluate method with different values"""
    from eradiate.kernel.core.xml import load_dict

    bsdf_dict = {
        'type': 'diffuse',
        "reflectance": {"type": "uniform", "value": 0.5}
    }

    with pytest.raises(TypeError):
        wrapper = view.MitsubaBSDFPluginAdapter(bsdf_dict)

    bsdf = load_dict(bsdf_dict)
    bsdf_wrapper = view.MitsubaBSDFPluginAdapter(bsdf)

    assert ek.allclose(bsdf_wrapper.evaluate(
        [0, 0, 1], [1, 1, 1], 550), 0.15915, atol=1e-4)
    assert ek.allclose(bsdf_wrapper.evaluate(
        [30, 60], [0, 0, 1], 550), 0.15915, atol=1e-4)
