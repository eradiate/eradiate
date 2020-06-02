from eradiate.scenes.lithosphere import Lambertian


def test_lambertian(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Default constructor
    ls = Lambertian()

    # Check if produced scene can be instanitated
    dict_scene = ls.add_to({"type": "scene"})
    assert load_dict(dict_scene) is not None

    # Constructor with arguments
    ls = Lambertian(
        reflectance=0.35,
        width=2e4,
    )

    # Check if produced scene can be instanitated
    dict_scene = ls.add_to({"type": "scene"})
    assert load_dict(dict_scene) is not None