from eradiate.scenes.base import scene_dict_empty
from eradiate.scenes.lithosphere import Lambertian


def test_lambertian(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Default constructor
    ls = Lambertian()

    # Check if produced scene can be instanitated
    scene_dict = ls.add_to(scene_dict_empty())
    assert load_dict(scene_dict) is not None

    # Constructor with arguments
    ls = Lambertian.from_dict({"width": 1000., "reflectance": 0.3})

    # Check if produced scene can be instantiated
    scene_dict = ls.add_to(scene_dict_empty())
    assert load_dict(scene_dict) is not None
