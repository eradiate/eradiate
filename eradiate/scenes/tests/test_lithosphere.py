from eradiate.scenes import SceneDict
from eradiate.scenes.lithosphere import Lambertian


def test_lambertian(variant_scalar_mono):
    # Default constructor
    ls = Lambertian()

    # Check if produced scene can be instanitated
    scene_dict = SceneDict.empty()
    scene_dict.add(ls)
    assert scene_dict.load() is not None

    # Constructor with arguments
    ls = Lambertian.from_dict({"width": 1000., "reflectance": 0.3})

    # Check if produced scene can be instantiated
    assert SceneDict.empty().add(ls).load() is not None
