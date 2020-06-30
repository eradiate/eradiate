from eradiate.scenes import SceneDict
from eradiate.scenes.lithosphere import Lambertian, RPV


def test_lambertian(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Default constructor
    ls = Lambertian()

    # Check if produced scene can be instantiated
    scene_dict = SceneDict.empty()
    scene_dict.add(ls)
    assert scene_dict.load() is not None

    # Constructor with arguments
    ls = Lambertian.from_dict({"width": 1000., "reflectance": 0.3})

    # Check if produced scene can be instantiated
    assert SceneDict.empty().add(ls).load() is not None
    assert load_dict(scene_dict) is not None


def test_rpv(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Default constructor
    ls = RPV()

    # Check if produced scene can be instanitated
    scene_dict = SceneDict.empty()
    scene_dict.add(ls)
    assert scene_dict.load() is not None

    # Constructor with arguments
    ls = Lambertian.from_dict({"width": 1000., "rho_0": 0.3, "k": 1.4, "ttheta": -0.23})

    # Check if produced scene can be instantiated
    assert SceneDict.empty().add(ls).load() is not None
    assert load_dict(scene_dict) is not None
