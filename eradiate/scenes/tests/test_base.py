import attr

from eradiate.scenes.base import SceneHelper, scene_dict_empty


@attr.s
class TinyDirectional(SceneHelper):
    DEFAULT_CONFIG = {"irradiance": 1.0}

    id = attr.ib(default="illumination")

    def kernel_dict(self, ref=True):
        return {
            self.id: {
                "type": "directional",
                "irradiance": self.config["irradiance"]
            }
        }


def test_scene_helper(variant_scalar_mono):
    from eradiate.kernel.core.xml import load_dict

    # Default constructor
    d = TinyDirectional()
    assert d.config == d.DEFAULT_CONFIG

    # Check that create scene can be instantiated
    assert load_dict(d.add_to(scene_dict_empty())) is not None

    # Construct using from_dict factory
    assert d == TinyDirectional.from_dict({})
