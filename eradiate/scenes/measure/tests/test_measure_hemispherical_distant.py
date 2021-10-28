import numpy as np

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure._hemispherical_distant import HemisphericalDistantMeasure


def test_measure_hemispherical_distant(modes_all):
    # Test default constructor
    d = HemisphericalDistantMeasure()
    ctx = KernelDictContext()
    assert KernelDict.from_elements(d, ctx=ctx).load() is not None

    # Test target support
    # -- Target a point
    d = HemisphericalDistantMeasure(target=[0, 0, 0])
    assert KernelDict.from_elements(d, ctx=ctx).load() is not None

    # -- Target an axis-aligned rectangular patch
    d = HemisphericalDistantMeasure(
        target={"type": "rectangle", "xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    )
    assert KernelDict.from_elements(d, ctx=ctx).load() is not None


def test_measure_hemispherical_distant_viewing_angles(mode_mono):
    # Test default constructor
    d = HemisphericalDistantMeasure(film_resolution=(2, 2))
    expected = (
        np.array(
            [
                [
                    (41.409622, 225),
                    (41.409622, 135),
                ],
                [
                    (41.409622, 315),
                    (41.409622, 45),
                ],
            ]
        )
        * ureg.deg
    )
    assert np.allclose(expected, d.viewing_angles)
