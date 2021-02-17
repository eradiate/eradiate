import enoki as ek
import numpy as np
import pytest

import eradiate
from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.scenes.core import KernelDict
from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.scenes.measure import (
    DistantMeasure,
    PerspectiveCameraMeasure,
    RadianceMeterHsphereMeasure,
    RadianceMeterPlaneMeasure,
    Target,
    TargetPoint,
    TargetRectangle
)
from eradiate.scenes.spectra import SolarIrradianceSpectrum
from eradiate.scenes.surface import RPVSurface


def test_target(mode_mono):
    from mitsuba.core import Point3f
    # TargetPoint: basic constructor
    with ucc.override({"length": "km"}):
        t = TargetPoint([0, 0, 0])
        assert t.xyz.units == ureg.km

    with pytest.raises(ValueError):
        TargetPoint(0)

    # TargetPoint: check kernel item
    with ucc.override({"length": "km"}), uck.override({"length": "m"}):
        t = TargetPoint([1, 2, 0])
        assert ek.allclose(t.kernel_item(), [1000, 2000, 0])

    # TargetRectangle: basic constructor
    with ucc.override({"length": "km"}):
        t = TargetRectangle(0, 1, 0, 1)
        assert t.xmin == 0. * ureg.km
        assert t.xmax == 1. * ureg.km
        assert t.ymin == 0. * ureg.km
        assert t.ymax == 1. * ureg.km

    with ucc.override({"length": "m"}):
        t = TargetRectangle(0, 1, 0, 1)
        assert t.xmin == 0. * ureg.m
        assert t.xmax == 1. * ureg.m
        assert t.ymin == 0. * ureg.m
        assert t.ymax == 1. * ureg.m

    with pytest.raises(ValueError):
        TargetRectangle(0, 1, "a", 1)

    with pytest.raises(ValueError):
        TargetRectangle(0, 1, 1, -1)

    # TargetRectangle: check kernel item
    t = TargetRectangle(-1, 1, -1, 1)

    with uck.override({"length": "mm"}):  # Tricky: we can't compare transforms directly
        kernel_item = t.kernel_item()["to_world"]
        assert ek.allclose(
            kernel_item.transform_point(Point3f(-1, -1, 0)), [-1000, -1000, 0]
        )
        assert ek.allclose(
            kernel_item.transform_point(Point3f(1, 1, 0)), [1000, 1000, 0]
        )
        assert ek.allclose(
            kernel_item.transform_point(Point3f(1, 1, 42)), [1000, 1000, 42]
        )

    # Factory: basic test
    with ucc.override({"length": "m"}):
        t = Target.new("point", xyz=[1, 1, 0])
        assert isinstance(t, TargetPoint)
        assert np.allclose(t.xyz, ureg.Quantity([1, 1, 0], ureg.m))

        t = Target.new("rectangle", 0, 1, 0, 1)
        assert isinstance(t, TargetRectangle)

    # Converter: basic test
    with ucc.override({"length": "m"}):
        t = Target.convert({"type": "point", "xyz": [1, 1, 0]})
        assert isinstance(t, TargetPoint)
        assert np.allclose(t.xyz, ureg.Quantity([1, 1, 0], ureg.m))

        t = Target.convert([1, 1, 0])
        assert isinstance(t, TargetPoint)
        assert np.allclose(t.xyz, ureg.Quantity([1, 1, 0], ureg.m))

        with pytest.raises(ValueError):
            Target.convert({"xyz": [1, 1, 0]})


def test_distant(mode_mono):
    # Test default constructor
    d = DistantMeasure()
    assert KernelDict.empty().add(d).load() is not None

    # Test target support
    # -- Target a point
    d = DistantMeasure(target=[0, 0, 0])
    assert KernelDict.empty().add(d).load() is not None

    # -- Target an axis-aligned rectangular patch
    d = DistantMeasure(target={
        "type": "rectangle", "xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1
    })
    assert KernelDict.empty().add(d).load() is not None


def test_perspective(mode_mono):
    # Constructor
    d = PerspectiveCameraMeasure()
    assert KernelDict.empty().add(d).load() is not None

    # origin and target cannot be the same
    for point in [[0, 0, 0], [1, 1, 1], [-1, 0.5, 1.3333]]:
        with pytest.raises(ValueError):
            d = PerspectiveCameraMeasure(origin=point, target=point)

    # up must differ from the camera's viewing direction
    with pytest.raises(ValueError):
        d = PerspectiveCameraMeasure(origin=[0, 1, 0],
                                     target=[1, 0, 0],
                                     up=[1, -1, 0])


def test_radiancemeter_hsphere_construct(mode_mono):
    # Test constructor
    d = RadianceMeterHsphereMeasure()
    assert KernelDict.empty().add(d).load() is not None


def test_radiancemeter_hsphere_postprocess(mode_mono):
    """To test the postprocess() method, we create a data dictionary mapping
    two sensor_ids to results. One ID will map to the expected results,
    the other will map to results of a different shape, prompting the test to
    fail, should these results be read. This way we can assert that the correct
    data are looked up and then reshaped, as expected.
    """
    sensor_id = ["test_sensor"]
    spp = [1]
    data = {
        "test_sensor": np.linspace(0, 1, 9 * 36),
        "test_wrong_sensor": np.linspace(0, 1, 18 * 72)
    }

    d = RadianceMeterHsphereMeasure(zenith_res=10, azimuth_res=10)
    data_reshaped = d.postprocess_results(sensor_id, spp, data)

    assert np.shape(data_reshaped) == (9, 36)


def test_radiancemeter_hsphere_hemisphere(mode_mono):
    # Test hemisphere selection
    d = RadianceMeterHsphereMeasure()
    d_back = RadianceMeterHsphereMeasure(hemisphere="back")

    directions_front = d._directions()
    directions_back = d_back._directions()

    assert np.allclose(-directions_front, directions_back)


def test_radiancemeter_hsphere_sensor_info():
    measure = RadianceMeterHsphereMeasure(spp=15)
    measure._spp_max_single = 10

    eradiate.set_mode("mono")
    assert measure.sensor_info() == [
        (f"{measure.id}_0", 10),
        (f"{measure.id}_1", 5)
    ]

    eradiate.set_mode("mono_double")
    assert measure.sensor_info() == [
        (f"{measure.id}", 15),
    ]


def test_radiancemeter_plane_construct(mode_mono):
    d = RadianceMeterPlaneMeasure()
    assert KernelDict.empty().add(d).load() is not None


def test_radiancemeter_plane_postprocess(mode_mono):
    """To test the postprocess method, we create a data dictionary, mapping
    three sensor_ids to results. Two IDs will map to the expected results,
    the other will map to results of a different shape, prompting the test to
    fail, should these results be read. This way we can assert that the correct
    data are looked up and then reshaped, as expected.
    """
    sensor_id = ["test_sensor_1", "test_sensor_2"]
    spp = [1, 1]
    data = {"test_sensor_1": np.linspace(0, 3, 2 * 2),
            "test_sensor_2": np.linspace(0, 3, 2 * 2),
            "test_sensor_wrong": np.linspace(0, 1, 18 * 72)}
    d = RadianceMeterPlaneMeasure(zenith_res=45)
    data_repacked = d.postprocess_results(sensor_id, spp, data)
    assert np.allclose(data_repacked, [[0, 1], [2, 3]], atol=0)


@pytest.mark.slow
def test_radiancemeter_plane_orientation(mode_mono):
    """To ensure that the principal plane sensor recovers the correct values, render two scenes:
    Once with a hemispherical view and once with a pplane view. Select the values corresponding to
    the pplane from the hemispherical dataset and compare with the pplane data."""

    measure_config = {
        "zenith_res": 9,
        "origin": [0, 0, 1],
        "direction": [0, 0, 1],
        "orientation": [1, 0, 0],
        "hemisphere": "back",
        "spp": 32
    }

    surface = RPVSurface(width=ureg.Quantity(800., ureg.km))
    illumination = DirectionalIllumination(zenith=45., irradiance=SolarIrradianceSpectrum())
    hemisphere = RadianceMeterHsphereMeasure(azimuth_res=1, **measure_config)
    pplane = RadianceMeterPlaneMeasure(**measure_config)

    # prepare the scene with hemispherical sensor and render
    kernel_dict_hemi = KernelDict.empty()
    kernel_dict_hemi.add([surface, illumination, hemisphere])
    scene_hemi = kernel_dict_hemi.load()
    sensor_hemi = scene_hemi.sensors()[0]
    scene_hemi.integrator().render(scene_hemi, sensor_hemi)
    film_hemi = sensor_hemi.film()
    data_hemi = {"hemi": np.array(film_hemi.bitmap(), dtype=float)}
    result_hemi = hemisphere.postprocess_results(["hemi"], [1], data_hemi)

    # prepare the scene with pplane sensor and render
    kernel_dict_pplane = KernelDict.empty()
    kernel_dict_pplane.add([surface, illumination, pplane])
    scene_pplane = kernel_dict_pplane.load()
    sensor_pplane = scene_pplane.sensors()[0]
    scene_pplane.integrator().render(scene_pplane, sensor_pplane)
    film_pplane = sensor_pplane.film()
    data_pplane = {"pplane": np.squeeze(np.array(film_pplane.bitmap(), dtype=float))}
    result_pplane = pplane.postprocess_results(["pplane"], [1], data_pplane)

    # select the data to compare
    reshaped_hemi = np.concatenate([
        np.squeeze(result_hemi[::-1, 180, ]),
        np.squeeze(result_hemi[:, 0])
    ])
    reshaped_pplane = np.concatenate([
        np.squeeze(result_pplane[::-1, 1]),
        np.squeeze(result_pplane[:, 0])
    ])

    assert np.allclose(reshaped_hemi, reshaped_pplane, rtol=1e-9)


def test_radiancemeter_plane_sensor_info():
    measure = RadianceMeterPlaneMeasure(spp=15)
    measure._spp_max_single = 10

    eradiate.set_mode("mono")
    assert measure.sensor_info() == [
        (f"{measure.id}_0", 10),
        (f"{measure.id}_1", 5)
    ]

    eradiate.set_mode("mono_double")
    assert measure.sensor_info() == [
        (f"{measure.id}", 15),
    ]
