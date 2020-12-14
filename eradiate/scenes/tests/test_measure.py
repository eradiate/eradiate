import numpy as np
import pytest

from eradiate.scenes.core import KernelDict
from eradiate.scenes.illumination import DirectionalIllumination
from eradiate.scenes.surface import RPVSurface
from eradiate.scenes.measure import DistantMeasure, PerspectiveCameraMeasure, \
    RadianceMeterHsphereMeasure, RadianceMeterPlaneMeasure
from eradiate.scenes.spectra import SolarIrradianceSpectrum
from eradiate.util.units import ureg


def test_distant(mode_mono):
    # Constructor
    d = DistantMeasure()
    assert KernelDict.empty().add(d).load() is not None


def test_perspective(mode_mono):
    # Constructor
    d = PerspectiveCameraMeasure()
    assert KernelDict.empty().add(d).load() is not None


def test_radiancemeter_hemispherical(mode_mono):
    # Test constructor
    d = RadianceMeterHsphereMeasure()
    assert KernelDict.empty().add(d).load() is not None


def test_hemispherical_postprocess(mode_mono):
    """To test the postprocess method, we create a data dictionary, mapping
    a two sensor_ids to results. One ID will map to the expected results,
     the other will map to results of a different shape, prompting the test to
     fail, should these results be read. This way we can assert that the correct
    data are looked up and then reshaped, as expected."""
    sensor_id = ["test_sensor"]
    spp = [1]
    data = {"test_sensor": np.linspace(0, 1, 9 * 36),
            "test_wrong_sensor": np.linspace(0, 1, 18 * 72)}

    d = RadianceMeterHsphereMeasure(zenith_res=10, azimuth_res=10)
    data_reshaped = d.postprocess_results(sensor_id, spp, data)

    assert np.shape(data_reshaped) == (9, 36)


def test_hemispherical_hsphere_selection(mode_mono):
    # Test hemisphere selection
    d = RadianceMeterHsphereMeasure()
    d_back = RadianceMeterHsphereMeasure(hemisphere="back")

    directions_front = d._directions()
    directions_back = d_back._directions()

    assert np.allclose(-directions_front, directions_back)

def test_plane_class(mode_mono):
    d = RadianceMeterPlaneMeasure()
    assert KernelDict.empty().add(d).load() is not None


def test_plane_postprocess(mode_mono):
    """To test the postprocess method, we create a data dictionary, mapping
    a three sensor_ids to results. Two IDs will map to the expected results,
     the other will map to results of a different shape, prompting the test to
     fail, should these results be read. This way we can assert that the correct
    data are looked up and then reshaped, as expected."""
    sensor_id = ["test_sensor", "test_sensor2"]
    spp = [1,1]
    data = {"test_sensor": np.linspace(0, 3, 2 * 2),
            "test_sensor2": np.linspace(0, 3, 2 * 2),
            "test_sensor_wrong": np.linspace(0, 1, 18 * 72)}
    d = RadianceMeterPlaneMeasure(zenith_res=45)
    data_repacked = d.postprocess_results(sensor_id, spp, data)
    assert np.allclose(data_repacked, [[0, 1], [2, 3]], atol=0)


@pytest.mark.slow
def test_plane_orientation(mode_mono):
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
    reshaped_hemi = np.concatenate([np.squeeze(result_hemi[::-1, 180, ]), np.squeeze(result_hemi[:, 0])])
    reshaped_pplane = np.concatenate([np.squeeze(result_pplane[::-1, 1]), np.squeeze(result_pplane[:, 0])])

    assert np.allclose(reshaped_hemi, reshaped_pplane, rtol=1e-9)

