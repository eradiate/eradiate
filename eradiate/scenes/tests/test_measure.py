import numpy as np
import pytest

from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure import DistantMeasure, PerspectiveCameraMeasure, RadianceMeterHsphereMeasure, RadianceMeterPPlaneMeasure
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


def test_hemispherical_repack(mode_mono):

    data = np.linspace(0, 1, 9*36)

    d = RadianceMeterHsphereMeasure(zenith_res=10, azimuth_res=10)
    data_reshaped = d.repack_results(data)

    assert np.shape(data_reshaped) == (9, 36)


def test_hemispherical_hsphere_selection(mode_mono):
    # Test hemisphere selection
    d = RadianceMeterHsphereMeasure()
    d_back = RadianceMeterHsphereMeasure(hemisphere="back")

    directions_front = d.directions()
    directions_back = d_back.directions()

    assert np.allclose(-directions_front, directions_back)

def test_pplane_class(mode_mono):
    d = RadianceMeterPPlaneMeasure()
    assert KernelDict.empty().add(d).load() is not None


def test_pplane_repack(mode_mono):
    data = np.array([0, 1, 2, 3])
    d = RadianceMeterPPlaneMeasure(zenith_res=45)
    data_repacked = d.repack_results(data)
    assert np.allclose(data_repacked, [[0, 1], [2, 3]], atol=0)


@pytest.mark.slow
def test_pplane_orientation(mode_mono):
    """To ensure that the principal plane sensor recovers the correct values, render two scenes:
    Once with a hemispherical view and once with a pplane view. Select the values corresponding to
    the pplane from the hemispherical dataset and compare with the pplane data."""

    from eradiate.scenes.core import SceneElementFactory, KernelDict

    surface_config = {
        "type": "rpv",
        "width": 800,
        "width_units": "km"
    }
    illumination_config = {
        "type": "directional",
        "zenith": 45,
        "irradiance": {
            "type": "solar_irradiance"
        }
    }
    hemisphere_config = {
        "type": "radiancemeter_hsphere",
        "zenith_res": 1,
        "azimuth_res": 1,
        "origin": [0, 0, 1],
        "direction": [0, 0, 1],
        "orientation": [1, 0, 0],
        "hemisphere": "back",
        "spp": 32
    }
    pplane_config = {
        "type": "radiancemeter_pplane",
        "zenith_res": 1,
        "origin": [0, 0, 1],
        "direction": [0, 0, 1],
        "orientation": [1, 0 ,0],
        "hemisphere": "back",
        "spp" : 32
    }
    factory = SceneElementFactory()

    surface = factory.create(surface_config)
    illumination = factory.create(illumination_config)
    hemisphere = factory.create(hemisphere_config)
    pplane = factory.create(pplane_config)

    # prepare the scene with hemispherical sensor and render
    kernel_dict_hemi = KernelDict.empty()
    kernel_dict_hemi.add([surface, illumination, hemisphere])
    scene_hemi = kernel_dict_hemi.load()
    sensor_hemi = scene_hemi.sensors()[0]
    scene_hemi.integrator().render(scene_hemi, sensor_hemi)
    film_hemi = sensor_hemi.film()
    result_hemi = hemisphere.repack_results(np.array(film_hemi.bitmap(), dtype=float))

    # prepare the scene with pplane sensor and render
    kernel_dict_pplane = KernelDict.empty()
    kernel_dict_pplane.add([surface, illumination, pplane])
    scene_pplane = kernel_dict_pplane.load()
    sensor_pplane = scene_pplane.sensors()[0]
    scene_pplane.integrator().render(scene_pplane, sensor_pplane)
    film_pplane = sensor_pplane.film()
    result_pplane = pplane.repack_results(np.squeeze(np.array(film_pplane.bitmap(), dtype=float)))

    # select the data to compare
    reshaped_hemi = np.concatenate([np.squeeze(result_hemi[::-1, 180, ]), np.squeeze(result_hemi[:, 0])])
    reshaped_pplane = np.concatenate([np.squeeze(result_hemi[::-1, 180, ]), np.squeeze(result_hemi[:, 0])])

    assert np.allclose(reshaped_hemi, reshaped_pplane, rtol=1e-9)

