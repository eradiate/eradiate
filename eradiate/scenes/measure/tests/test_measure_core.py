from eradiate.scenes.measure._core import Measure, SensorInfo


def test_measure(mode_mono):
    """
    Unit tests for :class:`.Measure`.
    """

    # Concrete class to test
    class MyMeasure(Measure):
        @property
        def film_resolution(self):
            return (640, 480)

        def _base_dicts(self):
            return [
                {
                    "type": "some_sensor",
                    "id": self.id,
                }
            ]

    m = MyMeasure()

    # This measure has a single sensor associated to it
    assert m.sensor_infos() == [SensorInfo(id=m.id, spp=m.spp)]

    # The kernel dict is well-formed
    m.spp = 256
    assert m.kernel_dict() == {
        m.id: {
            "type": "some_sensor",
            "id": m.id,
            "film": {
                "type": "hdrfilm",
                "width": 640,
                "height": 480,
                "pixel_format": "luminance",
                "component_format": "float32",
                "rfilter": {"type": "box"},
            },
            "sampler": {
                "type": "independent",
                "sample_count": 256,
            },
        }
    }


def test_measure_spp_splitting(mode_mono):
    """
    Unit tests for SPP splitting.
    """

    class MyMeasure(Measure):
        @property
        def film_resolution(self):
            return (32, 32)

        def _base_dicts(self):
            pass

    m = MyMeasure(id="my_measure", spp=256, spp_splitting_threshold=100)
    assert m._split_spp() == [100, 100, 56]
    assert m.sensor_infos() == [
        SensorInfo(id="my_measure_0", spp=100),
        SensorInfo(id="my_measure_1", spp=100),
        SensorInfo(id="my_measure_2", spp=56),
    ]

    # fmt: off
    assert m._film_dicts() == [{
        "film": {
            "type": "hdrfilm",
            "width": 32,
            "height": 32,
            "pixel_format": "luminance",
            "component_format": "float32",
            "rfilter": {"type": "box"},
        }
    }] * 3
    # fmt: on

    assert m._sampler_dicts() == [
        {"sampler": {"type": "independent", "sample_count": 100}},
        {"sampler": {"type": "independent", "sample_count": 100}},
        {"sampler": {"type": "independent", "sample_count": 56}},
    ]
