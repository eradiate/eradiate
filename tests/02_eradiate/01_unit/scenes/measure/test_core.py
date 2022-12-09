from eradiate.scenes.measure import Measure


def test_spp_splitting(mode_mono):
    """
    Unit tests for SPP splitting.
    """

    class MyMeasure(Measure):
        @property
        def film_resolution(self):
            return (32, 32)

        def _kernel_dict_impl(self, sensor_id, spp):
            pass

    m = MyMeasure(id="my_measure", spp=256, split_spp=100)
    assert m._sensor_spps() == [100, 100, 56]
    assert m._sensor_ids() == ["my_measure_spp0", "my_measure_spp1", "my_measure_spp2"]
