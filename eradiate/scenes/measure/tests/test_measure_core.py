from eradiate.contexts import CKDSpectralContext, MonoSpectralContext
from eradiate.scenes.measure._core import (
    CKDMeasureSpectralConfig,
    Measure,
    MeasureSpectralConfig,
    MonoMeasureSpectralConfig,
)


def test_mono_spectral_config(modes_all_mono):
    """
    Unit tests for :class:`.MonoMeasureSpectralConfig`.
    """
    # The new() class method constructor selects an appropriate config class
    # depending on the active mode
    cfg = MeasureSpectralConfig.new(wavelengths=[500.0, 600.0])
    assert isinstance(cfg, MonoMeasureSpectralConfig)

    # Generated spectral contexts are of the appropriate type and in correct numbers
    ctxs = cfg.spectral_ctxs()
    assert len(ctxs) == 2
    assert all(isinstance(ctx, MonoSpectralContext) for ctx in ctxs)


def test_ckd_spectral_config(modes_all_ckd):
    """
    Unit tests for :class:`.MeasureSpectralConfig` and child classes.
    """
    # The new() class method constructor selects an appropriate config class
    # depending on the active mode
    cfg = MeasureSpectralConfig.new(bin_set="10nm_test", bins=["540", "550"])
    assert isinstance(cfg, CKDMeasureSpectralConfig)

    # Generated spectral contexts are of the appropriate type and in correct numbers
    ctxs = cfg.spectral_ctxs()
    assert len(ctxs) == 32
    assert all(isinstance(ctx, CKDSpectralContext) for ctx in ctxs)

    # In CKD mode, we can also select bins with an interval specification
    cfg = MeasureSpectralConfig.new(
        bin_set="10nm_test",
        bins=[
            ("interval", {"wmin": 500.0, "wmax": 520.0}),
            {"type": "interval", "filter_kwargs": {"wmin": 530.0, "wmax": 550.0}},
            "575",
        ],
    )
    assert isinstance(cfg, CKDMeasureSpectralConfig)
    ctxs = cfg.spectral_ctxs()
    assert len(ctxs) == 96
    assert all(isinstance(ctx, CKDSpectralContext) for ctx in ctxs)


def test_measure_spp_splitting(mode_mono):
    """
    Unit tests for SPP splitting.
    """

    class MyMeasure(Measure):
        @property
        def film_resolution(self):
            return (32, 32)

        def kernel_dict(self, ctx):
            pass

    m = MyMeasure(id="my_measure", spp=256, split_spp=100)
    assert m._sensor_spps() == [100, 100, 56]
    assert m._sensor_ids() == ["my_measure_spp0", "my_measure_spp1", "my_measure_spp2"]
