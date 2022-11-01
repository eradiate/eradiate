from pathlib import Path

import pytest

from eradiate import data
from eradiate import unit_registry as ureg
from eradiate.ckd import Bin
from eradiate.contexts import CKDSpectralContext, MonoSpectralContext
from eradiate.exceptions import DataError
from eradiate.quad import Quad
from eradiate.scenes.measure import Measure, MeasureSpectralConfig
from eradiate.scenes.measure._core import (
    CKDMeasureSpectralConfig,
    MonoMeasureSpectralConfig,
    _active,
)
from eradiate.scenes.spectra import InterpolatedSpectrum, UniformSpectrum


def test_active(mode_mono):
    # Bins are unconditionally selected when using a uniform SRF
    assert _active(
        Bin(id=id, wmin=545.0, wmax=555.0, quad=Quad.new("gauss_legendre", 16)),
        UniformSpectrum(value=1.0),
    )

    # Bins are selected when included in the support of an interpolated SRF
    assert _active(
        Bin(id=id, wmin=545.0, wmax=555.0, quad=Quad.new("gauss_legendre", 16)),
        InterpolatedSpectrum(wavelengths=[500.0, 600.0] * ureg.nm, values=[1.0, 1.0]),
    )

    # Bins are selected when partially overlapping the support of an
    # interpolated SRF
    assert _active(
        Bin(id=id, wmin=480.0, wmax=510.0, quad=Quad.new("gauss_legendre", 16)),
        InterpolatedSpectrum(wavelengths=[500.0, 600.0] * ureg.nm, values=[1.0, 1.0]),
    )

    # Bins are not selected when not overlapping the support of an interpolated
    # SRF
    assert not _active(
        Bin(id=id, wmin=610.0, wmax=615.0, quad=Quad.new("gauss_legendre", 16)),
        InterpolatedSpectrum(wavelengths=[500.0, 600.0] * ureg.nm, values=[1.0, 1.0]),
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

    # Selecting wavelengths outside SRF range raises
    with pytest.raises(ValueError):
        MeasureSpectralConfig.new(
            wavelengths=[500.0, 600.0],
            srf={
                "type": "interpolated",
                "wavelengths": [300.0, 400.0],
                "values": [1.0, 1.0],
            },
        )

    # Works with at least one wavelength in SRF range
    cfg = MeasureSpectralConfig.new(
        wavelengths=[300.0, 550.0],
        srf={
            "type": "interpolated",
            "wavelengths": [500.0, 600.0],
            "values": [1.0, 1.0],
        },
    )

    # Off-SRF values are filtered out
    assert len(cfg.spectral_ctxs()) == 1


def test_mono_spectral_config_srf(modes_all_mono, tmpdir):
    """
    A SRF is loaded from the data store/a local file.
    """
    # existing prepared SRF
    assert MeasureSpectralConfig.new(wavelengths=[650.0], srf="sentinel_2a-msi-4")

    # raw SRF
    assert MeasureSpectralConfig.new(wavelengths=[650.0], srf="sentinel_2a-msi-4-raw")

    # local SRF file
    ds = data.load_dataset("spectra/srf/sentinel_2a-msi-4.nc")
    tmpfile = Path(tmpdir / "srf.nc")
    ds.to_netcdf(tmpfile)
    assert MeasureSpectralConfig.new(wavelengths=[650.0], srf=tmpfile)


def test_mono_spectral_config_srf_invalid(modes_all_mono):
    """
    An invalid SRF value raises a DataError.
    """
    with pytest.raises(DataError):
        MeasureSpectralConfig.new(srf="file_does_not_exist.nc")

    with pytest.raises(DataError):
        MeasureSpectralConfig.new(srf="srf-not-in-data-store-raw")


def test_ckd_spectral_config(modes_all_ckd):
    """
    Unit tests for :class:`.MeasureSpectralConfig` and child classes.
    """
    # The new() class method constructor selects an appropriate config class
    # depending on the active mode
    cfg = MeasureSpectralConfig.new(bin_set="10nm", bins=["540", "550"])
    assert isinstance(cfg, CKDMeasureSpectralConfig)

    # Generated spectral contexts are of the appropriate type and in correct numbers
    ctxs = cfg.spectral_ctxs()
    assert len(ctxs) == 32
    assert all(isinstance(ctx, CKDSpectralContext) for ctx in ctxs)

    # In CKD mode, we can also select bins with an interval specification
    cfg = MeasureSpectralConfig.new(
        bin_set="10nm",
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
