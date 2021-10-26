import enoki as ek
import numpy as np
import pytest

from eradiate._util import onedict_value
from eradiate.contexts import KernelDictContext
from eradiate.scenes.core import KernelDict
from eradiate.scenes.measure._distant import DistantFluxMeasure, DistantRadianceMeasure


def test_distant_radiance_postprocessing_mono(modes_all_mono):
    # We use a peculiar rectangular film size to make sure that we get dimensions
    # right
    d = DistantRadianceMeasure(film_resolution=(32, 16))

    # Add test data to results
    d.results.raw = {
        550.0: {"values": {"sensor": np.ones((16, 32, 1))}, "spp": {"sensor": 128}}
    }

    # Postprocessing succeeds and viewing angles have correct bounds
    ds = d.postprocess()
    assert "vza" in ds.coords
    assert np.allclose(ds.vza.min(), 5.06592926)  # Value calculated manually
    assert np.allclose(ds.vza.max(), 86.47273911)  # Value calculated manually
    assert "vaa" in ds.coords
    assert np.allclose(ds.vaa.min(), -177.09677419)  # Value calculated manually
    assert np.allclose(ds.vaa.max(), 177.09677419)  # Value calculated manually

    # We now move on to the plane case
    d._film_resolution = (32, 1)
    # Mismatched film size and raw data dimensions raises
    with pytest.raises(ValueError):
        d.postprocess()

    # Postprocessing succeeds and viewing angles have correct bounds
    d.results.raw = {
        550.0: {"values": {"sensor": np.ones((1, 32, 1))}, "spp": {"sensor": 128}}
    }
    ds = d.postprocess()
    assert "vza" in ds.coords
    assert np.allclose(ds.vza.min(), -87.1875)  # Value manually calculated
    assert np.allclose(ds.vza.max(), 87.1875)  # Value manually calculated
    assert "vaa" in ds.coords
    assert np.allclose(ds.vaa, 0.0)


def test_distant_radiance_postprocessing_ckd(modes_all_ckd):
    # We use a peculiar rectangular film size to make sure that we get dimensions
    # right
    d = DistantRadianceMeasure(
        film_resolution=(32, 16),
        spectral_cfg={"bin_set": "10nm_test", "bins": ["550"]},
    )
    quad = d.spectral_cfg.bin_set.quad

    # Add test data to results
    d.results.raw = {
        **{
            ("550", i): {
                "values": {"sensor": np.ones((16, 32, 1))},
                "spp": {"sensor": 128},
            }
            for i, _ in enumerate(quad.nodes)
        },
    }

    # Postprocessing succeeds and viewing angles have correct bounds
    ds = d.postprocess()
    assert "vza" in ds.coords
    assert np.allclose(ds.vza.min(), 5.06592926)  # Value calculated manually
    assert np.allclose(ds.vza.max(), 86.47273911)  # Value calculated manually
    assert "vaa" in ds.coords
    assert np.allclose(ds.vaa.min(), -177.09677419)  # Value calculated manually
    assert np.allclose(ds.vaa.max(), 177.09677419)  # Value calculated manually

    # Spectral dimension has the same length as the number of selected bins
    assert len(ds.w) == 1

    # We now move on to the plane case
    d._film_resolution = (32, 1)
    # Mismatched film size and raw data dimensions raises
    with pytest.raises(ValueError):
        d.postprocess()

    # Postprocessing succeeds and viewing angles have correct bounds
    d.results.raw = {
        **{
            ("550", i): {
                "values": {"sensor": np.ones((1, 32, 1))},
                "spp": {"sensor": 128},
            }
            for i, _ in enumerate(quad.nodes)
        },
    }
    ds = d.postprocess()
    assert "vza" in ds.coords
    assert np.allclose(ds.vza.min(), -87.1875)  # Value manually calculated
    assert np.allclose(ds.vza.max(), 87.1875)  # Value manually calculated
    assert "vaa" in ds.coords
    assert np.allclose(ds.vaa, 0.0)


def test_distant_flux_postprocessing_mono(modes_all_mono):
    # We use a peculiar rectangular film size to make sure that we get dimensions
    # right
    d = DistantFluxMeasure(film_resolution=(32, 16))

    # Add test data to results
    d.results.raw = {
        550.0: {"values": {"sensor": np.ones((16, 32, 1))}, "spp": {"sensor": 128}},
        560.0: {"values": {"sensor": np.ones((16, 32, 1))}, "spp": {"sensor": 128}},
    }

    # Postprocessing succeeds and flux field has the same size as the number of
    # spectral loop iterations
    ds = d.postprocess()
    assert "flux" in ds.data_vars
    assert ds["flux"].shape == (2,)
    assert np.allclose(ds.flux, 1.0 * 16 * 32)


def test_distant_flux_postprocessing_ckd(modes_all_ckd):
    # We use a peculiar rectangular film size to make sure that we get dimensions
    # right
    d = DistantFluxMeasure(
        film_resolution=(32, 16),
        spectral_cfg={"bin_set": "10nm_test", "bins": ["550", "560"]},
    )
    quad = d.spectral_cfg.bin_set.quad

    # Add test data to results
    d.results.raw = {
        **{
            ("550", i): {
                "values": {"sensor": np.ones((16, 32, 1))},
                "spp": {"sensor": 128},
            }
            for i, _ in enumerate(quad.nodes)
        },
        **{
            ("560", i): {
                "values": {"sensor": np.ones((16, 32, 1))},
                "spp": {"sensor": 128},
            }
            for i, _ in enumerate(quad.nodes)
        },
    }

    # Postprocessing succeeds and flux field has the same size as the number of
    # selected CKD bins
    ds = d.postprocess()
    print(ds)
    assert "flux" in ds.data_vars
    assert ds["flux"].shape == (2,)
    assert np.allclose(ds.flux, 1.0 * 16 * 32)
