import pathlib
import tempfile

import numpy as np
import pytest

from eradiate.contexts import SpectralContext
from eradiate.kernel.gridvolume import read_binary_grid3d, write_binary_grid3d
from eradiate.radprops.rad_profile import US76ApproxRadProfile


@pytest.fixture
def test_radprops(mode_mono):
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    return US76ApproxRadProfile().to_dataset(spectral_ctx=spectral_ctx)


def test_write_read_binary_grid3d():
    """Reads what was written."""
    write_values = np.random.random(10).reshape(1, 1, 10)
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_filename = pathlib.Path(tmp_dir, "test.vol")
    write_binary_grid3d(filename=tmp_filename, values=write_values)
    read_values = read_binary_grid3d(tmp_filename)
    assert np.allclose(write_values, read_values)
