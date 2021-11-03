import pathlib
import tempfile

import numpy as np
import pinttr
import pytest

from eradiate import path_resolver, unit_context_config
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.kernel.gridvolume import read_binary_grid3d, write_binary_grid3d
from eradiate.radprops import US76ApproxRadProfile
from eradiate.radprops.rad_profile import AFGL1986RadProfile, ArrayRadProfile
from eradiate.scenes.atmosphere._heterogeneous_legacy import (
    HeterogeneousAtmosphereLegacy,
)
from eradiate.scenes.core import KernelDict


def test_read_binary_grid3d():
    # write a volume data binary file and test that we read what we wrote
    write_values = np.random.random((1, 1, 10))
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_filename = pathlib.Path(tmp_dir, "test.vol")
    write_binary_grid3d(filename=tmp_filename, values=write_values)
    read_values = read_binary_grid3d(tmp_filename)
    assert np.allclose(write_values, read_values)


def test_heterogeneous_write_volume_data_files(mode_mono, tmpdir):
    """
    Writes the volume data files and produces a kernel dictionary that can be
    loaded by the kernel.
    """
    ctx = KernelDictContext()
    with unit_context_config.override({"length": "km"}):
        atmosphere = HeterogeneousAtmosphereLegacy(
            width=100.0,
            profile={
                "type": "array",
                "levels": np.linspace(0, 9, 10),
                "sigma_t_values": np.random.random(9),
                "albedo_values": np.ones(9),
            },
            cache_dir=tmpdir,
        )

    assert KernelDict.from_elements(atmosphere, ctx=ctx).load() is not None


def test_heterogeneous_ckd(mode_ckd, tmpdir):
    """
    Writes the volume data files and produces a kernel dictionary that can be
    loaded by the kernel.
    """
    ctx = KernelDictContext()
    with unit_context_config.override({"length": "km"}):
        atmosphere = HeterogeneousAtmosphereLegacy(
            width=100.0,
            profile={
                "type": "afgl_1986",
            },
            cache_dir=tmpdir,
        )

    assert KernelDict.from_elements(atmosphere, ctx=ctx).load() is not None


def test_heterogeneous_default(mode_mono):
    """
    Assigns default values.
    """
    atmosphere = HeterogeneousAtmosphereLegacy()
    assert isinstance(atmosphere.profile, US76ApproxRadProfile)
    assert atmosphere.albedo_filename == "albedo.vol"
    assert atmosphere.sigma_t_filename == "sigma_t.vol"
    assert atmosphere.cache_dir.is_dir()


def test_heterogeneous_get_bottom():
    """Returns 0.0 km."""
    assert HeterogeneousAtmosphereLegacy().bottom == ureg.Quantity(0.0, "km")


@pytest.fixture
def test_absorption_data_set():
    return path_resolver.resolve(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )


def test_heterogeneous_get_top_us76(mode_mono, tmpdir, test_absorption_data_set):
    """
    Sets the atmosphere top to the maximum altitude level value in the
    underlying US76ApproxRadProfile.
    """
    atmosphere = HeterogeneousAtmosphereLegacy(
        profile=US76ApproxRadProfile(
            absorption_data_set=test_absorption_data_set,
            thermoprops=dict(levels=ureg.Quantity(np.linspace(0, 86, 87), "km")),
        ),
        cache_dir=tmpdir,
    )
    assert atmosphere.top == ureg.Quantity(86, "km")


@pytest.fixture
def test_absorption_data_sets():
    return {
        "CH4": path_resolver.resolve(
            "tests/spectra/absorption/CH4-spectra-4000_11502.nc"
        ),
        "CO2": path_resolver.resolve(
            "tests/spectra/absorption/CO2-spectra-4000_14076.nc"
        ),
        "CO": path_resolver.resolve(
            "tests/spectra/absorption/CO-spectra-4000_14478.nc"
        ),
        "H2O": path_resolver.resolve(
            "tests/spectra/absorption/H2O-spectra-4000_25711.nc"
        ),
        "N2O": path_resolver.resolve(
            "tests/spectra/absorption/N2O-spectra-4000_10364.nc"
        ),
        "O2": path_resolver.resolve(
            "tests/spectra/absorption/O2-spectra-4000_17273.nc"
        ),
        "O3": path_resolver.resolve("tests/spectra/absorption/O3-spectra-4000_6997.nc"),
    }


def test_heterogeneous_get_top_afgl_1986(mode_mono, tmpdir, test_absorption_data_sets):
    """
    Sets the atmosphere top to the maximum altitude level value in the
    underlying AFGL1986RadProfile.
    """
    a = HeterogeneousAtmosphereLegacy(
        profile=AFGL1986RadProfile(
            absorption_data_sets=test_absorption_data_sets,
            thermoprops=dict(
                model_id="us_standard",
                levels=ureg.Quantity(np.linspace(0, 100, 101), "km"),
            ),
        )
    )
    assert a.top == ureg.Quantity(100.0, "km")


def test_heterogeneous_invalid_width_units(mode_mono):
    """
    Raises when the width units are invalid.
    """
    with pytest.raises(pinttr.exceptions.UnitsError):
        HeterogeneousAtmosphereLegacy(
            width=ureg.Quantity(100.0, "m^2"),
            profile=ArrayRadProfile(
                levels=np.linspace(0, 9, 10),
                sigma_t_values=np.random.random(9),
                albedo_values=np.random.random(9),
            ),
        )


def test_heterogeneous_invalid_width_value(mode_mono, tmpdir):
    """
    Raises when width value is negative.
    """
    with pytest.raises(ValueError):
        HeterogeneousAtmosphereLegacy(
            width=-100.0,
            profile=ArrayRadProfile(
                levels=np.linspace(0, 9, 10),
                sigma_t_values=np.random.random(9),
                albedo_values=np.random.random(9),
            ),
            cache_dir=tmpdir,
        )
