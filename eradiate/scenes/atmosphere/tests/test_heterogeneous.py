import pathlib
import tempfile

import numpy as np
import pinttr
import pytest

from eradiate import path_resolver, unit_context_config, unit_context_kernel
from eradiate import unit_registry as ureg
from eradiate._util import onedict_value
from eradiate.data import _presolver
from eradiate.radprops import AFGL1986RadProfile, US76ApproxRadProfile
from eradiate.scenes.atmosphere._heterogeneous import (
    HeterogeneousAtmosphere,
    read_binary_grid3d,
    write_binary_grid3d,
)
from eradiate.scenes.core import KernelDict


def test_read_binary_grid3d():
    # write a volume data binary file and test that we read what we wrote
    write_values = np.random.random(10).reshape(1, 1, 10)
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_filename = pathlib.Path(tmp_dir, "test.vol")
    write_binary_grid3d(filename=tmp_filename, values=write_values)
    read_values = read_binary_grid3d(tmp_filename)
    assert np.allclose(write_values, read_values)


def test_heterogeneous_nowrite(mode_mono):
    from mitsuba.core.xml import load_dict

    # Test default constructor
    a = HeterogeneousAtmosphere(
        width=ureg.Quantity(100.0, ureg.km),
        toa_altitude=ureg.Quantity(100.0, ureg.km),
        sigma_t_fname=_presolver.resolve(
            "tests/textures/heterogeneous_atmosphere_mono/sigma_t.vol"
        ),
        albedo_fname=_presolver.resolve(
            "tests/textures/heterogeneous_atmosphere_mono/albedo.vol"
        ),
    )

    # Check if default output can be loaded
    p = a.phase()
    assert load_dict(onedict_value(p)) is not None

    m = a.media()
    assert load_dict(onedict_value(m)) is not None

    s = a.shapes()
    assert load_dict(onedict_value(s)) is not None

    # Load all elements at once (and use references)
    with unit_context_kernel.override({"length": "km"}):
        kernel_dict = KernelDict.new()
        kernel_dict.add(a)
        scene = kernel_dict.load()
        assert scene is not None


def test_heterogeneous_write(mode_mono, tmpdir):
    # Check if volume data file creation works as expected
    with unit_context_config.override({"length": "km"}):
        a = HeterogeneousAtmosphere(
            width=100.0,
            profile={
                "type": "array",
                "levels": np.linspace(0, 3, 4),
                "sigma_t_values": np.ones((3, 3, 3)),
                "albedo_values": np.ones((3, 3, 3)),
            },
            cache_dir=tmpdir,
        )

    a.kernel_dict()
    # If file creation is successful, volume data files must exist
    assert a.albedo_fname.is_file()
    assert a.sigma_t_fname.is_file()
    # Check if written files can be loaded
    assert KernelDict.new(a).load() is not None

    # Check that inconsistent init will raise
    with pytest.raises(ValueError):
        a = HeterogeneousAtmosphere()

    with pytest.raises(FileNotFoundError):
        a = HeterogeneousAtmosphere(
            profile=None,
            albedo_fname=tmpdir / "doesnt_exist.vol",
            sigma_t_fname=tmpdir / "doesnt_exist.vol",
        )


def test_heterogeneous_us76(mode_mono, tmpdir):
    # Volume data file creation works as expected
    # Underlying radiative properties profile is correct
    # Atmosphere height is correctly derived from the radiative
    # properties profile
    test_absorption_data_set = path_resolver.resolve(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )
    a = HeterogeneousAtmosphere(
        width=ureg.Quantity(1000.0, "km"),
        profile={
            "type": "us76_approx",
            "levels": ureg.Quantity(np.linspace(0, 86, 87), "km"),
            "absorption_data_set": test_absorption_data_set,
        },
        cache_dir=tmpdir,
    )
    assert a.height == ureg.Quantity(86, "km")
    profile = a.profile
    assert isinstance(profile, US76ApproxRadProfile)
    assert profile.sigma_a.shape == (1, 1, 86)
    assert profile.sigma_s.shape == (1, 1, 86)
    assert profile.sigma_t.shape == (1, 1, 86)
    assert profile.albedo.shape == (1, 1, 86)

    a.kernel_dict()
    # If file creation is successful, volume data files must exist
    assert a.albedo_fname.is_file()
    assert a.sigma_t_fname.is_file()
    # Check if written files can be loaded
    assert KernelDict.new(a).load() is not None


def test_heterogeneous_afgl1986(mode_mono, tmpdir):
    # AFGL 1986 - US Standard atmosphere with custom level altitudes and
    # absorbing species concentrations
    test_absorption_data_sets = {
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
    n_layers = 100
    a = HeterogeneousAtmosphere(
        profile={
            "type": "afgl1986",
            "model": "us_standard",
            "levels": ureg.Quantity(np.linspace(0, 100, n_layers + 1), "km"),
            "concentrations": {
                "H2O": ureg.Quantity(5e23, "m^-2"),
                "O3": ureg.Quantity(0.5, "dobson_units"),
                "CO2": ureg.Quantity(400e-6, ""),
            },
            "absorption_data_sets": test_absorption_data_sets,
        }
    )
    assert a.height == ureg.Quantity(100, "km")
    assert isinstance(a.profile, AFGL1986RadProfile)


def test_heterogeneous_units(mode_mono):
    # Initialising a heterogeneous atmosphere with the wrong units raises an error
    with pytest.raises(pinttr.exceptions.UnitsError):
        a = HeterogeneousAtmosphere(
            width=ureg.Quantity(100.0, "m^2"),
            toa_altitude=1000.0,
            profile={
                "type": "array",
                "levels": np.linspace(0, 3, 4),
                "sigma_t_values": np.ones((3, 3, 3)),
                "albedo_values": np.ones((3, 3, 3)),
            },
        )

    with pytest.raises(pinttr.exceptions.UnitsError):
        a = HeterogeneousAtmosphere(
            width=100.0,
            toa_altitude=ureg.Quantity(1000.0, "s"),
            profile={
                "type": "array",
                "levels": np.linspace(0, 3, 4),
                "sigma_t_values": np.ones((3, 3, 3)),
                "albedo_values": np.ones((3, 3, 3)),
            },
        )


def test_heterogeneous_values(mode_mono, tmpdir):
    # test that initialising a heterogeneous atmosphere with invalid values raises an error
    with pytest.raises(ValueError):
        a = HeterogeneousAtmosphere(
            width=-100.0,
            toa_altitude=1000.0,
            profile={
                "type": "array",
                "levels": np.linspace(0, 3, 4),
                "sigma_t_values": np.ones((3, 3, 3)),
                "albedo_values": np.ones((3, 3, 3)),
            },
            cache_dir=tmpdir,
        )

    with pytest.raises(ValueError):
        a = HeterogeneousAtmosphere(
            width=100.0,
            toa_altitude=-1000.0,
            profile={
                "type": "array",
                "levels": np.linspace(0, 3, 4),
                "sigma_t_values": np.ones((3, 3, 3)),
                "albedo_values": np.ones((3, 3, 3)),
            },
            cache_dir=tmpdir,
        )
