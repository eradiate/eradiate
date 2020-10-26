import numpy as np
import pytest

from eradiate.data import _presolver
from eradiate.scenes.atmosphere.heterogeneous import HeterogeneousAtmosphere
from eradiate.scenes.core import KernelDict
from eradiate.util.collections import onedict_value
from eradiate.util.units import config_default_units, kernel_default_units, ureg


def test_heterogeneous_nowrite(mode_mono):
    from eradiate.kernel.core.xml import load_dict

    # Test default constructor
    a = HeterogeneousAtmosphere(
        width=ureg.Quantity(100., ureg.km),
        height=ureg.Quantity(100., ureg.km),
        sigma_t_fname=_presolver.resolve(
            "tests/textures/heterogeneous_atmosphere_mono/sigma_t.vol"
        ),
        albedo_fname=_presolver.resolve(
            "tests/textures/heterogeneous_atmosphere_mono/albedo.vol"
        )
    )

    # Check if default output can be loaded
    p = a.phase()
    assert load_dict(onedict_value(p)) is not None

    m = a.media()
    assert load_dict(onedict_value(m)) is not None

    s = a.shapes()
    assert load_dict(onedict_value(s)) is not None

    # Load all elements at once (and use references)
    with kernel_default_units.override({"length": "km"}):
        kernel_dict = KernelDict.empty()
        kernel_dict.add(a)
        scene = kernel_dict.load()
        assert scene is not None


def test_heterogeneous_write(mode_mono, tmpdir):
    # Check if volume data file creation works as expected
    with config_default_units.override({"length": "km"}):
        a = HeterogeneousAtmosphere(
            width=100.,
            height=1000.,
            profile={
                "type": "array",
                "sigma_t_values": np.ones((3, 3, 3)),
                "albedo_values": np.ones((3, 3, 3)),
            },
            cache_dir=tmpdir
        )

    a.kernel_dict()
    # If file creation is successful, volume data files must exist
    assert a.albedo_fname.is_file()
    assert a.sigma_t_fname.is_file()
    # Check if written files can be loaded
    assert KernelDict.empty().add(a).load() is not None

    # Check that inconsistent init will raise
    with pytest.raises(ValueError):
        a = HeterogeneousAtmosphere()

    with pytest.raises(FileNotFoundError):
        a = HeterogeneousAtmosphere(
            profile=None,
            albedo_fname=tmpdir / "doesnt_exist.vol",
            sigma_t_fname=tmpdir / "doesnt_exist.vol",
        )
