import numpy as np
import pytest

import eradiate.data as data
from eradiate.radprops.absorption import compute_sigma_a
from eradiate import unit_registry as ureg


def test_compute_sigma_a():
    # returns a single-valued quantity with default p and t parameters
    ds = data.open(category="absorption_spectrum", id="test")
    x = compute_sigma_a(ds, wl=630.8)
    assert isinstance(x, ureg.Quantity)
    assert isinstance(x.magnitude, float)
    assert x.units == ureg.Unit("m^-1")

    # returns an array quantity when an array is passed to p
    x = compute_sigma_a(ds, wl=630.8, p=np.array([1., 2.]))
    assert isinstance(x.magnitude, np.ndarray)
    assert x.magnitude.size == 2

    # absorption coefficient scales with number density
    x = compute_sigma_a(ds, wl=630.8, n=np.array([1., 2.]))
    assert x[1] == 2 * x[0]

    # dataset with no temperature coordinate is not interpolated on temperature
    x = compute_sigma_a(ds, wl=630.8, t=200., n=1.)
    y = compute_sigma_a(ds, wl=630.8, t=300., n=1.)
    assert x == y

    # raises when outside of range
    with pytest.raises(ValueError):
        compute_sigma_a(ds, wl=600.)

    with pytest.raises(ValueError):
        compute_sigma_a(ds, wl=630.8, p=1e-7)

    # does not raise if p_fill_value is not None
    x = compute_sigma_a(ds, wl=630.8, p=1e-7, p_fill_value=0.)
