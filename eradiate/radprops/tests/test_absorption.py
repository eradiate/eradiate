import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.radprops.absorption import compute_sigma_a


def test_compute_sigma_a_no_t_coord():
    # data set with no t coordinate
    ds = xr.Dataset(
        data_vars={"xs": (("w", "p"), np.random.random((4, 3)), dict(units="cm^2"))},
        coords={
            "w": ("w", np.linspace(18000, 18010, 4), dict(units="cm^-1")),
            "p": ("p", np.linspace(90000, 110000, 3), dict(units="Pa")),
        },
    )

    # returns a single absorption coefficient value when input pressure is
    # a scalar
    x = compute_sigma_a(ds, wl=555.5)
    assert isinstance(x, ureg.Quantity)
    assert x.check("[length]^-1")
    assert len(x) == 1

    # does not raise when 'fill_value' is used
    x = compute_sigma_a(ds, wl=555.5, p=120000.0, fill_values=dict(pt=0.0))

    # raises when wavelength out of range
    with pytest.raises(ValueError):
        compute_sigma_a(ds, wl=560.0)

    # raises when pressure out of range
    with pytest.raises(ValueError):
        compute_sigma_a(ds, wl=555.5, p=120000.0)

    # returns an array of absorption coefficient values when input pressure is
    # an array
    x = compute_sigma_a(
        ds=ds, wl=555.5, p=np.linspace(ds.p.values.min(), ds.p.values.max(), num=10)
    )
    assert isinstance(x, ureg.Quantity)
    assert x.check("[length]^-1")
    assert len(x) == 10

    # absorption coefficient scales with number density
    x = compute_sigma_a(ds, wl=555.5, n=np.array([1.0, 2.0]))
    assert x[1] == 2 * x[0]

    # dataset with no temperature coordinate is not interpolated on temperature
    x = compute_sigma_a(ds, wl=555.5, t=200.0, n=1.0)
    y = compute_sigma_a(ds, wl=555.5, t=300.0, n=1.0)
    assert x == y


def test_compute_sigma_a_t_coord():
    # data set with t coordinate
    ds = xr.Dataset(
        data_vars={
            "xs": (("w", "p", "t"), np.random.random((4, 3, 2)), dict(units="cm^2"))
        },
        coords={
            "w": ("w", np.linspace(18000, 18010, 4), dict(units="cm^-1")),
            "p": ("p", np.linspace(90000, 110000, 3), dict(units="Pa")),
            "t": ("t", np.array([250, 300]), dict(units="K")),
        },
    )

    # returns a single absorption coefficient value when input pressure
    # and temperature are scalars
    x = compute_sigma_a(ds, wl=555.5)
    assert isinstance(x, ureg.Quantity)
    assert x.check("[length]^-1")
    assert len(x) == 1

    # does not raise when 'fill_value' is used
    x = compute_sigma_a(ds, wl=555.5, p=120000.0, fill_values=dict(pt=0.0))

    # raises when wavelength out of range
    with pytest.raises(ValueError):
        compute_sigma_a(ds, wl=560.0)

    # raises when pressure out of range
    with pytest.raises(ValueError):
        compute_sigma_a(ds, wl=555.5, p=120000.0)

    # returns an array of absorption coefficient values when input pressure is
    # an array
    x = compute_sigma_a(
        ds=ds, wl=555.5, p=np.linspace(ds.p.values.min(), ds.p.values.max(), num=10)
    )
    assert isinstance(x, ureg.Quantity)
    assert x.check("[length]^-1")
    assert len(x) == 10
    assert not any(np.isnan(x.magnitude))

    # returns a 1D array of absorption coefficient values when input pressure
    # and temperature are arrays
    x = compute_sigma_a(
        ds=ds,
        wl=555.5,
        p=np.linspace(ds.p.values.min(), ds.p.values.max(), num=12),
        t=np.linspace(ds.t.values.min(), ds.t.values.max(), num=12),
    )
    assert isinstance(x, ureg.Quantity)
    assert x.check("[length]^-1")
    assert len(x) == 12
    assert not any(np.isnan(x.magnitude))

    # raises when input pressure and temperature are arrays with different
    # lengths
    with pytest.raises(ValueError):
        x = compute_sigma_a(
            ds=ds,
            wl=555.5,
            p=np.linspace(ds.p.values.min(), ds.p.values.max(), num=12),
            t=np.linspace(ds.t.values.min(), ds.t.values.max(), num=11),
        )

    # absorption coefficient scales with number density
    x = compute_sigma_a(ds, wl=555.5, n=np.array([1.0, 2.0]))
    assert x[1] == 2 * x[0]

    # dataset with temperature coordinate is interpolated on temperature
    x = compute_sigma_a(ds, wl=555.5, t=260.0, n=1.0)
    y = compute_sigma_a(ds, wl=555.5, t=280.0, n=1.0)
    assert x != y
