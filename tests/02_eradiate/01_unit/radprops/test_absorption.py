import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.radprops.absorption import compute_sigma_a
from eradiate.units import to_quantity


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
    wl = ureg.Quantity(555.5, "nm")
    x = compute_sigma_a(ds, wl=wl)
    assert isinstance(x, ureg.Quantity)
    assert x.check("[length]^-1")
    assert len(x) == 1

    # handles multiple wavelength values
    x = compute_sigma_a(ds, wl=ureg.Quantity(np.array([555.50, 555.51]), "nm"))
    assert len(x) == 2

    # does not raise when 'fill_value' is used
    x = compute_sigma_a(
        ds, wl=wl, p=ureg.Quantity(120000.0, "Pa"), fill_values=dict(pt=0.0)
    )

    # raises when wavelength out of range
    with pytest.raises(ValueError):
        compute_sigma_a(ds, wl=ureg.Quantity(560.0, "nm"))

    # raises when pressure out of range
    with pytest.raises(ValueError):
        compute_sigma_a(ds, wl=wl, p=ureg.Quantity(120000.0, "Pa"))

    # returns an array of absorption coefficient values when input pressure is
    # an array
    ds_p = to_quantity(ds.p)
    x = compute_sigma_a(ds=ds, wl=wl, p=np.linspace(ds_p.min(), ds_p.max(), num=10))
    assert isinstance(x, ureg.Quantity)
    assert x.check("[length]^-1")
    assert len(x) == 10

    # absorption coefficient scales with number density
    x = compute_sigma_a(ds, wl=wl, n=ureg.Quantity(np.array([1.0, 2.0]), "m^-3"))
    assert x[1] == 2 * x[0]

    # dataset with no temperature coordinate is not interpolated on temperature
    x = compute_sigma_a(
        ds, wl=wl, t=ureg.Quantity(200.0, "K"), n=ureg.Quantity(1.0, "m^-3")
    )
    y = compute_sigma_a(
        ds, wl=wl, t=ureg.Quantity(300.0, "K"), n=ureg.Quantity(1.0, "m^-3")
    )
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
    wl = ureg.Quantity(555.5, "nm")
    x = compute_sigma_a(ds, wl=wl)
    assert isinstance(x, ureg.Quantity)
    assert x.check("[length]^-1")
    assert len(x) == 1

    # handles multiple wavelength values
    x = compute_sigma_a(ds, wl=ureg.Quantity(np.array([555.50, 555.51]), "nm"))
    assert len(x) == 2

    # does not raise when 'fill_value' is used
    x = compute_sigma_a(
        ds, wl=wl, p=ureg.Quantity(120000.0, "Pa"), fill_values=dict(pt=0.0)
    )

    # raises when wavelength out of range
    with pytest.raises(ValueError):
        compute_sigma_a(ds, wl=ureg.Quantity(560.0, "nm"))

    # raises when pressure out of range
    with pytest.raises(ValueError):
        compute_sigma_a(ds, wl=wl, p=ureg.Quantity(120000.0, "Pa"))

    # returns an array of absorption coefficient values when input pressure is
    # an array
    ds_p = to_quantity(ds.p)
    x = compute_sigma_a(ds=ds, wl=wl, p=np.linspace(ds_p.min(), ds_p.max(), num=10))
    assert isinstance(x, ureg.Quantity)
    assert x.check("[length]^-1")
    assert len(x) == 10
    assert not any(np.isnan(x.magnitude))

    # returns a 1D array of absorption coefficient values when input pressure
    # and temperature are arrays
    ds_t = to_quantity(ds.t)
    x = compute_sigma_a(
        ds=ds,
        wl=wl,
        p=np.linspace(ds_p.min(), ds_p.max(), num=12),
        t=np.linspace(ds_t.min(), ds_t.max(), num=12),
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
            wl=wl,
            p=np.linspace(ds_p.min(), ds_p.max(), num=12),
            t=np.linspace(ds_t.min(), ds_t.max(), num=11),
        )

    # absorption coefficient scales with number density
    x = compute_sigma_a(ds, wl=wl, n=ureg.Quantity(np.array([1.0, 2.0]), "m^-3"))
    assert x[1] == 2 * x[0]

    # dataset with temperature coordinate is interpolated on temperature
    x = compute_sigma_a(
        ds, wl=wl, t=ureg.Quantity(260.0, "K"), n=ureg.Quantity(1.0, "m^-3")
    )
    y = compute_sigma_a(
        ds, wl=wl, t=ureg.Quantity(280.0, "K"), n=ureg.Quantity(1.0, "m^-3")
    )
    assert x != y
