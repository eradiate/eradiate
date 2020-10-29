from eradiate.data.absorption_spectra import _AbsorptionGetter


def test_absorption_getter():
    getter = _AbsorptionGetter
    ds = getter.open(id="test")

    # check that the multi-files dataset has the pressure dimension and
    # coordinate right
    assert "p" in ds.dims
    assert "p" in ds.coords
    assert len(ds.p) == 64

    # check that we can interpolate alont the wavenumber axis
    ds.xs.interp(w=15852.2)

    # check that we can interpolate along the pressure axis
    ds.xs.interp(p=1e5)

    # check that we can interpolate succesively along the two axes
    ds.xs.interp(w=15852.2).interp(p=1e5)
