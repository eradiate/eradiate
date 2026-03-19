import pytest

from eradiate.data.io import load_aerosol_libradtran


@pytest.mark.parametrize(
    "fname, kwargs, loading_exception",
    [
        ("tests/libradtran_samples/iprt_waso.mie.cdf", {}, None),
        ("tests/libradtran_samples/waso.mie.cdf", {"hum": 50.0}, None),
        ("tests/libradtran_samples/waso.mie.cdf", {}, TypeError),
        ("tests/libradtran_samples/soot.mie.cdf", {"hum": 0.0}, None),
        ("tests/libradtran_samples/soot.mie.cdf", {}, None),
        (
            "tests/libradtran_samples/mopsmap.cdf",
            {
                "fallback_units": {
                    "reff": "percent",
                    "wavelen": "um",
                    "ext": "m ** 3 / g / km",
                    "ssa": "",
                }
            },
            None,
        ),
        ("tests/libradtran_samples/mopsmap.cdf", {}, ValueError),
    ],
    ids=[
        "iprt_waso",
        "waso",
        "waso_nohum",
        "soot",
        "soot_nohum",
        "mopsmap",
        "mopsmap_nounits",
    ],
)
def test_load_aerosol_libradtran(mode_mono, fname, kwargs, loading_exception):
    """
    Test libRadtran aerosol file loading (automatic conversion) with various
    samples.
    """
    if loading_exception is None:
        assert load_aerosol_libradtran(fname, **kwargs)
        # TODO: Add dataset validation once Pandera integration is here

    else:
        with pytest.raises(loading_exception):
            load_aerosol_libradtran(fname, **kwargs)
