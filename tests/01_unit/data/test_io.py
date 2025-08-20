import pprint

import pytest

from eradiate.data._validation import DatasetValidator
from eradiate.data.io import load_aerosol_libradtran


@pytest.mark.parametrize(
    "fname, kwargs, loading_exception",
    [
        ("tests/libradtran_samples/iprt_waso.mie.cdf", {}, None),
        ("tests/libradtran_samples/waso.mie.cdf", {"hum": 50.0}, None),
        ("tests/libradtran_samples/waso.mie.cdf", {}, TypeError),
        ("tests/libradtran_samples/soot.mie.cdf", {"hum": 0.0}, None),
        ("tests/libradtran_samples/soot.mie.cdf", {}, None),
    ],
    ids=["iprt_waso", "waso", "waso_nohum", "soot", "soot_nohum"],
)
def test_load_aerosol_libradtran(mode_mono, fname, kwargs, loading_exception):
    """
    Test libRadtran aerosol converter with a monochromatic
    """
    if loading_exception is None:
        ds = load_aerosol_libradtran(fname, **kwargs)
        # Check that the produced dataset validates against the aerosol format schema
        v = DatasetValidator()
        v.validate(ds, schema="particle_dataset_v1")
        assert not v.errors, f"Dataset validation errors\n{pprint.pformat(v.errors)}"

    else:
        with pytest.raises(loading_exception):
            load_aerosol_libradtran(fname, **kwargs)
