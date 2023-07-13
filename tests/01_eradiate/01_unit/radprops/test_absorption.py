import joseki
import numpy as np
import pytest

from eradiate import data
from eradiate import unit_registry as ureg
from eradiate.radprops.absorption import (
    eval_sigma_a_ckd_impl,
    eval_sigma_a_mono_impl,
    wrange_ckd,
    wrange_mono,
)
from eradiate.units import to_quantity


@pytest.mark.parametrize(
    "w",
    [
        np.array([550.0]) * ureg.nm,
        np.linspace(540.0, 560.0) * ureg.nm,
    ],
)
def test_eval_sigma_a_mono_impl(w, error_handler_config):
    """
    The shape of the absorption coefficient array is consistent with the
    wavelength (w) and altitude (z) arrays.
    """
    z = np.linspace(0.0, 10.0, 11) * ureg.km
    thermoprops = joseki.make(
        identifier="afgl_1986-us_standard",
        z=z,
        additional_molecules=False,
    )
    ds = data.load_dataset("spectra/absorption/mono/komodo/komodo.nc")
    sigma_a = eval_sigma_a_mono_impl(
        absorption_data={wrange_mono(ds): ds},
        thermoprops=thermoprops,
        w=w,
        error_handler_config=error_handler_config,
    )

    # sigma_a should have a shape of (w, z)
    assert sigma_a.shape == (w.size, z.size)


def test_eval_sigma_a_ckd_impl(error_handler_config):
    """
    The shape of the absorption coefficient array is consistent with the
    wavelength (w) and altitude (z) arrays.
    """
    z = np.linspace(0.0, 10.0, 11) * ureg.km
    thermoprops = joseki.make(
        identifier="afgl_1986-us_standard",
        z=z,
        additional_molecules=False,
    )
    ds = data.load_dataset("spectra/absorption/ckd/monotropa/monotropa-18100_18200.nc")
    wcenter = to_quantity(ds.w)
    sigma_a = eval_sigma_a_ckd_impl(
        absorption_data={wrange_ckd(ds): ds},
        thermoprops=thermoprops,
        w=wcenter,
        g=0.5,
        error_handler_config=error_handler_config,
    )

    # sigma_a should have a shape of (w, z)
    assert sigma_a.shape == (1, z.size)
