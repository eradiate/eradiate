import importlib

import pytest


def test_top_level_import():
    # Check if Mitsuba can be imported and aliased
    import mitsuba
    import eradiate.kernel
    assert eradiate.kernel is mitsuba

    # We check that C++ functions are available
    eradiate.kernel.set_variant("scalar_mono")
    from eradiate.kernel.core.xml import load_string as load_string_ert
    from mitsuba.core.xml import load_string as load_string_mts
    assert load_string_ert is load_string_mts

    # We check that a few top-level functions are available
    eradiate.kernel.variants()
    eradiate.kernel.config.PYTHON_EXECUTABLE


def test_variants():
    import eradiate.kernel

    # We check if we can use all compiled variants
    for v in eradiate.kernel.variants():
        eradiate.kernel.set_variant(v)


@pytest.mark.parametrize("submod", ["python", "python.autodiff", "python.test"])
def test_python_module_imports(submod):
    # We ensure that Mitsuba modules are correctly loaded
    mts_mod = importlib.import_module("mitsuba." + submod)
    ert_mod = importlib.import_module("eradiate.kernel." + submod)
    assert ert_mod.__file__ == mts_mod.__file__

    # TODO: At the moment, the modules are not properly aliased and retain their 
    # own name; this should be addressed in the future
    # assert ert_mod is mts_mod


def test_python_classes():
    # We ensure that Mitsuba Python classes (and functions) are correctly aliased
    from mitsuba.python.chi2 import ChiSquareTest as MitsubaChiSquareTest
    from eradiate.kernel.python.chi2 import ChiSquareTest as EradiateChiSquareTest
    # TODO: this currently fails
    # assert MitsubaChiSquareTest is EradiateChiSquareTest
