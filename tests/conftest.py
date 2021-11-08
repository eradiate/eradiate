import pytest

import eradiate
from eradiate import data

eradiate.plot.set_style()

# ------------------------------------------------------------------------------
#                           Kernel variant fixtures
# ------------------------------------------------------------------------------


def generate_fixture(variant):
    @pytest.fixture()
    def fixture():
        try:
            import mitsuba

            mitsuba.set_variant(variant)

        except Exception:
            pytest.skip(f"Mitsuba variant '{variant}' is not enabled!")

    globals()["variant_" + variant] = fixture


for variant in [
    "scalar_mono",
    "scalar_mono_double",
    "scalar_mono_polarized",
    "scalar_rgb",
    "scalar_spectral",
    "packet_rgb",
    "packet_spectral",
]:
    generate_fixture(variant)
del generate_fixture


# ------------------------------------------------------------------------------
#                                 Mode fixtures
# ------------------------------------------------------------------------------


def generate_fixture(mode):
    @pytest.fixture()
    def fixture():
        import eradiate

        eradiate.set_mode(mode)

    globals()["mode_" + mode] = fixture


for mode in eradiate.modes():
    generate_fixture(mode)
del generate_fixture


def generate_fixture_group(name, modes):
    @pytest.fixture(params=modes)
    def fixture(request):
        mode = request.param
        import eradiate

        eradiate.set_mode(mode)

    globals()["modes_" + name] = fixture


variant_groups = {
    "all_mono": [x for x in eradiate.modes() if x.startswith("mono")],
    "all_ckd": [x for x in eradiate.modes() if x.startswith("ckd")],
    "all_mono_ckd": [
        x for x in eradiate.modes() if (x.startswith("mono") or x.startswith("ckd"))
    ],
    "all_single": [x for x in eradiate.modes() if not x.endswith("double")],
    "all_double": [x for x in eradiate.modes() if x.endswith("double")],
    "all": list(eradiate.modes()),
}

for name, variants in variant_groups.items():
    generate_fixture_group(name, variants)
del generate_fixture_group


# ------------------------------------------------------------------------------
#                              Other configuration
# ------------------------------------------------------------------------------


def pytest_runtest_setup(item):
    for mark in item.iter_markers(name="skipif_data_not_found"):
        if "dataset_category" not in mark.kwargs:
            dataset_category = mark.args[0]
        else:
            dataset_category = mark.kwargs["dataset_category"]

        if "dataset_id" not in mark.kwargs:
            dataset_id = mark.args[1]
        else:
            dataset_id = mark.kwargs["dataset_id"]

        # Try to get path to dataset file
        dataset_path = data.getter(dataset_category).PATHS[dataset_id]

        # If the data is missing, we skip the test
        if not data.find(dataset_category)[dataset_id]:
            pytest.skip(
                f"Could not find dataset '{dataset_category}.{dataset_id}'; "
                f"please download dataset files and place them in "
                f"'data/{dataset_path}' directory."
            )


def pytest_configure(config):
    markexpr = config.getoption("markexpr", "False")

    if not "not slow" in markexpr:
        print(
            """\033[93mRunning the full test suite. To skip slow tests, please run "pytest -m 'not slow'"\033[0m"""
        )

    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )

    config.addinivalue_line(
        "markers",
        "skipif_data_not_found(dataset_category, dataset_id): "
        "skip the given test function if the referenced dataset is not found. "
        "Example: skipif_dataset_not_found('absorption_spectrum', "
        "'us76_approx') skips if the 'absorption_spectrum.us76_approx' files "
        "cannot be found in paths resolved by Eradiate.",
    )
