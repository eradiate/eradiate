import os

import pytest

import eradiate

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
    "llvm_rgb",
    "llvm_spectral",
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
#               Customizable output dir for test artifacts
# ------------------------------------------------------------------------------


def pytest_addoption(parser):
    eradiate_dir = os.environ.get("ERADIATE_DIR", ".")
    parser.addoption(
        "--artefact-dir",
        action="store",
        default=os.path.join(eradiate_dir, "build/test_artefacts/"),
    )


# See: https://stackoverflow.com/a/55301318/3645374
@pytest.fixture(scope="session")
def artefact_dir(pytestconfig):
    option_value = pytestconfig.getoption("artefact_dir")

    if not os.path.isdir(option_value):
        os.makedirs(option_value)

    return option_value


# ------------------------------------------------------------------------------
#                              Other configuration
# ------------------------------------------------------------------------------


def pytest_configure(config):
    markexpr = config.getoption("markexpr", "False")

    if not "not slow" in markexpr:
        print(
            """\033[93mRunning the full test suite. To skip slow tests, please run "pytest -m 'not slow'"\033[0m"""
        )

    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )
