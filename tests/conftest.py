import logging
import os

import matplotlib as mpl
import pytest

import eradiate

mpl.use("Agg")
eradiate.plot.set_style()
# Required for HTML rendering in Robot
mpl.rcParams["legend.framealpha"] = 0.15


# ------------------------------------------------------------------------------
#                      Robot Configuration and fixtures
# ------------------------------------------------------------------------------


# Load necessary robot plugin for pytest interoperability
pytest_plugins = ("robotframework",)


# ------------------------------------------------------------------------------
#               Customizable output dir for test artifacts
# ------------------------------------------------------------------------------


def pytest_addoption(parser):
    eradiate_source_dir = os.environ.get("ERADIATE_SOURCE_DIR", ".")
    parser.addoption(
        "--artefact-dir",
        action="store",
        default=os.path.join(eradiate_source_dir, "test_artefacts/"),
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
    has_slow = "not slow" not in markexpr
    has_regression = "not regression" not in markexpr

    if has_slow:
        print(
            "\033[93m"
            "Running slow tests. To skip them, please run "
            "'pytest -m \"not slow\"' "
            "\033[0m"
        )

    if has_regression:
        print(
            "\033[93m"
            "Running regression tests. To skip them, please run "
            "'pytest -m \"not regression\"' "
            "\033[0m"
        )

    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )

    config.addinivalue_line(
        "markers",
        "regression: marks tests as potentially very slow regression tests "
        "(deselect with -m 'not regression')",
    )

    # Disable Eradiate and Mitsuba's progressbars
    eradiate.config.settings.progress = 0

    # Mitsuba's logger is configured to be silent, unless Pytest's configuration
    # explicitly requires an increased verbosity (e.g, passing the -v option to
    # its CLI)
    eradiate.kernel.install_logging()
    pytest_verbosity = config.get_verbosity()
    logging.getLogger("mitsuba").setLevel(logging.WARNING)
    logging.getLogger("eradiate").setLevel(logging.INFO)
    if pytest_verbosity >= 2:
        logging.getLogger("mitsuba").setLevel(logging.INFO)
        logging.getLogger("eradiate").setLevel(logging.DEBUG)

    # Silent Joseki
    logging.getLogger("joseki").setLevel(logging.WARNING)


# ------------------------------------------------------------------------------
#                            Pre-process helpers
# ------------------------------------------------------------------------------

# This improves assert handling when using the check_scene_element() function
pytest.register_assert_rewrite("eradiate.test_tools.types.check_scene_element")


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


variants = [x for x in eradiate.modes() if x not in {"mono", "ckd"}]  # Remove aliases
variant_groups = {
    "all_mono": [x for x in variants if x.startswith("mono")],
    "all_ckd": [x for x in variants if x.startswith("ckd")],
    "all_mono_ckd": [
        x for x in variants if (x.startswith("mono") or x.startswith("ckd"))
    ],
    "all_single": [x for x in variants if "single" in x],
    "all_double": [x for x in variants if "double" in x],
    "all_polarized": [x for x in variants if "polarized" in x],
    "all": variants,
}

for name, variants in variant_groups.items():
    generate_fixture_group(name, variants)
del generate_fixture_group

# ------------------------------------------------------------------------------
#                                 Other fixtures
# ------------------------------------------------------------------------------

from eradiate.test_tools.fixtures import *  # noqa: E402, F401, F403
