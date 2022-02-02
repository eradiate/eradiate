import os

import pytest

import eradiate

eradiate.plot.set_style()

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
            """\033[93mRunning the full test suite. To skip slow tests, please run 'pytest -m "not slow"' \033[0m"""
        )

    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )

    config.addinivalue_line(
        "markers", "regression: marks tests as potentially very slow regression tests"
    )


@pytest.fixture(scope="session")
def session_timestamp():
    from datetime import datetime

    return datetime.now()
