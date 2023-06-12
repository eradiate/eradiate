from pathlib import Path

import pytest
from environ.exceptions import ConfigError, MissingEnvValueError
from importlib.metadata import PackageNotFoundError, version

from eradiate._config import EradiateConfig, ProgressLevel
from eradiate.exceptions import ConfigWarning
from environ.exceptions import ConfigError


def eradiate_mitsuba_status(expect_installed=False):

    """
    Check if eradiate-mitsuba is installed in order to skip tests depending
    on the setup type (production or development).
    """

    try:
        version("eradiate-mitsuba")
        return expect_installed
    except PackageNotFoundError:
        return not expect_installed


@pytest.mark.skipif(eradiate_mitsuba_status(True), reason='This is a production environment: eradiate-mitsuba is installed')
def test_development_config(tmpdir):
    # Create an appropriate file tree for tests
    tmpdir_path = Path(tmpdir)
    (tmpdir_path / "eradiate/src/eradiate").mkdir(parents=True)
    (tmpdir_path / "eradiate/src/eradiate/__init__.py").touch()

    # Raises if ERADIATE_SOURCE_DIR and eradiate_mitsuba are missing
    with pytest.raises(ConfigError):
        EradiateConfig.from_environ({})

    # When ERADIATE_SOURCE_DIR points to a correct path
    # (i.e. contains eradiate.__init__.py), config init succeeds
    assert EradiateConfig.from_environ(
        {"ERADIATE_SOURCE_DIR": tmpdir_path / "eradiate"}
    )

    # Otherwise it raises
    with pytest.raises(ConfigError):
        EradiateConfig.from_environ({"ERADIATE_SOURCE_DIR": tmpdir_path})



@pytest.mark.skipif(eradiate_mitsuba_status(), reason='This is not a production environment: eradiate-mitsuba is not installed')
def test_production_config(tmpdir):
    # Create an appropriate file tree for tests
    tmpdir_path = Path(tmpdir)
    (tmpdir_path / "eradiate/src/eradiate").mkdir(parents=True)
    (tmpdir_path / "eradiate/src/eradiate/__init__.py").touch()


    # Should not fail, should detect eradiate-mitsuba instead
    config = EradiateConfig.from_environ({})
    assert config.source_dir is None

    # When ERADIATE_SOURCE_DIR points to a correct path
    # (i.e. contains eradiate.__init__.py), config init succeeds
    # This mixed setup is supported, but should give precedence to ???
    assert EradiateConfig.from_environ(
        {"ERADIATE_SOURCE_DIR": tmpdir_path / "eradiate"}
    )


def test_config_progress():
    cfg = EradiateConfig.from_environ()

    # We can set progress level using a level
    cfg.progress = ProgressLevel.NONE
    assert cfg.progress is ProgressLevel.NONE

    # We can set progress level using an int
    cfg.progress = 0
    assert cfg.progress is ProgressLevel.NONE

    # We can set progress using a string
    cfg.progress = "NONE"
    assert cfg.progress is ProgressLevel.NONE

    # Progress levels are comparable to int
    assert cfg.progress < 1
    assert cfg.progress == 0
