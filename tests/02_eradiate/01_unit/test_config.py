from pathlib import Path

import pytest
from environ.exceptions import ConfigError, MissingEnvValueError

from eradiate._config import EradiateConfig, ProgressLevel
from eradiate.exceptions import ConfigWarning


def test_config(tmpdir):
    # Create an appropriate file tree for tests
    tmpdir_path = Path(tmpdir)
    (tmpdir_path / "eradiate/src/eradiate").mkdir(parents=True)
    (tmpdir_path / "eradiate/src/eradiate/__init__.py").touch()

    # Raises if ERADIATE_SOURCE_DIR is missing
    with pytest.raises(MissingEnvValueError):
        EradiateConfig.from_environ({})

    # When ERADIATE_SOURCE_DIR points to a correct path
    # (i.e. contains eradiate.__init__.py), config init succeeds
    assert EradiateConfig.from_environ(
        {"ERADIATE_SOURCE_DIR": tmpdir_path / "eradiate"}
    )

    # Otherwise it raises
    with pytest.raises(ConfigError):
        EradiateConfig.from_environ({"ERADIATE_SOURCE_DIR": tmpdir_path})


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
