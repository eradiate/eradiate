from pathlib import Path

import pytest
from environ.exceptions import ConfigError, MissingEnvValueError

from eradiate._config import EradiateConfig
from eradiate.exceptions import ConfigWarning


def test_config(tmpdir):
    # Create an appropriate file tree for tests
    tmpdir_path = Path(tmpdir)
    (tmpdir_path / "eradiate/eradiate").mkdir(parents=True)
    (tmpdir_path / "eradiate/eradiate/__init__.py").touch()

    # Raises if ERADIATE_DIR is missing
    with pytest.raises(MissingEnvValueError):
        EradiateConfig.from_environ({})

    # When ERADIATE_DIR points to a correct path (i.e. contains eradiate.__init__.py),
    # config init succeeds
    assert EradiateConfig.from_environ({"ERADIATE_DIR": tmpdir_path / "eradiate"})

    # Otherwise it raises
    with pytest.raises(ConfigError):
        EradiateConfig.from_environ({"ERADIATE_DIR": tmpdir_path})

    # Warns if nonexisting paths are passed
    with pytest.warns(ConfigWarning):
        paths = [
            f"{tmpdir_path / 'eradiate_data_0'}",
            f"{tmpdir_path / 'eradiate_data_1'}",
        ]

        # Colon-separated paths are processed correctly
        cfg = EradiateConfig.from_environ(
            {
                "ERADIATE_DIR": tmpdir_path / "eradiate",
                "ERADIATE_DATA_PATH": ":".join(paths),
            }
        )

        assert [str(x) for x in cfg.data_path] == paths

        # Empty paths are omitted
        cfg = EradiateConfig.from_environ(
            {
                "ERADIATE_DIR": tmpdir_path / "eradiate",
                "ERADIATE_DATA_PATH": ":::" + "::".join(paths) + ":",
            }
        )

        assert [str(x) for x in cfg.data_path] == paths
