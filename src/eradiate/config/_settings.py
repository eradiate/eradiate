from __future__ import annotations

import enum
import importlib.resources
from pathlib import Path

from dynaconf import Dynaconf, Validator

from ._env import ENV, SOURCE_DIR
from ..frame import AzimuthConvention


class ProgressLevel(enum.IntEnum):
    """
    An enumeration defining valid progress levels.

    This is an integer enumeration, meaning that levels can be compared to
    numerics.
    """

    NONE = 0  #: No progress
    SPECTRAL_LOOP = enum.auto()  #: Up to spectral loop level progress
    KERNEL = enum.auto()  #: Up to kernel level progress


def _default_download_dir():
    return (
        Path("./eradiate_downloads").absolute().resolve()
        if SOURCE_DIR is None
        else SOURCE_DIR / ".eradiate_downloads"
    )


DEFAULTS = importlib.resources.files("eradiate.config").joinpath(f"{ENV}.toml")

#: Main settings data structure. See the `Dynaconf documentation <https://www.dynaconf.com/>`_
#: for details.
settings = Dynaconf(
    settings_files=[DEFAULTS, "eradiate.toml"],
    envvar_prefix="ERADIATE",
    merge_enabled=True,
    validate_on_update=True,
    validators=[
        Validator(
            "AZIMUTH_CONVENTION",
            cast=lambda x: (AzimuthConvention[x.upper()] if isinstance(x, str) else x),
        ),
        Validator("DATA_STORE_URL", cast=str),
        Validator(
            "DOWNLOAD_DIR",
            default=_default_download_dir(),
            cast=lambda x: Path(x).resolve(),
        ),
        Validator("OFFLINE", cast=bool),
        Validator(
            "PROGRESS",
            cast=lambda x: (
                ProgressLevel[x.upper()] if isinstance(x, str) else ProgressLevel(x)
            ),
        ),
        Validator("SMALL_FILES_REGISTRY_URL", cast=str),
        Validator("SMALL_FILES_REGISTRY_REVISION", cast=str),
    ],
)
