from __future__ import annotations

import enum
from pathlib import Path

import importlib_resources
from dynaconf import Dynaconf, Validator

from ._env import ENV
from ..frame import AzimuthConvention


class ProgressLevel(enum.IntEnum):
    """
    An enumeration defining valid progress levels.

    This is an integer enumeration, meaning that levels can be compared.
    """

    NONE = 0  #: No progress
    SPECTRAL_LOOP = enum.auto()  #: Up to spectral loop level progress
    KERNEL = enum.auto()  #: Up to kernel level progress


_DEFAULT_SETTINGS = importlib_resources.files("eradiate.config").joinpath(f"{ENV}.toml")

SETTINGS = Dynaconf(
    settings_files=[_DEFAULT_SETTINGS, "eradiate.toml"],
    env_prefix="ERADIATE",
    merge_enabled=True,
    validate_on_update=True,
    validators=[
        Validator(
            "AZIMUTH_CONVENTION",
            cast=lambda v: AzimuthConvention[v.upper()] if isinstance(v, str) else v,
        ),
        Validator("DATA_STORE_URL", cast=str),
        Validator(
            "DOWNLOAD_DIR",
            default=None,
            cast=lambda x: x if x is None else Path(x).resolve(),
        ),
        Validator("OFFLINE", cast=bool),
        Validator(
            "PROGRESS_LEVEL",
            cast=lambda v: (
                ProgressLevel[v.upper()] if isinstance(v, str) else ProgressLevel(v)
            ),
        ),
        Validator("SMALL_FILES_REGISTRY_URL", cast=str),
        Validator("SMALL_FILES_REGISTRY_REVISION", cast=str),
    ],
)
