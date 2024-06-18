from __future__ import annotations

import enum
import importlib.resources
import typing as t
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

    @staticmethod
    def convert(value: t.Any) -> ProgressLevel:
        """
        Attempt conversion of a value to an :class:`.ProgressLevel`
        instance. The conversion protocol is as follows:

        * If ``value`` is a string, it is converted to upper case and passed to
          the indexing operator of :class:`.ProgressLevel`.
        * If ``value`` is an integer, it is passed to the call operator of
          :class:`.ProgressLevel`.
        * If ``value`` is a :class:`.ProgressLevel` instance, it is returned
          without change.
        * Otherwise, the method raises an exception.

        Parameters
        ----------
        value
            Value to attempt conversion of.

        Returns
        -------
        Converted value

        Raises
        ------
        TypeError
            If no conversion protocol exists for ``value``.
        """
        if isinstance(value, ProgressLevel):
            return value
        elif isinstance(value, str):
            return ProgressLevel[value.upper()]
        elif isinstance(value, int):
            return ProgressLevel(value)
        else:
            raise TypeError(f"Cannot convert a {type(value)} instance to ProgressLevel")

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
        Validator("AZIMUTH_CONVENTION", cast=AzimuthConvention.convert),
        Validator("DATA_STORE_URL", cast=str),
        Validator(
            "DOWNLOAD_DIR",
            default=_default_download_dir(),
            cast=lambda x: Path(x).resolve(),
        ),
        Validator("OFFLINE", cast=bool),
        Validator("PROGRESS", cast=ProgressLevel.convert),
        Validator("SMALL_FILES_REGISTRY_URL", cast=str),
        Validator("SMALL_FILES_REGISTRY_REVISION", cast=str),
    ],
)
