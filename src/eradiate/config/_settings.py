from __future__ import annotations

import enum
import os
import typing as t
from pathlib import Path

from dynaconf import Dynaconf, Validator

from . import _defaults
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


def _validate_source_dir(value: Path | None) -> bool:
    if value is None:
        # Import must be local and not use the lazy loader to avoid circular imports
        from ..kernel._versions import kernel_installed

        # Detect whether kernel is installed
        kernel_is_installed, _ = kernel_installed()

        # Detect Read the Docs build
        rtd = os.environ.get("READTHEDOCS", "") == "True"

        if not kernel_is_installed and not rtd:
            raise RuntimeError(
                "Could not find a suitable production installation for the "
                "Eradiate kernel. This is either because you are using Eradiate "
                "in a production environment without having the eradiate-mitsuba "
                "package installed, or because you are using Eradiate directly "
                "from the sources. In the latter case, please make sure the "
                "'ERADIATE_SOURCE_DIR' environment variable is correctly set to "
                "the Eradiate installation directory. If you are using Eradiate "
                "directly from the sources, you can alternatively source the "
                "provided setpath.sh script. You can install the eradiate-mitsuba "
                "package using 'pip install eradiate-mitsuba'."
            )

    else:
        eradiate_init = value / "src" / "eradiate" / "__init__.py"

        if not eradiate_init.is_file():
            raise RuntimeError(
                f"While configuring Eradiate: could not find {eradiate_init} file. "
                "Please make sure the 'ERADIATE_SOURCE_DIR' environment variable is "
                "correctly set to the Eradiate installation directory. If you are "
                "using Eradiate directly from the sources, you can alternatively "
                "source the provided setpath.sh script. If you wish to use Eradiate "
                "in a production environment, you can install the eradiate-mitsuba "
                "package using 'pip install eradiate-mitsuba' and unset the "
                "'ERADIATE_SOURCE_DIR' environment variable."
            ) from FileNotFoundError(eradiate_init)

    return True


#: Main settings data structure. See the `Dynaconf documentation <https://www.dynaconf.com/>`__
#: for details.
settings = Dynaconf(
    settings_files=["eradiate.yml", "eradiate.yaml", "eradiate.toml"],
    envvar_prefix="ERADIATE",
    merge_enabled=True,
    validate_on_update=True,
    validators=[
        Validator(  # Process first, other parameters might depend on it
            "SOURCE_DIR",
            cast=lambda x: Path(x).resolve() if x is not None else None,
            condition=_validate_source_dir,
            default=_defaults.source_dir,
        ),
        Validator(
            "ABSORPTION_DATABASE.ERROR_HANDLING",
            default=_defaults.absorption_database__error_handling,
        ),
        Validator(
            "AZIMUTH_CONVENTION",
            cast=AzimuthConvention.convert,
            default=_defaults.azimuth_convention,
        ),
        Validator(
            "DATA_URL",
            cast=str,
            default=_defaults.data_url,
        ),
        Validator(
            "DATA_PATH",
            cast=Path,
            default=_defaults.data_path,
        ),
        Validator(
            "OFFLINE",
            cast=bool,
            default=_defaults.offline,
        ),
        Validator(
            "PATH",
            cast=list,
            default=_defaults.path,
        ),
        Validator(
            "PROGRESS",
            cast=ProgressLevel.convert,
            default=_defaults.progress,
        ),
    ],
)


def __getattr__(name):
    # Called if attribute cannot be resolved
    if name == "SOURCE_DIR":
        return settings.get("SOURCE_DIR")

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
