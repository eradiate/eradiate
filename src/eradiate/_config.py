"""
Global configuration components. The centric part is the
:class:`.EradiateConfig` class, whose defaults can be set using environment
variables. A single instance :data:`.config` is aliased in the top-level module
for convenience.
"""

import enum
import os.path
import pathlib
import warnings

import attr
import environ
from environ._environ_config import CNF_KEY, RAISE, _ConfigEntry
from environ.exceptions import ConfigError

from .exceptions import ConfigWarning


def var(
    default=RAISE,
    converter=None,
    name=None,
    validator=None,
    help=None,
    on_setattr=None,
):
    """
    Reimplementation of `environ-config`'s :func:`var` with `on_setattr` support.

    Declare a configuration attribute on the body of `config`-decorated class.

    It will be attempted to be filled from an environment variable based on the
    prefix and *name*.

    Parameters
    ----------
    default
        Setting this to a value makes the config attribute optional.

    name : str
        Overwrite name detection with a string.  If not set, the name of the
        attribute is used.

    converter
        A callable that is run with the found value and its return value is
        used.  Please not that it is also run for default values.

    validator
        A callable that is run with the final value. See ``attrs``'s
        `chapter on validation <https://www.attrs.org/en/stable/init.html#validators>`_
        for details.
        You can also use any validator that is
        `shipped with attrs <https://www.attrs.org/en/stable/api.html#validators>`_.

    help : str
        A help string that is used by `generate_help`.

    on_setattr : callable or list of callables or None or attr.setters.NO_OP, optional
        This argument is directly forwarded to :func:`attr.ib`, with the notable
        difference that the default behaviour executes converters and
        validators.
    """
    if on_setattr is None:
        on_setattr = attr.setters.pipe(attr.setters.convert, attr.setters.validate)

    return attr.ib(
        default=default,
        metadata={CNF_KEY: _ConfigEntry(name, default, None, None, help)},
        converter=converter,
        validator=validator,
        on_setattr=on_setattr,
    )


class ProgressLevel(enum.IntEnum):
    """
    An enumeration defining valid progress levels.

    This is an integer enumeration, meaning that levels can be compared.
    """

    NONE = 0  #: No progress
    SPECTRAL_LOOP = enum.auto()  #: Up to spectral loop level progress
    KERNEL = enum.auto()  #: Up to kernel level progress


def format_help_dicts_rst(help_dicts, display_defaults=False):
    """
    Help dictionary formatter for environment variable help generation.
    """
    help_strs = []
    for help_dict in help_dicts:
        help_str = f".. envvar:: {help_dict['var_name']}\n\n"
        help_str += f"   *{'Required' if help_dict['required'] else 'Optional'}"

        if help_dict.get("default") and display_defaults:
            help_str += f", default={repr(help_dict['default'])}.* "
        else:
            help_str += "*. "

        if help_dict.get("help_str"):
            help_str += f"{help_dict['help_str']}"
        help_strs.append(help_str)

    return "\n\n".join(help_strs)


@environ.config(prefix="ERADIATE")
class EradiateConfig:
    """
    Global configuration for Eradiate.

    This class, instantiated once as the :data:`eradiate.config` attribute,
    contains global configuration parameters for Eradiate. It is initialised
    using :ref:`environment variables <sec-config-env_vars>` as defaults.

    See Also
    --------
    :data:`eradiate.config`,
    `the environ-config library <https://environ-config.readthedocs.io>`_
    """

    #: Path to the Eradiate source directory.
    source_dir = var(
        converter=lambda x: pathlib.Path(x).absolute(),
        help="Path to the Eradiate source directory.",
    )

    @source_dir.validator
    def _dir_validator(self, var, dir):
        eradiate_init = dir / "src" / "eradiate" / "__init__.py"

        if not eradiate_init.is_file():
            raise ConfigError(
                f"While configuring Eradiate: could not find {eradiate_init} file. "
                "Please make sure the 'ERADIATE_SOURCE_DIR' environment variable is "
                "correctly set to the Eradiate installation directory. If you are "
                "using Eradiate directly from the sources, you can alternatively "
                "source the provided setpath.sh script."
            ) from FileNotFoundError(eradiate_init)

    #: A colon-separated list of paths where to search for data files.
    data_path = var(
        default=None,
        converter=lambda x: [pathlib.Path(y) for y in x.split(":") if y]
        if isinstance(x, str)
        else x,
        help="A colon-separated list of paths where to search for data files.",
    )

    @data_path.validator
    def _data_path_validator(self, var, paths):
        if paths is None:
            return

        do_not_exist = []

        for path in paths:
            if not path.is_dir():
                do_not_exist.append(path)

        if do_not_exist:
            warnings.warn(
                "While configuring Eradiate: 'ERADIATE_DATA_PATH' contains "
                f"paths to nonexisting directories {[str(x) for x in do_not_exist]}",
                ConfigWarning,
            )

    #: URL where large data files are located.
    data_store_url = var(
        default="http://eradiate.eu/data/store/",
        converter=str,
        help="URL where large data files are located.",
    )

    #: Path to the Eradiate download directory.
    download_dir = var(
        default="$ERADIATE_SOURCE_DIR/resources/downloads",
        converter=lambda x: pathlib.Path(os.path.expandvars(x)).absolute(),
        help="Path to the Eradiate download directory.",
    )

    #: An integer flag setting the level of progress display (see
    #: :class:`ProgressLevel`). Values are preferrably using strings
    #: (``["NONE", "SPECTRAL_LOOP", "KERNEL"]``). Only affects tqdm-based progress
    #: bars.
    progress = var(
        default="KERNEL",
        converter=lambda x: ProgressLevel[x.upper()]
        if isinstance(x, str)
        else ProgressLevel(x),
        help="An integer flag setting the level of progress display (see"
        ":class:`.ProgressLevel`). Values are preferrably using strings "
        '(``["NONE", "SPECTRAL_LOOP", "KERNEL"]``). Only affects tqdm-based '
        "progress bars.",
    )


try:
    #: Global configuration object instance.
    #: See :class:`EradiateConfig`.
    config = EradiateConfig.from_environ()

except environ.exceptions.MissingEnvValueError as e:
    if "ERADIATE_SOURCE_DIR" in e.args:
        raise ConfigError(
            f"While configuring Eradiate: environment variable '{e}' is missing. "
            "Please set this variable to the absolute path to the Eradiate "
            "installation directory. If you are using Eradiate directly from the "
            "sources, you can alternatively source the provided setpath.sh script."
        ) from e
    else:
        raise e
