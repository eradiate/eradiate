"""
Global configuration components. The centric part is the
:class:`.EradiateConfig` class, whose defaults can be set using environment
variables. A single instance :data:`.config` is aliased in the top-level module
for convenience.
"""

import enum
import os.path
import pathlib

import attrs
import environ
from environ._environ_config import CNF_KEY, RAISE, _ConfigEntry
from environ.exceptions import ConfigError

from .frame import AzimuthConvention


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
        used.  Please note that it is also run for default values.

    validator
        A callable that is run with the final value. See ``attrs``'s
        `chapter on validation <https://www.attrs.org/en/stable/init.html#validators>`_
        for details.
        You can also use any validator that is
        `shipped with attrs <https://www.attrs.org/en/stable/api.html#validators>`_.

    help : str
        A help string that is used by `generate_help`.

    on_setattr : callable or list of callables or None or attrs.setters.NO_OP, optional
        This argument is directly forwarded to :func:`attrs.field`, with the notable
        difference that the default behaviour executes converters and
        validators.
    """
    if on_setattr is None:
        on_setattr = attrs.setters.pipe(attrs.setters.convert, attrs.setters.validate)

    return attrs.field(
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
    contains global configuration parameters for Eradiate. It is initialized
    using :ref:`environment variables <sec-config-env_vars>` as defaults.

    See Also
    --------
    :data:`eradiate.config`,
    `the environ-config library <https://environ-config.readthedocs.io>`_
    """

    #: Path to the Eradiate source directory.
    source_dir = var(
        converter=lambda x: x if x is None else pathlib.Path(x).absolute(),
        help="Path to the Eradiate source directory. If it is not set, then the "
        "current setup is assumed to be a production installation of Eradiate.",
        default=None,
    )

    @source_dir.validator
    def _dir_validator(self, var, dir):
        if dir is None:
            if not __import__("eradiate").kernel.ERADIATE_MITSUBA_PACKAGE:
                raise ConfigError(
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
            return

        eradiate_init = dir / "src" / "eradiate" / "__init__.py"

        if not eradiate_init.is_file():
            raise ConfigError(
                f"While configuring Eradiate: could not find {eradiate_init} file. "
                "Please make sure the 'ERADIATE_SOURCE_DIR' environment variable is "
                "correctly set to the Eradiate installation directory. If you are "
                "using Eradiate directly from the sources, you can alternatively "
                "source the provided setpath.sh script. If you wish to use Eradiate "
                "in a production environment, you can install the eradiate-mitsuba "
                "package using 'pip install eradiate-mitsuba' and unset the "
                "'ERADIATE_SOURCE_DIR' environment variable."
            ) from FileNotFoundError(eradiate_init)

    #: URL where small data files are located (production use only)
    small_files_registry_url = var(
        default="https://raw.githubusercontent.com/eradiate/eradiate-data",
        converter=str,
        help="URL where small data files are located (production use only)",
    )

    #: Revision of the small files registry (production use only)
    small_files_registry_revision = var(
        default="master",
        converter=str,
        help="Revision of the small files registry (production use only)",
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
        converter=lambda x: pathlib.Path(
            os.path.expanduser(os.path.expandvars(x))
        ).absolute(),
        help="Path to the Eradiate download directory.",
    )

    #: If ``True``, activate the offline mode. All online data stores
    #: will be disconnected.
    offline = environ.bool_var(
        default=False,
        help="If ``True``, activate the offline mode. All online data stores "
        "will be disconnected.",
    )

    #: An integer flag setting the level of progress display (see
    #: :class:`ProgressLevel`). Values are preferably set using strings
    #: (``["NONE", "SPECTRAL_LOOP", "KERNEL"]``). Only affects tqdm-based progress
    #: bars.
    progress = var(
        default="KERNEL",
        converter=lambda x: ProgressLevel[x.upper()]
        if isinstance(x, str)
        else ProgressLevel(x),
        help="An integer flag setting the level of progress display (see"
        ":class:`.ProgressLevel`). Values are preferably set using strings "
        '(``["NONE", "SPECTRAL_LOOP", "KERNEL"]``). Only affects tqdm-based '
        "progress bars.",
    )

    #: The convention applied when interpreting azimuth values as part
    #: of the specification of a direction (see :class:`.AzimuthConvention`).
    #: Values are preferably set using strings (*e.g.* ``"EAST_RIGHT"``,
    #: ``"NORTH_LEFT"``, etc.).
    azimuth_convention = var(
        default="EAST_RIGHT",
        converter=lambda x: AzimuthConvention[x.upper()] if isinstance(x, str) else x,
        validator=attrs.validators.instance_of(AzimuthConvention),
        help="The convention applied when interpreting azimuth values as part "
        "of the specification of a direction (see :class:`.AzimuthConvention`). "
        'Values are preferably set using strings (*e.g.* ``"EAST_RIGHT"``, '
        '``"NORTH_LEFT"``, etc.).',
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
