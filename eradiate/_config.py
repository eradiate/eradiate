import pathlib
import warnings

import environ
from environ.exceptions import ConfigError

from eradiate.exceptions import ConfigWarning


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
    Global configuration for Eradiate. This configuration object is written with
    the `environ-config library <https://environ-config.readthedocs.io>`_.
    """

    dir = environ.var(
        converter=lambda x: pathlib.Path(x).absolute(),
        help="Path to the Eradiate source directory.",
    )

    @dir.validator
    def _dir_validator(self, var, dir):
        eradiate_init = dir / "eradiate" / "__init__.py"

        if not eradiate_init.is_file():
            raise ConfigError(
                f"While configuring Eradiate: could not find {eradiate_init} file. "
                "Please make sure the 'ERADIATE_DIR' environment variable is "
                "correctly set to the Eradiate installation directory. If you are "
                "using Eradiate directly from the sources, you can alternatively "
                "source the provided setpath.sh script."
            ) from FileNotFoundError(eradiate_init)

    data_path = environ.var(
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
                ConfigWarning(
                    "While configuring Eradiate: 'ERADIATE_DATA_PATH' contains "
                    f"paths to nonexisting directories {[str(x) for x in do_not_exist]}"
                )
            )

    progress = environ.var(
        default=2,
        converter=int,
        help="An integer flag setting the level of progress display "
        "[0: None; 1: Spectral loop; 2: Kernel]. Only affects tqdm-based "
        "progress bars.",
    )


try:
    config = EradiateConfig.from_environ()

except environ.exceptions.MissingEnvValueError as e:
    if "ERADIATE_DIR" in e.args:
        raise ConfigError(
            f"While configuring Eradiate: environment variable '{e}' is missing. "
            "Please set this variable to the absolute path to the Eradiate "
            "installation directory. If you are using Eradiate directly from the "
            "sources, you can alternatively source the provided setpath.sh script."
        ) from e
    else:
        raise e
