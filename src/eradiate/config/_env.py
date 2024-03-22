import os
from pathlib import Path


def _validate_source_dir(v):
    if SOURCE_DIR is None:
        # Import must be local and not use the lazy loader to avoid circular imports
        from ..kernel._versions import kernel_installed

        kernel_is_installed, _ = kernel_installed()

        if not kernel_is_installed:
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

    eradiate_init = v / "src" / "eradiate" / "__init__.py"

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


SOURCE_DIR = os.environ.get("ERADIATE_SOURCE_DIR")
if SOURCE_DIR is not None:
    SOURCE_DIR = Path(SOURCE_DIR).resolve()

ENV = os.environ.get("ERADIATE_ENV", "default")
