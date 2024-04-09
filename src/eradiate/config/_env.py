from __future__ import annotations

import os
from pathlib import Path


def _validate_source_dir(value: Path | None):
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


#: Path to the Eradiate source code directory, if relevant. Takes the value of
#: the ``ERADIATE_SOURCE_DIR`` environment variable if it is set; otherwise
#: defaults to ``None``.
SOURCE_DIR: Path | None = os.environ.get("ERADIATE_SOURCE_DIR", default=None)
if SOURCE_DIR is not None:
    SOURCE_DIR = Path(SOURCE_DIR).resolve()
_validate_source_dir(SOURCE_DIR)

#: Identifier of the environment in which Eradiate is used. Takes the value of
#: the ``ERADIATE_ENV`` environment variable if it is set; otherwise defaults to
#: ``"default"``.
ENV: str = os.environ.get("ERADIATE_ENV", "default")
