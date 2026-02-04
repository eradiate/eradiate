from __future__ import annotations

import logging
from typing import Literal

from axsdb import (
    AbsorptionDatabase,
    AbsorptionDatabaseFactory,
    CKDAbsorptionDatabase,
    ErrorHandlingConfiguration,
    MonoAbsorptionDatabase,
)

import eradiate

from .._mode import Mode
from ..data import fresolver
from ..exceptions import UnsupportedModeError

logger = logging.getLogger(__name__)


def get_error_handling_config():
    from ..config import settings

    return ErrorHandlingConfiguration.convert(
        settings.get("ABSORPTION_DATABASE.ERROR_HANDLING")
    )


KNOWN_DATABASES = {
    "gecko": {
        "cls": MonoAbsorptionDatabase,
        "path": "absorption_mono/gecko",
        "kwargs": {"lazy": True},
    },
    "komodo": {
        "cls": MonoAbsorptionDatabase,
        "path": "absorption_mono/komodo",
        "kwargs": {"lazy": True},
    },
    "monotropa": {
        "cls": CKDAbsorptionDatabase,
        "path": "absorption_ckd/monotropa",
    },
    "mycena": {
        "cls": CKDAbsorptionDatabase,
        "path": "absorption_ckd/mycena",
    },
    "panellus": {
        "cls": CKDAbsorptionDatabase,
        "path": "absorption_ckd/panellus",
    },
    "tuber": {
        "cls": CKDAbsorptionDatabase,
        "path": "absorption_ckd/tuber",
    },
}

DEFAULT_DATABASES = {"mono": "komodo", "ckd": "monotropa"}


def _init_absdb_factory(
    error_handling_config: ErrorHandlingConfiguration | None = None,
) -> AbsorptionDatabaseFactory:
    """
    Initialize ``absdb_factory`` according to configuration defaults.
    """
    absdb_factory = AbsorptionDatabaseFactory()
    error_handling_config = error_handling_config or get_error_handling_config()

    for k, v in KNOWN_DATABASES.items():

        def resolve_path(path=str(v["path"])):
            return fresolver.resolve(path)

        kwargs = {"error_handling_config": error_handling_config}
        if "kwargs" in v:
            kwargs.update(v["kwargs"])

        absdb_factory.register(name=k, cls=v["cls"], path=resolve_path, kwargs=kwargs)

    return absdb_factory


#: Absorption database factory
absdb_factory: AbsorptionDatabaseFactory = _init_absdb_factory()


def get_default_absdb(
    mode: Literal["mono", "ckd"] | Mode | None = None,
) -> AbsorptionDatabase:
    """
    Return a default absorption database, depending on the active mode.

    Defaults are as follows:

    * Monochromatic: ``"komodo"``
    * CKD: ``"monotropa"``

    Parameters
    ----------
    mode : {"mono", "ckd"} or Mode, optional
        The spectral mode for which the default database is returned. If unset,
        the current active mode is used.

    Returns
    -------
    AbsorptionDatabase
    """
    mode_error = UnsupportedModeError(supported=["mono", "ckd"])

    if mode is None:
        mode = eradiate.get_mode()

    if isinstance(mode, Mode):
        if mode.is_mono:
            mode = "mono"
        elif mode.is_ckd:
            mode = "ckd"
        else:
            raise mode_error

    if mode == "mono":
        default = "komodo"
    elif mode == "ckd":
        default = "monotropa"
    else:
        raise mode_error

    return absdb_factory.create(default)
