"""
This module provides defaults settings for Eradiate. Dynaconf's double
underscore convention is used to represent parameter dotted hierarchical naming.
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs


def absorption_database__error_handling(settings=None, validator=None) -> dict:
    return {
        # Ignore bound errors on pressure and temperature because this usually
        # occurs at high altitude, where the absorption coefficient is very low
        # and can be safely forced to 0
        "p": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
        "t": {"missing": "raise", "scalar": "raise", "bounds": "ignore"},
        # Ignore missing molecule coordinates, raise on bound error
        "x": {"missing": "ignore", "scalar": "ignore", "bounds": "raise"},
    }


def azimuth_convention(settings=None, validator=None) -> str:
    return "east_right"


def data_path(settings=None, validator=None) -> Path:
    return Path(platformdirs.user_cache_dir(appname="eradiate"))


def download_dir(settings, validator=None) -> Path:
    source_dir = settings.get("SOURCE_DIR")

    return (
        Path("./eradiate_downloads").absolute().resolve()
        if source_dir is None
        else source_dir / ".eradiate_downloads"
    )


def data_store_url(settings=None, validator=None) -> str:
    return "https://eradiate.eu/data/store/"


def data_url(settings=None, validator=None) -> str:
    return "https://eradiate-data-registry.s3.eu-west-3.amazonaws.com/registry-v1/"


def offline(settings=None, validator=None) -> bool:
    return False


def path(settings=None, validator=None) -> list:
    return []


def progress(settings=None, validator=None) -> str:
    return "spectral_loop"


def small_files_registry_url(settings=None, validator=None) -> str:
    return "https://raw.githubusercontent.com/eradiate/eradiate-data"


def small_files_registry_revision(settings=None, validator=None) -> str:
    return "master"


def source_dir(settings=None, validator=None) -> str | None:
    # This is slightly hacky, but so far the best we've found
    return os.environ.get("ERADIATE_SOURCE_DIR", None)
