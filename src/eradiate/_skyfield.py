"""
Astronomical computations based on Skyfield.

Skyfield is an optional dependency: it is imported, and ephemeris data are
loaded, only when a computation requires them.
"""

from __future__ import annotations

import atexit
import datetime
import functools
import importlib
import warnings
from pathlib import Path

from .config import settings

__all__ = ["earth_sun_distance", "require_skyfield"]


def require_skyfield() -> None:
    """
    Check that Skyfield is installed.

    Raises
    ------
    ModuleNotFoundError
        If Skyfield cannot be imported. A warning with installation
        instructions is emitted beforehand.
    """
    try:
        importlib.import_module("skyfield")
    except ModuleNotFoundError:
        warnings.warn(
            "To use the date-based Solar irradiance scaling feature, you "
            "must install skyfield.\n"
            "See instructions on https://rhodesmill.org/skyfield/.",
            stacklevel=3,
        )
        raise


@functools.cache
def _ephemeris():
    """
    Load the timescale and the DE421 ephemeris.

    Data are taken from the skyfield-data package if it is installed, and
    otherwise downloaded to Eradiate's cache directory. The result is cached
    for the lifetime of the interpreter.

    Returns
    -------
    ts : skyfield.timelib.Timescale
        Timescale.

    ephemeris : skyfield.jpllib.SpiceKernel
        DE421 ephemeris.
    """
    from skyfield.api import Loader

    try:
        from skyfield_data import get_skyfield_data_path
    except ImportError:
        data_dir = Path(settings["data_path"]) / "cached" / "skyfield"
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        # skyfield-data checks file expiry dates on each call; finals2000A.all
        # (Earth orientation parameters) is unused: only de421.bsp is needed
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=r"The file finals2000A\.all has expired",
            )
            data_dir = get_skyfield_data_path()

    loader = Loader(data_dir)
    ephemeris = loader("de421.bsp")
    # Skyfield keeps the ephemeris file open for performance reasons; close it
    # on exit to avoid a ResourceWarning
    atexit.register(ephemeris.close)

    return loader.timescale(), ephemeris


def earth_sun_distance(dt: datetime.datetime) -> float:
    """
    Compute the Earth-Sun distance at a given date and time.

    Parameters
    ----------
    dt : datetime.datetime
        Date and time. A naive value is interpreted as UTC.

    Returns
    -------
    float
        Earth-Sun distance in astronomical units.
    """
    ts, ephemeris = _ephemeris()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    t = ts.from_datetime(dt)

    return ephemeris["earth"].at(t).observe(ephemeris["sun"]).distance().au
