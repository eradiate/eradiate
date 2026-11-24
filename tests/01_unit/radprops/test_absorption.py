from axsdb import ErrorHandlingAction

from eradiate.config import settings
from eradiate.radprops import absdb_factory
from eradiate.radprops._absorption import DEFAULT_DATABASES


def test_absdb_factory_error_handling_config(mode_mono, monkeypatch):
    # The error handling setting is read when a database is created
    error_handling = settings["ABSORPTION_DATABASE"]["ERROR_HANDLING"]
    for action in ["warn", "raise"]:
        monkeypatch.setitem(
            error_handling, "t", {**error_handling["t"], "missing": action}
        )
        absdb = absdb_factory.create(DEFAULT_DATABASES["mono"])
        assert (
            absdb.error_handling_config.t.missing is ErrorHandlingAction[action.upper()]
        )
