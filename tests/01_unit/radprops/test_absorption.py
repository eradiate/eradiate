from axsdb import ErrorHandlingAction, ErrorHandlingConfiguration

from eradiate.radprops._absorption import DEFAULT_DATABASES, _init_absdb_factory


def test_init_absdb_factory(mode_mono):
    # Check that the error handling policy applied to the absorption database
    # factory propagates to created instances
    for action in ["warn", "raise"]:
        error_handling_config = ErrorHandlingConfiguration.convert(
            {"t": {"missing": action}}
        )
        factory = _init_absdb_factory(error_handling_config=error_handling_config)
        absdb = factory.create(DEFAULT_DATABASES["mono"])
        assert (
            absdb.error_handling_config.t.missing is ErrorHandlingAction[action.upper()]
        )
