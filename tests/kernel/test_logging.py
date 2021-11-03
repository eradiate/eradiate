import logging

import pytest

import eradiate


def test_logging_setup(mode_mono):
    eradiate.kernel.logging.install_logging()
    assert eradiate.kernel.logging.mts_logger is eradiate.kernel.logging._get_logger()


def test_logging_mitsuba(mode_mono, caplog):
    eradiate.kernel.logging.install_logging()
    from mitsuba.core import Log, LogLevel

    # TRACE log level is defined and set to the expected value
    assert logging.TRACE == logging.DEBUG - 5

    # Set log capture fixture level such that we can capture all log output
    caplog.set_level(logging.TRACE)

    # Trace and debug are not shown by default (Mitsuba kills them before they
    # can be passed to the Python logger)
    Log(LogLevel.Trace, "trace message")
    Log(LogLevel.Debug, "debug message")
    assert not caplog.records

    # We can raise log level
    eradiate.kernel.logging.mts_logger.set_log_level(LogLevel.Trace)
    Log(LogLevel.Trace, "trace message")
    record = caplog.records.pop()
    assert record.levelname == "TRACE"
    assert "trace message" in record.msg

    Log(LogLevel.Debug, "debug message")
    record = caplog.records.pop()
    assert record.levelname == "DEBUG"
    assert "debug message" in record.msg

    # Info message is displayed
    Log(LogLevel.Info, "info message")
    record = caplog.records.pop()
    assert record.levelname == "INFO"
    assert "info message" in record.msg

    # Warning message as well
    Log(LogLevel.Warn, "warning message")
    record = caplog.records.pop()
    assert record.levelname == "WARNING"
    assert "warning message" in record.msg

    # Error message raises
    with pytest.raises(RuntimeError):
        Log(LogLevel.Error, "boom")
