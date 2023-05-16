import logging

import mitsuba as mi
import pytest

import eradiate


def test_logging_setup(mode_mono):
    eradiate.kernel.install_logging()
    assert eradiate.kernel.logging.mi_logger is eradiate.kernel.logging._get_logger()


def test_logging_mitsuba(mode_mono, caplog):
    eradiate.kernel.install_logging()

    # TRACE log level is defined and set to the expected value
    assert logging.TRACE == logging.DEBUG - 5

    # Set log capture fixture level such that we can capture all log output
    caplog.set_level(logging.TRACE)

    # Trace and debug are not shown by default (Mitsuba kills them before they
    # can be passed to the Python logger)
    mi.Log(mi.LogLevel.Trace, "trace message")
    mi.Log(mi.LogLevel.Debug, "debug message")
    assert not caplog.records

    # We can raise log level
    eradiate.kernel.logging.mi_logger.set_log_level(mi.LogLevel.Trace)
    mi.Log(mi.LogLevel.Trace, "trace message")
    record = caplog.records.pop()
    assert record.levelname == "TRACE"
    assert "trace message" in record.msg

    mi.Log(mi.LogLevel.Debug, "debug message")
    record = caplog.records.pop()
    assert record.levelname == "DEBUG"
    assert "debug message" in record.msg

    # Info message is displayed
    mi.Log(mi.LogLevel.Info, "info message")
    record = caplog.records.pop()
    assert record.levelname == "INFO"
    assert "info message" in record.msg

    # Warning message as well
    mi.Log(mi.LogLevel.Warn, "warning message")
    record = caplog.records.pop()
    assert record.levelname == "WARNING"
    assert "warning message" in record.msg

    # Error message raises
    with pytest.raises(RuntimeError):
        mi.Log(mi.LogLevel.Error, "boom")
