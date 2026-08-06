"""
Tests for the test report logging system.
"""

import logging
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from eradiate.test_tools.report import ReportLogger, figure_to_html, report_logger


def test_figure_to_html():
    """
    The HTML rendering of a figure is a self-contained <svg> fragment with
    theme-aware styling.
    """
    fig, _ = plt.subplots()
    html = figure_to_html(fig)
    plt.close(fig)

    assert html.startswith("<svg")
    assert html.rstrip().endswith("</svg>")
    # The XML prolog and DOCTYPE must be stripped: this is embedded in HTML
    assert "<?xml" not in html
    assert "<!DOCTYPE" not in html
    # The style is injected right after the root tag, and must leave the colours
    # Matplotlib sets itself alone
    assert "<style>path:not([style]) { fill: var(--text-color); }</style>" in html
    assert html.index("<style>") < html.index("<metadata>")


class TestReportLogger:
    """
    Test report infrastructure tests.
    """

    class RobotLoggerStub:
        """
        Robot Framework logger stand-in that records calls.
        """

        def __init__(self):
            self.calls = []

        def info(self, msg, **kwargs):
            self.calls.append((msg, kwargs))

    def test_logging_fallback(self, caplog):
        """
        Without the Robot backend, messages go to standard logging and HTML
        fragments are discarded silently.
        """
        logger = ReportLogger(use_robot=False)

        with caplog.at_level(logging.INFO, logger="eradiate.test_tools.report"):
            logger.info("hello report")
            logger.html("<svg></svg>")

        assert "hello report" in caplog.text
        assert "<svg></svg>" not in caplog.text

    def test_robot_delegation(self, monkeypatch):
        """
        With the Robot backend, messages and HTML fragments are forwarded to
        the Robot logger with the expected flags.
        """
        # Importing the submodule also binds it on robot.api, which is what
        # makes it patchable below
        pytest.importorskip("robot.api.logger")

        # The backend is resolved at call time by importing robot.api.logger:
        # substitute a stub there to capture the forwarded calls
        stub = self.RobotLoggerStub()
        monkeypatch.setattr("robot.api.logger", stub)

        logger = ReportLogger(use_robot=True)
        assert logger._robot is stub

        logger.info("message")
        logger.html("<b>fragment</b>")

        assert stub.calls == [
            ("message", {"also_console": True}),
            ("<b>fragment</b>", {"html": True, "also_console": False}),
        ]

    @pytest.mark.parametrize("active", [False, True])
    def test_robot_autoselection(self, monkeypatch, active):
        """
        With ``use_robot`` unset, the Robot backend is selected only during an
        active Robot run: outside one, ``robot.api.logger`` falls back to
        standard logging and would dump HTML fragments into pytest's captured
        log.
        """
        pytest.importorskip("robot")

        # Swap the module-level lookup rather than the real context stack: an
        # actual Robot run (report tasks) holds its own reference to it and
        # must keep pushing and popping contexts undisturbed
        monkeypatch.setattr(
            "robot.running.context.EXECUTION_CONTEXTS",
            SimpleNamespace(current=object() if active else None),
        )

        assert ReportLogger().reporting is active

    def test_smoke(self):
        """
        Send a message and a chart through the default report logger. This test
        always passes: its purpose is to leave one known entry in the generated
        test report (e.g. ``reports/log.html``) so that the reporting pipeline
        can be checked end-to-end by visual inspection. Without an active report
        backend, both calls are no-ops.
        """
        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [0.0, 1.0])

        report_logger.info(
            "Report smoke test: this message should appear in the test report"
        )
        report_logger.html(figure_to_html(fig))
        plt.close(fig)
