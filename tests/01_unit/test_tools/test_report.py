"""
Tests for the test report logging system.
"""

import logging

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
    assert "var(--text-color)" in html
    # Attributes of the wrapping <svg> tag must be separated
    assert 'viewBox="0 0 810 540"\nxmlns="http://www.w3.org/2000/svg">' in html


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

    def test_robot_delegation(self):
        """
        With the Robot backend, messages and HTML fragments are forwarded to
        the Robot logger with the expected flags.
        """
        pytest.importorskip("robot")

        # Exercise the Robot backend selection, then substitute a stub to
        # capture the forwarded calls
        logger = ReportLogger(use_robot=True)
        assert logger._robot is not None
        logger._robot = stub = self.RobotLoggerStub()

        logger.info("message")
        logger.html("<b>fragment</b>")

        assert stub.calls == [
            ("message", {"also_console": True}),
            ("<b>fragment</b>", {"html": True, "also_console": False}),
        ]

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
