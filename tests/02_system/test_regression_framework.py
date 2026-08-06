"""
Self-check of the regression testing framework: type I and type II error rates
of :class:`.ZTest`, measured on real simulation output.

This is not a regression test — it has no stored reference. It renders the same
RAMI4ATM scene twice and compares the two results against each other, which is
the only configuration exercising the framework's real data model:
per-viewing-angle means with their Monte Carlo variance.
"""

import functools

import pytest

import eradiate
from eradiate.test_tools.regression import ZTest
from eradiate.test_tools.test_cases import rami4atm

CASE = "hom00_bla_sd2s_m03_z30a000_brfpp"

# Sample count, matching the production regression test (test_rami4atm)
SPP = 1000

# Family-wise significance level. The type I error rate of the framework is
# this value by construction, so keep it low enough for the check not to be
# flaky.
THRESHOLD = 1e-4

# Variable under test, matching the production regression test. It is the
# band-integrated radiance, so a comparison pairs one value per viewing angle:
# n = 76.
VARIABLE = "radiance_srf"

#  Relative radiance bias the framework must detect. The decision keys on the
#  most extreme of n = 76 comparisons, so at ``THRESHOLD`` the Šidák-corrected
#  per-comparison level is ~1.3e-6, i.e. a rejection past ~4.85 sigma; the
#  detection floor is that many standard errors of the paired difference,
#  √2 × the per-pixel standard error. At this sample count the per-pixel
#  relative standard error measures 0.6 % (0.33 % to 0.81 % across viewing
#  angles) and the paired difference 0.87 %, band integration over the CKD bins
#  having shrunk it well below the ~2 % the per-bin radiance carried.
#
#  Measured family p-values, single realization: bias 0 → 0.11, 1 % → 9e-4
#  (accepted), 2 % → 2e-6, 3 % → 5e-11, 5 % → 6e-29, 10 % → 7e-110. Detection
#  therefore sets in just under 2 %, and this bias clears the floor by a wide
#  margin — the check is about the framework working at all, not about
#  pinpointing its sensitivity.
BIAS = 0.1


@functools.cache
def _render_pair():
    """
    Two independent renders of the same scene. Cached: both tests below compare
    the same pair.
    """
    (exp,) = rami4atm.CASES[CASE].make_experiments(spp=SPP)
    return eradiate.run(exp), eradiate.run(exp)


def _evaluate(value, reference):
    return ZTest(THRESHOLD, variable=VARIABLE).evaluate(value, reference)


@pytest.mark.slow
def test_false_alarm(mode_ckd_double):
    """
    *Type I error (false alarm)*

    Two independent renders of the same scene differ only by Monte Carlo noise,
    so the test must accept them. A failure means the framework rejects data
    drawn from the same distribution, i.e. its false alarm rate exceeds the
    threshold it advertises.
    """
    result, reference = _render_pair()
    outcome = _evaluate(result, reference)
    assert outcome.passed, (
        f"family p-value {outcome.metric_value} <= threshold {THRESHOLD}"
    )


@pytest.mark.slow
def test_missed_regression(mode_ckd_double):
    """
    *Type II error (missed regression)*

    Scaling the reference radiance by ``1 + BIAS`` introduces a bias large
    compared with the Monte Carlo noise, so the test must reject it. A failure
    means the framework misses a regression of that size.
    """
    result, reference = _render_pair()
    biased = reference.copy()
    biased[VARIABLE] = reference[VARIABLE] * (1.0 + BIAS)

    outcome = _evaluate(result, biased)
    assert not outcome.passed, (
        f"family p-value {outcome.metric_value} > threshold {THRESHOLD}"
    )
