"""
Unit tests for the One Euro Filter.

Covers constant-signal preservation, smoothing of a step, the internal
alpha formula, and adaptive tracking of fast movement.
"""

import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests  # noqa: F401

from model.extraction.one_euro_filter import OneEuroFilter

logger = logging.getLogger(__name__)


def test_constant_signal_is_preserved():
    """A constant input should produce a (near-)constant output."""
    f = OneEuroFilter(t0=0.0, x0=5.0)

    outputs = [f(t=i * 0.01, x=5.0) for i in range(1, 50)]

    logger.info("=== Constant Signal ===")
    logger.info("  last three outputs: %s", outputs[-3:])
    assert all(abs(o - 5.0) < 1e-6 for o in outputs)
    logger.info("  PASSED\n")


def test_step_response_is_smoothed():
    """A sudden step should not fully reach the target in one sample."""
    f = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=0.0)

    out = f(t=0.01, x=10.0)

    logger.info("=== Step Smoothing ===")
    logger.info("  filtered step (target=10): %.3f", out)
    assert 0.0 < out < 10.0
    logger.info("  PASSED\n")


def test_non_positive_dt_returns_raw_value():
    """If the elapsed time is zero or negative, the raw value is returned."""
    f = OneEuroFilter(t0=1.0, x0=0.0)

    same_time = f(t=1.0, x=7.5)
    earlier = f(t=0.5, x=9.25)

    logger.info("=== Non-positive dt ===")
    logger.info("  same-time output: %s, earlier-time output: %s", same_time, earlier)
    assert same_time == 7.5
    assert earlier == 9.25
    logger.info("  PASSED\n")


def test_alpha_formula_matches_reference():
    """Internal _alpha should match the 1 / (1 + tau/te) definition."""
    f = OneEuroFilter(t0=0.0, x0=0.0)

    te = 0.02
    cutoff = 2.0
    expected = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / te)

    logger.info("=== Alpha Formula ===")
    logger.info("  expected=%.6f, actual=%.6f", expected, f._alpha(te, cutoff))
    assert abs(f._alpha(te, cutoff) - expected) < 1e-9
    logger.info("  PASSED\n")


def test_high_beta_tracks_fast_movement():
    """A higher beta should produce less lag on a fast-moving signal."""
    low_beta = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=0.0)
    high_beta = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=50.0)

    for i in range(1, 20):
        t = i * 0.01
        x = float(i)
        low_out = low_beta(t, x)
        high_out = high_beta(t, x)

    logger.info("=== Adaptive Tracking ===")
    logger.info("  low-beta final: %.3f, high-beta final: %.3f, target: 19.0", low_out, high_out)
    assert high_out > low_out
    logger.info("  PASSED\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    test_constant_signal_is_preserved()
    test_step_response_is_smoothed()
    test_non_positive_dt_returns_raw_value()
    test_alpha_formula_matches_reference()
    test_high_beta_tracks_fast_movement()
    logger.info("All tests passed!")
