"""
Unit tests for the One Euro Filter.

Covers constant-signal preservation, smoothing of a step, the internal
alpha formula, and adaptive tracking of fast movement.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests  # noqa: F401

from model.extraction.one_euro_filter import OneEuroFilter


def test_constant_signal_is_preserved():
    """A constant input should produce a (near-)constant output."""
    f = OneEuroFilter(t0=0.0, x0=5.0)

    outputs = [f(t=i * 0.01, x=5.0) for i in range(1, 50)]

    print("=== Constant Signal ===")
    print(f"  last three outputs: {outputs[-3:]}")
    assert all(abs(o - 5.0) < 1e-6 for o in outputs)
    print("  PASSED\n")


def test_step_response_is_smoothed():
    """A sudden step should not fully reach the target in one sample."""
    f = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=0.0)

    out = f(t=0.01, x=10.0)

    print("=== Step Smoothing ===")
    print(f"  filtered step (target=10): {out:.3f}")
    assert 0.0 < out < 10.0
    print("  PASSED\n")


def test_non_positive_dt_returns_raw_value():
    """If the elapsed time is zero or negative, the raw value is returned."""
    f = OneEuroFilter(t0=1.0, x0=0.0)

    same_time = f(t=1.0, x=7.5)
    earlier = f(t=0.5, x=9.25)

    print("=== Non-positive dt ===")
    print(f"  same-time output: {same_time}, earlier-time output: {earlier}")
    assert same_time == 7.5
    assert earlier == 9.25
    print("  PASSED\n")


def test_alpha_formula_matches_reference():
    """Internal _alpha should match the 1 / (1 + tau/te) definition."""
    f = OneEuroFilter(t0=0.0, x0=0.0)

    te = 0.02
    cutoff = 2.0
    expected = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / te)

    print("=== Alpha Formula ===")
    print(f"  expected={expected:.6f}, actual={f._alpha(te, cutoff):.6f}")
    assert abs(f._alpha(te, cutoff) - expected) < 1e-9
    print("  PASSED\n")


def test_high_beta_tracks_fast_movement():
    """A higher beta should produce less lag on a fast-moving signal."""
    low_beta = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=0.0)
    high_beta = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=50.0)

    for i in range(1, 20):
        t = i * 0.01
        x = float(i)
        low_out = low_beta(t, x)
        high_out = high_beta(t, x)

    print("=== Adaptive Tracking ===")
    print(f"  low-beta final: {low_out:.3f}, high-beta final: {high_out:.3f}, target: 19.0")
    assert high_out > low_out
    print("  PASSED\n")


if __name__ == "__main__":
    test_constant_signal_is_preserved()
    test_step_response_is_smoothed()
    test_non_positive_dt_returns_raw_value()
    test_alpha_formula_matches_reference()
    test_high_beta_tracks_fast_movement()
    print("All tests passed!")
