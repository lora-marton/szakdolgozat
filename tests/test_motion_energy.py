"""
Unit tests for motion energy computation and active-range detection.
"""

import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests
from config_for_tests import make_stick_figure_landmarks

from model.preprocessing.motion_energy import MotionEnergy

logger = logging.getLogger(__name__)


def test_compute_motion_energy_static_pose():
    """A perfectly static pose should have zero motion energy everywhere."""
    lm = make_stick_figure_landmarks(num_frames=10)

    energy = MotionEnergy.compute_motion_energy(lm)

    logger.info("=== Motion Energy Static ===")
    logger.info("  shape: %s, max: %.6f", energy.shape, energy.max())
    assert energy.shape == (9,)
    assert np.allclose(energy, 0.0)
    logger.info("  PASSED\n")


def test_compute_motion_energy_moving_pose():
    """A pose that is translated each frame should yield a positive constant energy."""
    lm = make_stick_figure_landmarks(num_frames=6)
    for f in range(6):
        lm[f, :, 0] += f * 0.01

    energy = MotionEnergy.compute_motion_energy(lm)

    logger.info("=== Motion Energy Moving ===")
    logger.info("  energy: %s", energy.tolist())
    assert np.all(energy > 0.0)
    assert np.allclose(energy, energy[0], atol=1e-6)
    logger.info("  PASSED\n")


def test_find_active_range_single_burst():
    """A single sustained burst in the middle should be the detected range."""
    energy = np.zeros(60, dtype=np.float32)
    energy[20:40] = 1.0

    start, end = MotionEnergy.find_active_range(
        energy,
        threshold_ratio=0.15,
        min_duration_frames=5,
        active_window_ratio=0.7,
    )

    logger.info("=== Active Range Single Burst ===")
    logger.info("  start: %d, end: %d", start, end)
    assert 18 <= start <= 22
    assert 38 <= end <= 42
    logger.info("  PASSED\n")


def test_find_active_range_noise_below_threshold():
    """Small noise that never reaches the threshold should not expand the range."""
    rng = np.random.default_rng(0)
    energy = rng.uniform(0.0, 0.05, size=50).astype(np.float32)
    energy[20:35] = 1.0

    start, end = MotionEnergy.find_active_range(
        energy,
        threshold_ratio=0.5,
        min_duration_frames=5,
        active_window_ratio=0.7,
    )

    logger.info("=== Active Range Noise ===")
    logger.info("  start: %d, end: %d", start, end)
    assert start >= 18
    assert end <= 37
    logger.info("  PASSED\n")


def test_find_active_range_short_burst_rejected():
    """A burst shorter than min_duration_frames should be ignored."""
    energy = np.zeros(40, dtype=np.float32)
    energy[5:7] = 1.0
    energy[20:32] = 1.0

    start, end = MotionEnergy.find_active_range(
        energy,
        threshold_ratio=0.15,
        min_duration_frames=5,
        active_window_ratio=0.7,
    )

    logger.info("=== Active Range Short Burst ===")
    logger.info("  start: %d, end: %d", start, end)
    assert start >= 18
    logger.info("  PASSED\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    test_compute_motion_energy_static_pose()
    test_compute_motion_energy_moving_pose()
    test_find_active_range_single_burst()
    test_find_active_range_noise_below_threshold()
    test_find_active_range_short_burst_rejected()
    logger.info("All tests passed!")
