"""
Unit tests for FrameProcessor.

Covers the pure-math helpers (hip center, torso length, affine matrix,
calibration) as well as the full process_frame pipeline using a fake
MediaPipe-like landmark object and a small synthetic mask.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests  # noqa: F401

from model.config import DEFAULT_EXTRACTION_CONFIG
from model.extraction.frame_processor import FrameProcessor


class FakeLandmark:
    """Minimal stand-in for a MediaPipe NormalizedLandmark."""

    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 0.95):
        """Store the four landmark fields."""
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _make_fake_landmarks() -> list:
    """Build a list of 33 fake normalized landmarks in stick-figure layout."""
    positions = {
        0: (0.50, 0.18),
        11: (0.42, 0.30),
        12: (0.58, 0.30),
        13: (0.38, 0.42),
        14: (0.62, 0.42),
        15: (0.34, 0.54),
        16: (0.66, 0.54),
        23: (0.44, 0.58),
        24: (0.56, 0.58),
        25: (0.44, 0.74),
        26: (0.56, 0.74),
        27: (0.44, 0.90),
        28: (0.56, 0.90),
    }
    return [FakeLandmark(*positions.get(i, (0.5, 0.5))) for i in range(33)]


def test_get_hip_center_averages_hip_landmarks():
    """_get_hip_center should average left and right hip positions."""
    fp = FrameProcessor(DEFAULT_EXTRACTION_CONFIG)
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[23] = [0.4, 0.6, 0.0, 1.0]
    lm[24] = [0.6, 0.6, 0.0, 1.0]

    center = fp._get_hip_center(lm, vid_w=100, vid_h=200)

    print("=== Hip Center ===")
    print(f"  center: {center.tolist()}")
    assert np.allclose(center, [50.0, 120.0])
    print("  PASSED\n")


def test_get_torso_length_distance():
    """_get_torso_length should equal the mid-shoulder to mid-hip distance."""
    fp = FrameProcessor(DEFAULT_EXTRACTION_CONFIG)
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[11] = [0.4, 0.3, 0.0, 1.0]
    lm[12] = [0.6, 0.3, 0.0, 1.0]
    lm[23] = [0.4, 0.6, 0.0, 1.0]
    lm[24] = [0.6, 0.6, 0.0, 1.0]

    length = fp._get_torso_length(lm, vid_w=100, vid_h=100)

    print("=== Torso Length ===")
    print(f"  length: {length}")
    assert abs(length - 30.0) < 1e-5
    print("  PASSED\n")


def test_calibrate_if_needed_sets_scale_once():
    """Calibration should run exactly once and ignore later frames."""
    fp = FrameProcessor(DEFAULT_EXTRACTION_CONFIG)
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[11] = [0.4, 0.3, 0.0, 1.0]
    lm[12] = [0.6, 0.3, 0.0, 1.0]
    lm[23] = [0.4, 0.6, 0.0, 1.0]
    lm[24] = [0.6, 0.6, 0.0, 1.0]

    fp._calibrate_if_needed(lm, vid_w=100, vid_h=100)
    first_scale = fp.fixed_scale

    lm[11] = [0.4, 0.1, 0.0, 1.0]
    fp._calibrate_if_needed(lm, vid_w=100, vid_h=100)

    print("=== Calibration ===")
    print(f"  first_scale: {first_scale}, after second call: {fp.fixed_scale}")
    assert fp.fixed_scale == first_scale
    assert abs(first_scale - (DEFAULT_EXTRACTION_CONFIG.target_torso_px / 30.0)) < 1e-5
    print("  PASSED\n")


def test_build_affine_matrix_shape_and_translation():
    """_build_affine should return a 2x3 matrix that centers the hip."""
    fp = FrameProcessor(DEFAULT_EXTRACTION_CONFIG)
    fp._fixed_scale = 2.0

    hip = np.array([50.0, 40.0], dtype=np.float32)
    affine = fp._build_affine(hip)

    cx, cy = DEFAULT_EXTRACTION_CONFIG.norm_center

    print("=== Affine Matrix ===")
    print(f"  matrix:\n{affine}")
    assert affine.shape == (2, 3)
    assert abs(affine[0, 0] - 2.0) < 1e-6
    assert abs(affine[1, 1] - 2.0) < 1e-6
    assert abs(affine[0, 2] - (cx - 50.0 * 2.0)) < 1e-5
    assert abs(affine[1, 2] - (cy - 40.0 * 2.0)) < 1e-5
    print("  PASSED\n")


def test_process_frame_end_to_end():
    """process_frame should return landmarks, normalized mask, and a 2D trajectory."""
    fp = FrameProcessor(DEFAULT_EXTRACTION_CONFIG)
    raw = _make_fake_landmarks()
    mask = np.ones((480, 640), dtype=np.float32)

    lm, norm_mask, traj = fp.process_frame(raw, mask, vid_w=640, vid_h=480, timestamp_ms=0.0)
    lm2, _, _ = fp.process_frame(raw, mask, vid_w=640, vid_h=480, timestamp_ms=16.0)

    print("=== Process Frame End To End ===")
    print(f"  lm shape: {lm.shape}, mask shape: {norm_mask.shape}, traj: {traj}")
    assert lm.shape == (33, 4)
    assert lm.dtype == np.float32
    assert norm_mask.shape == DEFAULT_EXTRACTION_CONFIG.target_mask_size
    assert norm_mask.dtype == np.uint8
    assert len(traj) == 2
    assert fp.fixed_scale is not None
    assert np.allclose(lm2[11, :2], lm[11, :2], atol=1e-3)
    print("  PASSED\n")


if __name__ == "__main__":
    test_get_hip_center_averages_hip_landmarks()
    test_get_torso_length_distance()
    test_calibrate_if_needed_sets_scale_once()
    test_build_affine_matrix_shape_and_translation()
    test_process_frame_end_to_end()
    print("All tests passed!")
