"""
Unit tests for mask-based comparison metrics.

Covers the Gaussian distance-transform scoring helpers, the shape
comparison over synthetic binary blobs, and the optical-flow based
frame-energy helper on small synthetic masks.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests
from config_for_tests import make_masks

from model.comparison.mask_metrics import MaskMetrics


def _disk_mask(h: int, w: int, cy: int, cx: int, r: int) -> np.ndarray:
    """Create a single uint8 circular mask (0 / 255)."""
    yy, xx = np.mgrid[0:h, 0:w]
    disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2
    return disk.astype(np.uint8) * 255


def test_dtm_one_direction_identical_shapes():
    """Identical binary masks should score 1.0."""
    binary = _disk_mask(64, 64, 32, 32, 15) > 0

    score = MaskMetrics._dtm_one_direction(binary, binary, sigma=5.0)

    print("=== DTM Identical ===")
    print(f"  score: {score:.4f}")
    assert abs(score - 1.0) < 1e-6
    print("  PASSED\n")


def test_dtm_one_direction_disjoint_shapes():
    """Far-apart binary masks should score near zero."""
    h, w = 64, 64
    reference = _disk_mask(h, w, 10, 10, 5) > 0
    query = _disk_mask(h, w, 55, 55, 5) > 0

    score = MaskMetrics._dtm_one_direction(reference, query, sigma=2.0)

    print("=== DTM Disjoint ===")
    print(f"  score: {score:.6f}")
    assert score < 0.01
    print("  PASSED\n")


def test_compare_shapes_dtm_identical_masks():
    """A stack of identical masks should score ~1.0 overall."""
    masks = make_masks(num_frames=5, h=64, w=64, radius=15)

    mean_score, per_frame = MaskMetrics._compare_shapes_dtm(
        masks,
        masks,
        sigma=5.0,
        n_harmonics=5,
        n_points=80,
        threshold=128,
    )

    print("=== Compare Shapes Identical ===")
    print(f"  mean_score: {mean_score}, per_frame min: {per_frame.min():.4f}")
    assert mean_score >= 0.98
    print("  PASSED\n")


def test_compare_shapes_dtm_both_empty_frames():
    """Frames where both masks are empty should score 1.0 and be inactive."""
    empty = np.zeros((3, 64, 64), dtype=np.uint8)

    mean_score, per_frame = MaskMetrics._compare_shapes_dtm(
        empty,
        empty,
        sigma=5.0,
        n_harmonics=5,
        n_points=80,
        threshold=128,
    )

    print("=== Compare Shapes Both Empty ===")
    print(f"  mean_score: {mean_score}, per_frame: {per_frame.tolist()}")
    assert mean_score == 1.0
    assert np.allclose(per_frame, 1.0)
    print("  PASSED\n")


def test_frame_energy_no_motion():
    """Two identical masks should produce ~zero optical flow energy."""
    mask = _disk_mask(64, 64, 32, 32, 12)

    energy = MaskMetrics._compute_frame_energy(mask, mask, winsize=15, threshold=128)

    print("=== Frame Energy Static ===")
    print(f"  energy: {energy:.4f}")
    assert energy >= 0.0
    assert energy < 0.5
    print("  PASSED\n")


def test_frame_energy_shifted_mask():
    """A clearly shifted mask should produce higher energy than a static one."""
    prev = _disk_mask(96, 96, 48, 40, 14)
    curr = _disk_mask(96, 96, 48, 52, 14)

    static_energy = MaskMetrics._compute_frame_energy(prev, prev, winsize=15, threshold=128)
    shift_energy = MaskMetrics._compute_frame_energy(prev, curr, winsize=15, threshold=128)

    print("=== Frame Energy Shifted ===")
    print(f"  static: {static_energy:.4f}, shifted: {shift_energy:.4f}")
    assert shift_energy > static_energy
    print("  PASSED\n")


def test_compare_mask_energy_single_frame_fallback():
    """With fewer than 2 frames the helper should return a perfect fallback."""
    masks = make_masks(num_frames=1, h=32, w=32, radius=8)

    result = MaskMetrics._compare_mask_energy(masks, masks, winsize=15, threshold=128)

    print("=== Mask Energy Single Frame ===")
    print(f"  result: {result}")
    assert result["energy_score"] == 1.0
    assert result["per_frame_ratios"].size == 0
    print("  PASSED\n")


if __name__ == "__main__":
    test_dtm_one_direction_identical_shapes()
    test_dtm_one_direction_disjoint_shapes()
    test_compare_shapes_dtm_identical_masks()
    test_compare_shapes_dtm_both_empty_frames()
    test_frame_energy_no_motion()
    test_frame_energy_shifted_mask()
    test_compare_mask_energy_single_frame_fallback()
    print("All tests passed!")
