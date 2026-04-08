"""
Unit tests for skeleton-based comparison metrics.

Covers joint-angle computation, center-of-gravity computation, the
per-joint and CoG scoring helpers, and the combined skeleton score.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests
from config_for_tests import make_stick_figure_landmarks

from model.comparison.skeleton_metrics import SkeletonMetrics
from model.config import DEFAULT_COMPARISON_CONFIG


def _landmarks_with_positions(positions: dict, num_frames: int = 1) -> np.ndarray:
    """Build a (num_frames, 33, 4) array from a sparse index->(x, y) mapping."""
    lm = np.zeros((num_frames, 33, 4), dtype=np.float32)
    lm[..., 3] = 1.0
    for idx, (x, y) in positions.items():
        lm[:, idx, 0] = x
        lm[:, idx, 1] = y
    return lm


def test_joint_angles_right_angle():
    """A perpendicular joint should measure 90 degrees."""
    lm = _landmarks_with_positions(
        {
            0: (0.0, 0.0),
            1: (0.0, 1.0),
            2: (1.0, 1.0),
        }
    )

    angles = SkeletonMetrics._compute_joint_angles(lm, ((0, 1, 2),))

    print("=== Right Angle ===")
    print(f"  angle: {angles[0, 0]:.2f}")
    assert abs(angles[0, 0] - 90.0) < 1e-3
    print("  PASSED\n")


def test_joint_angles_straight_line():
    """A straight (opposite-direction) joint should measure 180 degrees."""
    lm = _landmarks_with_positions(
        {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (2.0, 0.0),
        }
    )

    angles = SkeletonMetrics._compute_joint_angles(lm, ((0, 1, 2),))

    print("=== Straight Line ===")
    print(f"  angle: {angles[0, 0]:.2f}")
    assert abs(angles[0, 0] - 180.0) < 1e-3
    print("  PASSED\n")


def test_joint_angles_handle_degenerate_vector():
    """Zero-length vectors must not blow up (div-by-zero guard)."""
    lm = _landmarks_with_positions(
        {
            0: (0.5, 0.5),
            1: (0.5, 0.5),
            2: (1.0, 0.5),
        }
    )

    angles = SkeletonMetrics._compute_joint_angles(lm, ((0, 1, 2),))

    print("=== Degenerate Vector ===")
    print(f"  angle: {angles[0, 0]:.2f}")
    assert np.isfinite(angles[0, 0])
    print("  PASSED\n")


def test_cog_weighted_average():
    """CoG should match a hand-computed weighted average."""
    lm = _landmarks_with_positions(
        {
            11: (0.0, 0.0),
            12: (1.0, 0.0),
        }
    )
    weights = {11: 1.0, 12: 3.0}

    cog = SkeletonMetrics._compute_cog(lm, weights)

    print("=== CoG Weighted Average ===")
    print(f"  cog: {cog[0].tolist()}")
    assert abs(cog[0, 0] - 0.75) < 1e-6
    assert abs(cog[0, 1] - 0.0) < 1e-6
    print("  PASSED\n")


def test_compare_cog_perfect_match():
    """Identical CoG sequences should score 100."""
    cog = np.array([[0.5, 0.5], [0.6, 0.5]], dtype=np.float32)

    score = SkeletonMetrics._compare_cog(cog, cog, sigma=0.08)

    print("=== CoG Perfect Match ===")
    print(f"  score: {score}")
    assert score == 100.0
    print("  PASSED\n")


def test_compare_cog_large_distance_drops_score():
    """Very distant CoG should drop the score well below 100."""
    a = np.array([[0.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 1.0]], dtype=np.float32)

    score = SkeletonMetrics._compare_cog(a, b, sigma=0.08)

    print("=== CoG Far Apart ===")
    print(f"  score: {score}")
    assert score < 5.0
    print("  PASSED\n")


def test_skeleton_score_identical_inputs():
    """Feeding the same stick-figure sequence twice should score near-perfect."""
    lm = make_stick_figure_landmarks(num_frames=30)

    result = SkeletonMetrics.compute_skeleton_score(lm, lm, DEFAULT_COMPARISON_CONFIG)

    print("=== Skeleton Identical ===")
    print(f"  score: {result['score']}")
    print(f"  per_joint: {result['per_joint_scores']}")
    assert result["score"] >= 99.0
    assert all(v >= 99.0 for v in result["per_joint_scores"].values())
    print("  PASSED\n")


def test_skeleton_score_drops_with_distorted_student():
    """Distorting one side of the student should lower the overall score."""
    teacher = make_stick_figure_landmarks(num_frames=20)
    student = teacher.copy()
    student[:, 13, 0] -= 0.30
    student[:, 13, 1] -= 0.30
    student[:, 15, 0] -= 0.30
    student[:, 15, 1] -= 0.30

    result = SkeletonMetrics.compute_skeleton_score(teacher, student, DEFAULT_COMPARISON_CONFIG)

    print("=== Skeleton Distorted ===")
    print(f"  score: {result['score']}")
    assert result["score"] < 99.0
    print("  PASSED\n")


def test_skeleton_score_ignores_low_visibility():
    """When all landmarks have zero visibility, per-joint scores fall back to 100."""
    lm = make_stick_figure_landmarks(num_frames=10, visibility=0.0)

    result = SkeletonMetrics.compute_skeleton_score(lm, lm, DEFAULT_COMPARISON_CONFIG)

    print("=== Skeleton Zero Visibility ===")
    print(f"  score: {result['score']}, per_joint: {result['per_joint_scores']}")
    assert all(v == 100.0 for v in result["per_joint_scores"].values())
    print("  PASSED\n")


if __name__ == "__main__":
    test_joint_angles_right_angle()
    test_joint_angles_straight_line()
    test_joint_angles_handle_degenerate_vector()
    test_cog_weighted_average()
    test_compare_cog_perfect_match()
    test_compare_cog_large_distance_drops_score()
    test_skeleton_score_identical_inputs()
    test_skeleton_score_drops_with_distorted_student()
    test_skeleton_score_ignores_low_visibility()
    print("All tests passed!")
