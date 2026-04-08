"""
Unit tests for the top-level Comparator pipeline.

Drives the full compare_dances method with synthetic teacher/student
data covering identical, distorted, and mismatched-trajectory cases.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests
from config_for_tests import make_masks, make_stick_figure_landmarks, make_trajectory

from model.comparison.comparator import Comparator

EXPECTED_KEYS = {
    "overall_score",
    "skeleton_score",
    "trajectory_score",
    "mask_score",
    "timing_cost",
    "alignment_path",
    "per_joint_scores",
    "worst_frames",
    "per_frame_shape",
    "energy_details",
    "direction_similarity",
    "teacher_fps",
    "student_fps",
}


def _make_dataset(num_frames: int = 20) -> dict:
    """Build a teacher/student data dict with matching dummy arrays."""
    return {
        "landmarks": make_stick_figure_landmarks(num_frames=num_frames),
        "masks": make_masks(num_frames=num_frames, h=64, w=64, radius=18),
        "trajectory": make_trajectory(num_frames=num_frames),
        "fps": 30.0,
    }


def test_compare_returns_all_expected_keys():
    """The result dict should contain all documented fields."""
    teacher = _make_dataset()
    student = _make_dataset()

    result = Comparator.compare_dances(teacher, student)

    print("=== Compare Keys ===")
    print(f"  keys: {sorted(result.keys())}")
    assert EXPECTED_KEYS.issubset(result.keys())
    print("  PASSED\n")


def test_compare_identical_near_perfect_score():
    """Identical teacher and student data should score nearly 100."""
    teacher = _make_dataset()
    student = _make_dataset()

    result = Comparator.compare_dances(teacher, student)

    print("=== Compare Identical ===")
    print(
        f"  overall: {result['overall_score']}, "
        f"skeleton: {result['skeleton_score']}, "
        f"trajectory: {result['trajectory_score']}, "
        f"mask: {result['mask_score']}"
    )
    assert result["overall_score"] >= 95.0
    assert result["skeleton_score"] >= 99.0
    assert result["trajectory_score"] >= 99.0
    print("  PASSED\n")


def test_compare_distorted_student_drops_skeleton_score():
    """Distorting student landmarks should lower the skeleton score."""
    teacher = _make_dataset()
    student = _make_dataset()
    student["landmarks"][:, 13, 0] -= 0.20
    student["landmarks"][:, 13, 1] -= 0.20
    student["landmarks"][:, 15, 0] -= 0.20
    student["landmarks"][:, 15, 1] -= 0.20

    result = Comparator.compare_dances(teacher, student)

    print("=== Compare Distorted ===")
    print(f"  skeleton: {result['skeleton_score']}, overall: {result['overall_score']}")
    assert result["skeleton_score"] < 95.0
    print("  PASSED\n")


def test_compare_mirrored_trajectory_drops_trajectory_score():
    """A mirrored trajectory should tank the trajectory component."""
    teacher = _make_dataset()
    student = _make_dataset()
    student["trajectory"] = make_trajectory(
        num_frames=student["trajectory"].shape[0],
        velocity=(-1.0, -0.5),
    )

    result = Comparator.compare_dances(teacher, student)

    print("=== Compare Mirrored Trajectory ===")
    print(f"  trajectory: {result['trajectory_score']}, direction: {result['direction_similarity']}")
    assert result["trajectory_score"] < 30.0
    assert result["direction_similarity"] < 0.2
    print("  PASSED\n")


def test_compare_preserves_fps_metadata():
    """Teacher and student FPS should be propagated into the result."""
    teacher = _make_dataset()
    student = _make_dataset()
    teacher["fps"] = 24.0
    student["fps"] = 48.0

    result = Comparator.compare_dances(teacher, student)

    print("=== Compare FPS ===")
    print(f"  teacher_fps: {result['teacher_fps']}, student_fps: {result['student_fps']}")
    assert result["teacher_fps"] == 24.0
    assert result["student_fps"] == 48.0
    print("  PASSED\n")


if __name__ == "__main__":
    test_compare_returns_all_expected_keys()
    test_compare_identical_near_perfect_score()
    test_compare_distorted_student_drops_skeleton_score()
    test_compare_mirrored_trajectory_drops_trajectory_score()
    test_compare_preserves_fps_metadata()
    print("All tests passed!")
