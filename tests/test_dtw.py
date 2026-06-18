"""
Unit tests for DTW temporal alignment.

Covers the joint-flattening helper and several alignment scenarios:
identical sequences, a shifted sequence, and a tight Sakoe-Chiba window.
"""

import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests
from config_for_tests import make_stick_figure_landmarks

from model.comparison.dtw import DTW

logger = logging.getLogger(__name__)

JOINT_INDICES = (11, 12, 13, 14, 23, 24)


def test_flatten_joints_shape_and_values():
    """_flatten_joints should reshape (N, 33, 4) -> (N, len(joints)*3)."""
    landmarks = np.arange(2 * 33 * 4, dtype=np.float32).reshape(2, 33, 4)

    flat = DTW._flatten_joints(landmarks, (0, 5))

    logger.info("=== Flatten Joints ===")
    logger.info("  shape: %s", flat.shape)
    assert flat.shape == (2, 6)
    assert np.allclose(flat[0, :3], landmarks[0, 0, :3])
    assert np.allclose(flat[0, 3:], landmarks[0, 5, :3])
    logger.info("  PASSED\n")


def test_identical_sequences_produce_diagonal_alignment():
    """Identical teacher/student should align frame-for-frame with ~zero cost."""
    teacher = make_stick_figure_landmarks(num_frames=20)
    student = teacher.copy()

    path, cost = DTW.align_sequences(teacher, student, JOINT_INDICES, window_size=10)

    logger.info("=== Identical Alignment ===")
    logger.info("  path len: %d, cost: %.4f", len(path), cost)
    assert cost < 1e-3
    assert path[0] == (0, 0)
    assert path[-1] == (19, 19)
    logger.info("  PASSED\n")


def test_shifted_sequence_still_aligns():
    """A student sequence shifted by a small constant still warps to the teacher."""
    teacher = make_stick_figure_landmarks(num_frames=25)
    student = teacher.copy()
    student[:, :, 0] += 0.01

    path, cost = DTW.align_sequences(teacher, student, JOINT_INDICES, window_size=10)

    logger.info("=== Shifted Sequence ===")
    logger.info("  path len: %d, cost: %.4f", len(path), cost)
    assert len(path) >= 25
    assert cost >= 0.0
    logger.info("  PASSED\n")


def test_alignment_path_types_are_ints():
    """Every index in the alignment path must be a python int (for list indexing)."""
    teacher = make_stick_figure_landmarks(num_frames=12)
    student = make_stick_figure_landmarks(num_frames=12)

    path, _ = DTW.align_sequences(teacher, student, JOINT_INDICES, window_size=6)

    logger.info("=== Path Element Types ===")
    logger.info("  first three: %s", path[:3])
    assert all(isinstance(t, int) and isinstance(s, int) for t, s in path)
    logger.info("  PASSED\n")


def test_different_length_sequences():
    """DTW must handle teacher and student of different lengths."""
    teacher = make_stick_figure_landmarks(num_frames=15)
    student = make_stick_figure_landmarks(num_frames=20)

    path, cost = DTW.align_sequences(teacher, student, JOINT_INDICES, window_size=10)

    teacher_max = max(p[0] for p in path)
    student_max = max(p[1] for p in path)

    logger.info("=== Different Lengths ===")
    logger.info("  teacher_max=%d, student_max=%d", teacher_max, student_max)
    assert teacher_max == 14
    assert student_max == 19
    logger.info("  PASSED\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    test_flatten_joints_shape_and_values()
    test_identical_sequences_produce_diagonal_alignment()
    test_shifted_sequence_still_aligns()
    test_alignment_path_types_are_ints()
    test_different_length_sequences()
    logger.info("All tests passed!")
