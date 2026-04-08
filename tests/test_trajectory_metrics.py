"""
Unit tests for trajectory-based comparison metrics.

Covers direction similarity, speed similarity, and the combined
trajectory score under several scenarios.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests  # noqa: F401

from model.comparison.trajectory_metrics import TrajectoryMetrics


def test_direction_similarity_same_direction():
    """Parallel velocity vectors should score 1.0."""
    v = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    speed = np.linalg.norm(v, axis=-1)

    sim = TrajectoryMetrics._direction_similarity(v, v, speed, speed)

    print("=== Direction Same ===")
    print(f"  sim: {sim.tolist()}")
    assert np.allclose(sim, 1.0)
    print("  PASSED\n")


def test_direction_similarity_opposite_direction():
    """Anti-parallel velocity vectors should score 0.0."""
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = -a

    sim = TrajectoryMetrics._direction_similarity(
        a,
        b,
        np.linalg.norm(a, axis=-1),
        np.linalg.norm(b, axis=-1),
    )

    print("=== Direction Opposite ===")
    print(f"  sim: {sim.tolist()}")
    assert abs(sim[0] - 0.0) < 1e-6
    print("  PASSED\n")


def test_direction_similarity_orthogonal():
    """Perpendicular velocity vectors should score 0.5."""
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)

    sim = TrajectoryMetrics._direction_similarity(
        a,
        b,
        np.linalg.norm(a, axis=-1),
        np.linalg.norm(b, axis=-1),
    )

    print("=== Direction Orthogonal ===")
    print(f"  sim: {sim.tolist()}")
    assert abs(sim[0] - 0.5) < 1e-6
    print("  PASSED\n")


def test_speed_similarity_equal_speeds():
    """Equal speeds should score 1.0."""
    speed = np.array([3.0, 5.0], dtype=np.float32)

    ratio = TrajectoryMetrics._speed_similarity(speed, speed)

    print("=== Speed Equal ===")
    print(f"  ratio: {ratio.tolist()}")
    assert np.allclose(ratio, 1.0)
    print("  PASSED\n")


def test_speed_similarity_doubled_speed():
    """One speed doubled should score 0.5."""
    t = np.array([2.0], dtype=np.float32)
    s = np.array([4.0], dtype=np.float32)

    ratio = TrajectoryMetrics._speed_similarity(t, s)

    print("=== Speed Doubled ===")
    print(f"  ratio: {ratio.tolist()}")
    assert abs(ratio[0] - 0.5) < 1e-6
    print("  PASSED\n")


def test_trajectory_score_identical_path():
    """Identical trajectories should produce a 100 score."""
    traj = np.cumsum(np.ones((10, 2), dtype=np.float32), axis=0)

    result = TrajectoryMetrics.compute_trajectory_score(
        traj,
        traj,
        weight_direction=0.75,
        weight_speed=0.25,
    )

    print("=== Trajectory Identical ===")
    print(f"  result: {result}")
    assert result["score"] == 100.0
    assert result["direction_similarity"] == 1.0
    print("  PASSED\n")


def test_trajectory_score_static_paths():
    """Two static trajectories (no active frames) fall back to a perfect score."""
    traj = np.zeros((5, 2), dtype=np.float32)

    result = TrajectoryMetrics.compute_trajectory_score(
        traj,
        traj,
        weight_direction=0.75,
        weight_speed=0.25,
    )

    print("=== Trajectory Static ===")
    print(f"  result: {result}")
    assert result["score"] == 100.0
    print("  PASSED\n")


def test_trajectory_score_mirrored_direction():
    """A mirrored trajectory should lower the direction component."""
    teacher = np.cumsum(np.ones((10, 2), dtype=np.float32), axis=0)
    student = np.cumsum(-np.ones((10, 2), dtype=np.float32), axis=0)

    result = TrajectoryMetrics.compute_trajectory_score(
        teacher,
        student,
        weight_direction=0.75,
        weight_speed=0.25,
    )

    print("=== Trajectory Mirrored ===")
    print(f"  result: {result}")
    assert result["direction_similarity"] < 0.1
    assert result["score"] < 30.0
    print("  PASSED\n")


if __name__ == "__main__":
    test_direction_similarity_same_direction()
    test_direction_similarity_opposite_direction()
    test_direction_similarity_orthogonal()
    test_speed_similarity_equal_speeds()
    test_speed_similarity_doubled_speed()
    test_trajectory_score_identical_path()
    test_trajectory_score_static_paths()
    test_trajectory_score_mirrored_direction()
    print("All tests passed!")
