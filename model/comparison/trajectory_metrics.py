"""
Trajectory comparison: direction and speed analysis of floor movement.
"""

import numpy as np


class TrajectoryMetrics:
    """Movement path comparison between two pose sequences."""

    @staticmethod
    def compute_trajectory_score(
        teacher_trajectory: np.ndarray,
        student_trajectory: np.ndarray,
        weight_direction: float,
        weight_speed: float,
        min_speed_threshold: float = 1e-4,
    ) -> dict:
        """Compute the combined trajectory comparison score.

        Compares velocity vectors between teacher and student using
        direction similarity (cosine) and speed similarity (min/max ratio),
        then combines them with configured weights.

        Args:
            teacher_trajectory: Array of shape (N, 2) with teacher hip positions.
            student_trajectory: Array of shape (N, 2) with student hip positions (DTW-aligned).
            weight_direction: Weight for the direction sub-score.
            weight_speed: Weight for the speed sub-score.
            min_speed_threshold: Minimum velocity magnitude to count as moving.

        Returns:
            Dict with 'score' (0-100) and 'direction_similarity' (0-1).
        """
        teacher_vel = np.diff(teacher_trajectory, axis=0)
        student_vel = np.diff(student_trajectory, axis=0)

        teacher_speed = np.linalg.norm(teacher_vel, axis=-1)
        student_speed = np.linalg.norm(student_vel, axis=-1)

        active = (teacher_speed > min_speed_threshold) | (student_speed > min_speed_threshold)

        if not active.any():
            return {"score": 100.0, "direction_similarity": 1.0}

        direction_scores = TrajectoryMetrics._direction_similarity(
            teacher_vel[active],
            student_vel[active],
            teacher_speed[active],
            student_speed[active],
        )
        mean_direction = float(direction_scores.mean())

        speed_scores = TrajectoryMetrics._speed_similarity(
            teacher_speed[active],
            student_speed[active],
        )
        mean_speed = float(speed_scores.mean())

        score = (weight_direction * mean_direction + weight_speed * mean_speed) * 100.0

        return {
            "score": round(score, 1),
            "direction_similarity": round(mean_direction, 3),
        }

    @staticmethod
    def _direction_similarity(
        teacher_vel: np.ndarray,
        student_vel: np.ndarray,
        teacher_speed: np.ndarray,
        student_speed: np.ndarray,
    ) -> np.ndarray:
        """Compute per-frame cosine similarity between velocity vectors.

        Returns values in [0, 1]: 1 means same direction, 0 means opposite.
        When one dancer is stationary, similarity is 0.

        Args:
            teacher_vel: Array of shape (M, 2) with teacher velocity vectors.
            student_vel: Array of shape (M, 2) with student velocity vectors.
            teacher_speed: Array of shape (M,) with teacher speed magnitudes.
            student_speed: Array of shape (M,) with student speed magnitudes.

        Returns:
            Array of shape (M,) with similarity scores in [0, 1].
        """
        dot = (teacher_vel * student_vel).sum(axis=-1)
        denom = teacher_speed * student_speed

        both_moving = denom > 0
        cosine = np.zeros_like(dot)
        cosine[both_moving] = dot[both_moving] / denom[both_moving]

        similarity = (cosine + 1.0) / 2.0

        return similarity

    @staticmethod
    def _speed_similarity(
        teacher_speed: np.ndarray,
        student_speed: np.ndarray,
    ) -> np.ndarray:
        """Compute per-frame speed ratio as min/max of the two speeds.

        Returns values in [0, 1]: 1 means same speed, 0 means one stationary.

        Args:
            teacher_speed: Array of shape (M,) with teacher speed magnitudes.
            student_speed: Array of shape (M,) with student speed magnitudes.

        Returns:
            Array of shape (M,) with speed ratio scores in [0, 1].
        """
        max_speed = np.maximum(teacher_speed, student_speed)
        min_speed = np.minimum(teacher_speed, student_speed)

        ratio = np.zeros_like(max_speed)
        nonzero = max_speed > 0
        ratio[nonzero] = min_speed[nonzero] / max_speed[nonzero]

        return ratio
