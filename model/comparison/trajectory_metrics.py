"""
Trajectory comparison: direction and speed analysis of floor movement.

Compares the movement patterns of teacher and student by analysing
velocity vectors — direction via cosine similarity and speed via ratio.
Stationary frames (where both dancers are nearly still) are filtered out
to avoid noisy comparisons.
"""
import numpy as np


def compare_trajectories(teacher_trajectory, student_trajectory,
                         weight_direction=0.75, weight_speed=0.25,
                         min_speed_threshold=1e-4):
    """
    Compare floor movement paths between teacher and student.

    Combines direction similarity (cosine of velocity vectors) and
    speed similarity (ratio of velocity magnitudes) into a single score.
    Frames where both dancers are nearly stationary are excluded from scoring.

    Args:
        teacher_trajectory: Array of shape (N, 2) — teacher hip positions.
        student_trajectory: Array of shape (N, 2) — student hip positions (DTW-aligned).
        weight_direction: Weight for direction sub-score (default 0.75).
        weight_speed: Weight for speed sub-score (default 0.25).
        min_speed_threshold: Minimum velocity magnitude to count as moving.
            Frames where BOTH dancers are below this are skipped.

    Returns:
        score: Float 0-100 — trajectory similarity score.
        direction_similarity: Float 0-1 — average cosine similarity of velocity vectors.
    """
    # Compute velocity vectors (frame-to-frame displacement)
    teacher_vel = np.diff(teacher_trajectory, axis=0)  # (N-1, 2)
    student_vel = np.diff(student_trajectory, axis=0)  # (N-1, 2)

    teacher_speed = np.linalg.norm(teacher_vel, axis=-1)  # (N-1,)
    student_speed = np.linalg.norm(student_vel, axis=-1)  # (N-1,)

    # Identify active frames (at least one dancer is moving)
    active_mask = (teacher_speed > min_speed_threshold) | (student_speed > min_speed_threshold)

    if not active_mask.any():
        # Both dancers are stationary the entire time — perfect "match"
        return 100.0, 1.0

    # --- Direction similarity (cosine) on active frames ---
    direction_scores = _direction_similarity(
        teacher_vel[active_mask], student_vel[active_mask],
        teacher_speed[active_mask], student_speed[active_mask],
    )
    mean_direction = float(direction_scores.mean())

    # --- Speed similarity (ratio) on active frames ---
    speed_scores = _speed_similarity(
        teacher_speed[active_mask], student_speed[active_mask],
    )
    mean_speed = float(speed_scores.mean())

    # --- Combined score ---
    score = (weight_direction * mean_direction + weight_speed * mean_speed) * 100.0

    return round(score, 1), round(mean_direction, 3)


# ── Helpers ──────────────────────────────────────────────────────────────


def _direction_similarity(teacher_vel, student_vel, teacher_speed, student_speed):
    """
    Compute per-frame cosine similarity between velocity vectors.

    Returns values in [0, 1]:  1 = same direction, 0 = opposite.
    When one dancer is stationary, similarity is 0 (they should be moving).
    """
    # Dot product per frame
    dot = (teacher_vel * student_vel).sum(axis=-1)  # (M,)
    denom = teacher_speed * student_speed

    # Where one is stationary: direction score = 0
    both_moving = denom > 0
    cosine = np.zeros_like(dot)
    cosine[both_moving] = dot[both_moving] / denom[both_moving]

    # Map from [-1, 1] → [0, 1]:  same direction=1, opposite=0
    similarity = (cosine + 1.0) / 2.0

    return similarity


def _speed_similarity(teacher_speed, student_speed):
    """
    Compute per-frame speed ratio: min/max of the two speeds.

    Returns values in [0, 1]:  1 = same speed, 0 = one stationary.
    """
    max_speed = np.maximum(teacher_speed, student_speed)
    min_speed = np.minimum(teacher_speed, student_speed)

    ratio = np.zeros_like(max_speed)
    nonzero = max_speed > 0
    ratio[nonzero] = min_speed[nonzero] / max_speed[nonzero]

    return ratio
