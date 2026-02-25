"""
Skeleton-based comparison metrics: joint angles and center of gravity.
"""
import numpy as np


def compute_joint_angles(landmarks, angle_definitions):
    """
    Calculate joint angles for a sequence of frames.

    Args:
        landmarks: Array of shape (N, 33, 4) — landmark data.
        angle_definitions: Tuple of (parent, joint, child) index triplets.

    Returns:
        angles: Array of shape (N, len(angle_definitions)) — angles in degrees.
    """
    # TODO: Implement joint angle calculation using dot product
    raise NotImplementedError("Joint angle computation not yet implemented")


def compute_cog(landmarks, cog_weights):
    """
    Calculate the weighted center of gravity for each frame.

    Args:
        landmarks: Array of shape (N, 33, 4) — landmark data.
        cog_weights: Dict mapping joint index → body segment weight.

    Returns:
        cog: Array of shape (N, 2) — center of gravity (x, y) per frame.
    """
    # TODO: Implement weighted CoG calculation
    raise NotImplementedError("CoG computation not yet implemented")


def compare_angles(teacher_angles, student_angles, tolerances):
    """
    Score the similarity of joint angles between aligned teacher and student frames.

    Args:
        teacher_angles: Array of shape (N, J) — teacher joint angles in degrees.
        student_angles: Array of shape (N, J) — student joint angles (DTW-aligned).
        tolerances: Dict of joint name → tolerance in degrees.

    Returns:
        score: Float 0-100 — overall skeleton accuracy score.
        per_joint_scores: Dict of joint name → score.
        worst_frames: List of (frame_idx, joint_name, error_degrees) for feedback.
    """
    # TODO: Implement angle comparison with tolerance buffers
    raise NotImplementedError("Angle comparison not yet implemented")
