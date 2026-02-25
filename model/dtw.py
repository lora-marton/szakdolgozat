"""
Dynamic Time Warping for temporal alignment of dance sequences.

Aligns student frames to teacher frames using skeleton data,
producing a mapping that other metrics use for frame-to-frame comparison.
"""
import numpy as np


def align_sequences(teacher_landmarks, student_landmarks, dtw_joints):
    """
    Align two landmark sequences using DTW on selected joints.

    Args:
        teacher_landmarks: Array of shape (T, 33, 4) — teacher's raw landmarks.
        student_landmarks: Array of shape (S, 33, 4) — student's raw landmarks.
        dtw_joints: Tuple of joint indices to use for alignment.

    Returns:
        alignment_path: List of (teacher_idx, student_idx) pairs.
        dtw_cost: Total alignment cost (lower = better timing match).
    """
    # TODO: Implement DTW alignment
    raise NotImplementedError("DTW alignment not yet implemented")
