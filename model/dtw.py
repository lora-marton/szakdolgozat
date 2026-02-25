"""
Dynamic Time Warping for temporal alignment of dance sequences.

Aligns student frames to teacher frames using skeleton data,
producing a mapping that other metrics use for frame-to-frame comparison.
"""
import numpy as np
from dtw import dtw


def align_sequences(teacher_landmarks, student_landmarks, dtw_joints, window_size=120):
    """
    Align two landmark sequences using DTW on selected joints.

    Uses Sakoe-Chiba banding to prevent cross-matching of repeated moves
    and the symmetric2 step pattern for fair weighting.

    Args:
        teacher_landmarks: Array of shape (T, 33, 4) — teacher's raw landmarks.
        student_landmarks: Array of shape (S, 33, 4) — student's raw landmarks.
        dtw_joints: Tuple of joint indices to use for alignment (e.g., 12 main joints).
        window_size: Sakoe-Chiba band width in frames (default 120 = 2s at 60fps).

    Returns:
        alignment_path: List of (teacher_idx, student_idx) pairs.
        dtw_cost: Normalized alignment cost (lower = better timing match).
    """
    # Extract only the selected joints' (x, y, z) and flatten to 1D per frame
    teacher_flat = _flatten_joints(teacher_landmarks, dtw_joints)
    student_flat = _flatten_joints(student_landmarks, dtw_joints)

    # Run DTW with Sakoe-Chiba constraint
    alignment = dtw(
        student_flat,
        teacher_flat,
        step_pattern='symmetric2',
        window_type='sakoechiba',
        window_args={'window_size': window_size},
        keep_internals=False,
    )

    # Build alignment path as list of (teacher_idx, student_idx) pairs
    alignment_path = list(zip(alignment.index2.tolist(), alignment.index1.tolist()))

    # Normalized distance (total cost / path length) for comparability
    normalized_cost = alignment.normalizedDistance

    return alignment_path, normalized_cost


def _flatten_joints(landmarks, joint_indices):
    """
    Extract selected joints and flatten (x, y, z) into a 1D vector per frame.

    Args:
        landmarks: Array of shape (N, 33, 4) — [x, y, z, visibility].
        joint_indices: Tuple of joint indices to extract.

    Returns:
        Array of shape (N, len(joint_indices) * 3).
    """
    selected = landmarks[:, joint_indices, :3]  # (N, J, 3) — drop visibility
    return selected.reshape(selected.shape[0], -1)  # (N, J*3)
