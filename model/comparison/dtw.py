"""
Dynamic Time Warping for temporal alignment of dance sequences.

Aligns student frames to teacher frames using skeleton data,
producing a mapping that other metrics use for frame-to-frame comparison.
"""

import numpy as np
from dtw import dtw


class DTW:
    """Temporally aligns two dance sequences using Dynamic Time Warping."""

    @staticmethod
    def align_sequences(
        teacher_landmarks: np.ndarray,
        student_landmarks: np.ndarray,
        dtw_joints: tuple,
        window_size: int = 120,
    ) -> tuple[list, float]:
        """
        Align two landmark sequences using DTW on selected joints.

        Uses Sakoe-Chiba banding to prevent cross-matching of repeated moves
        and the symmetric2 step pattern for fair weighting.

        Args:
            teacher_landmarks: Array of shape (T, 33, 4) with teacher landmarks.
            student_landmarks: Array of shape (S, 33, 4) with student landmarks.
            dtw_joints: Tuple of joint indices to use for alignment.
            window_size: Sakoe-Chiba band width in frames.

        Returns:
            Tuple of (alignment_path, dtw_cost) where alignment_path is a list
            of (teacher_idx, student_idx) pairs and dtw_cost is the normalized
            alignment cost (lower means better timing match).
        """
        teacher_flat = DTW._flatten_joints(teacher_landmarks, dtw_joints)
        student_flat = DTW._flatten_joints(student_landmarks, dtw_joints)

        alignment = dtw(
            student_flat,
            teacher_flat,
            step_pattern="symmetric2",
            window_type="sakoechiba",
            window_args={"window_size": window_size},
            keep_internals=False,
        )

        alignment_path = list(zip(alignment.index2.tolist(), alignment.index1.tolist()))
        normalized_cost = alignment.normalizedDistance

        return alignment_path, normalized_cost

    @staticmethod
    def _flatten_joints(landmarks: np.ndarray, joint_indices: tuple) -> np.ndarray:
        """
        Extract selected joints and flatten (x, y, z) into a 1D vector per frame.

        Args:
            landmarks: Array of shape (N, 33, 4) with [x, y, z, visibility].
            joint_indices: Tuple of joint indices to extract.

        Returns:
            Array of shape (N, len(joint_indices) * 3).
        """
        selected = landmarks[:, joint_indices, :3]
        return selected.reshape(selected.shape[0], -1)
