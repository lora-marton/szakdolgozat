"""
Dance comparison orchestrator.

Loads teacher and student HDF5 data, runs all comparison metrics,
and produces a structured result with scores and feedback.
"""
import os
import numpy as np
import h5py

from model.config import DEFAULT_COMPARISON_CONFIG
from model.preprocessor import preprocess
from model.dtw import align_sequences
from model.skeleton_metrics import compute_joint_angles, compute_cog, compare_angles
from model.mask_metrics import compute_iou
from model.trajectory_metrics import compare_trajectories


def compare_dances(output_dir, teacher_video=None, student_video=None, config=None):
    """
    Compare teacher and student dance data from a session directory.

    Args:
        output_dir: Path to session directory containing teacher_*.h5 and student_*.h5 files.
        teacher_video: Path to the teacher video file (needed for audio sync).
        student_video: Path to the student video file (needed for audio sync).
        config: ComparisonConfig instance (uses DEFAULT_COMPARISON_CONFIG if None).

    Returns:
        results: Dict with scores and feedback:
            {
                'overall_score': float (0-100),
                'skeleton_score': float (0-100),
                'trajectory_score': float (0-100),
                'mask_score': float (0-100),
                'timing_cost': float,
                'alignment_path': list,
                'feedback': list of str,
            }
    """
    if config is None:
        config = DEFAULT_COMPARISON_CONFIG

    # --- Load data ---
    teacher_data = _load_session_data(output_dir, 'teacher')
    student_data = _load_session_data(output_dir, 'student')

    # --- Phase 0: Preprocessing (sync + trim) ---
    if teacher_video and student_video:
        teacher_data, student_data = preprocess(
            teacher_data, student_data,
            teacher_video, student_video,
        )
    else:
        print("[Comparator] Video paths not provided — skipping audio sync & trimming.")

    # --- Phase A: Temporal Alignment (DTW) ---
    alignment_path, timing_cost = align_sequences(
        teacher_data['landmarks'],
        student_data['landmarks'],
        config.dtw_joints,
    )

    # Apply alignment: reindex student data to match teacher frames
    teacher_idx = [pair[0] for pair in alignment_path]
    student_idx = [pair[1] for pair in alignment_path]

    aligned_teacher_lm = teacher_data['landmarks'][teacher_idx]
    aligned_student_lm = student_data['landmarks'][student_idx]
    aligned_teacher_masks = teacher_data['masks'][teacher_idx]
    aligned_student_masks = student_data['masks'][student_idx]
    aligned_teacher_traj = teacher_data['trajectory'][teacher_idx]
    aligned_student_traj = student_data['trajectory'][student_idx]

    # --- Phase B: Skeleton Comparison ---
    teacher_angles = compute_joint_angles(aligned_teacher_lm, config.joint_angles)
    student_angles = compute_joint_angles(aligned_student_lm, config.joint_angles)
    skeleton_score, per_joint_scores, worst_frames = compare_angles(
        teacher_angles, student_angles, config.joint_tolerances,
    )

    # --- Phase C: Mask Comparison ---
    mask_score, per_frame_iou = compute_iou(aligned_teacher_masks, aligned_student_masks)
    mask_score_pct = mask_score * 100  # Convert 0-1 to 0-100

    # --- Phase D: Trajectory Comparison ---
    trajectory_score, direction_similarity = compare_trajectories(
        aligned_teacher_traj, aligned_student_traj,
    )

    # --- Weighted Final Score ---
    overall_score = (
        config.weight_skeleton * skeleton_score
        + config.weight_trajectory * trajectory_score
        + config.weight_mask * mask_score_pct
    )

    # --- Generate Feedback ---
    feedback = _generate_feedback(worst_frames, direction_similarity, mask_score)

    return {
        'overall_score': round(overall_score, 1),
        'skeleton_score': round(skeleton_score, 1),
        'trajectory_score': round(trajectory_score, 1),
        'mask_score': round(mask_score_pct, 1),
        'timing_cost': round(timing_cost, 3),
        'alignment_path': alignment_path,
        'per_joint_scores': per_joint_scores,
        'feedback': feedback,
    }


def _load_session_data(output_dir, label):
    """Load landmarks, masks, and trajectory from a session's HDF5 files."""
    data_path = os.path.join(output_dir, f'{label}_data.h5')
    mask_path = os.path.join(output_dir, f'{label}_masks.h5')

    with h5py.File(data_path, 'r') as f:
        landmarks = f['raw'][:]
        trajectory = f['trajectory'][:]
        fps = f.attrs.get('fps', 60.0)
        fixed_scale = f.attrs.get('fixed_scale', 1.0)

    with h5py.File(mask_path, 'r') as f:
        masks = f['masks'][:]

    return {
        'landmarks': landmarks,
        'trajectory': trajectory,
        'masks': masks,
        'fps': fps,
        'fixed_scale': fixed_scale,
    }


def _generate_feedback(worst_frames, direction_similarity, mask_iou):
    """Generate human-readable feedback from comparison results."""
    feedback = []

    # TODO: Implement rule-based feedback generation
    # Based on worst_frames, direction_similarity, mask_iou
    feedback.append("Comparison complete. Detailed feedback coming soon.")

    return feedback
