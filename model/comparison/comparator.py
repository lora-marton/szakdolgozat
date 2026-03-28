"""
Dance comparison orchestrator.

Runs all comparison metrics on preprocessed teacher and student data
and produces a structured result with scores and per-frame details.
"""
import numpy as np

from model.config import DEFAULT_COMPARISON_CONFIG
from model.comparison.dtw import align_sequences
from model.comparison.skeleton_metrics import compute_joint_angles, compute_cog, compare_angles, compare_cog
from model.comparison.mask_metrics import compute_mask_score
from model.comparison.trajectory_metrics import compare_trajectories


def compare_dances(teacher_data, student_data, config=None):
    """
    Compare preprocessed teacher and student dance data.

    Args:
        teacher_data: Dict with 'landmarks', 'masks', 'trajectory' arrays + 'fps'.
        student_data: Same structure as teacher_data.
        config: ComparisonConfig instance (uses DEFAULT_COMPARISON_CONFIG if None).

    Returns:
        Dict with scores and per-frame details:
            {
                'overall_score': float (0-100),
                'skeleton_score': float (0-100),
                'trajectory_score': float (0-100),
                'mask_score': float (0-100),
                'timing_cost': float,
                'alignment_path': list,
                'per_joint_scores': dict,
                'worst_frames': list,
            }
    """
    if config is None:
        config = DEFAULT_COMPARISON_CONFIG

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
    angle_score, per_joint_scores, worst_frames = compare_angles(
        teacher_angles, student_angles, config.joint_tolerances, config.angle_sigma,
    )

    teacher_cog = compute_cog(aligned_teacher_lm, config.cog_weights)
    student_cog = compute_cog(aligned_student_lm, config.cog_weights)
    cog_score = compare_cog(teacher_cog, student_cog, config.cog_sigma)

    skeleton_score = config.weight_angles * angle_score + config.weight_cog * cog_score

    # --- Phase C: Mask Comparison ---
    mask_result = compute_mask_score(aligned_teacher_masks, aligned_student_masks, config)
    mask_score_pct = mask_result['score']

    # --- Phase D: Trajectory Comparison ---
    trajectory_score, direction_similarity = compare_trajectories(
        aligned_teacher_traj, aligned_student_traj,
        weight_direction=config.trajectory_weight_direction,
        weight_speed=config.trajectory_weight_speed,
    )

    # --- Weighted Final Score ---
    overall_score = (
        config.weight_skeleton * skeleton_score
        + config.weight_trajectory * trajectory_score
        + config.weight_mask * mask_score_pct
    )

    return {
        'overall_score': round(overall_score, 1),
        'skeleton_score': round(skeleton_score, 1),
        'trajectory_score': round(trajectory_score, 1),
        'mask_score': round(mask_score_pct, 1),
        'timing_cost': round(timing_cost, 3),
        'alignment_path': alignment_path,
        'per_joint_scores': per_joint_scores,
        'worst_frames': worst_frames,
        'per_frame_shape': mask_result['per_frame_shape'],
        'energy_details': mask_result['energy_details'],
        'teacher_fps': teacher_data['fps'],
        'student_fps': student_data['fps'],
    }
