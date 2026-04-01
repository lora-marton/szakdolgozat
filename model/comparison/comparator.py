"""
Dance comparison orchestrator.

Runs all comparison metrics on preprocessed teacher and student data
and produces a structured result dict with scores and per-frame details.

Pipeline phases:
  1. Temporal alignment (DTW) — warp student frames to match teacher timing.
  2. Skeleton comparison — joint angles and center of gravity.
  3. Mask comparison — silhouette shape (DTM) and movement energy.
  4. Trajectory comparison — floor movement direction and speed.
  5. Weighted final score — combine all metric scores.
"""
from model.comparison.dtw import DTW
from model.comparison.mask_metrics import MaskMetrics
from model.comparison.skeleton_metrics import SkeletonMetrics
from model.comparison.trajectory_metrics import TrajectoryMetrics
from model.config import DEFAULT_COMPARISON_CONFIG


class Comparator:
    """Top-level dance comparison pipeline."""

    @staticmethod
    def compare_dances(teacher_data: dict, student_data: dict, config=None) -> dict:
        """Compare preprocessed teacher and student dance data.

        Args:
            teacher_data: Dict with 'landmarks', 'masks', 'trajectory' arrays and 'fps'.
            student_data: Same structure as teacher_data.
            config: ComparisonConfig instance (uses default if None).

        Returns:
            Dict with overall and per-metric scores, per-joint breakdowns,
            worst frames, per-frame shape data, energy details, direction
            similarity, and FPS info.
        """
        if config is None:
            config = DEFAULT_COMPARISON_CONFIG

        alignment_path, timing_cost = DTW.align_sequences(
            teacher_data['landmarks'],
            student_data['landmarks'],
            config.dtw_joints,
            config.dtw_window_size,
        )

        teacher_idx = [pair[0] for pair in alignment_path]
        student_idx = [pair[1] for pair in alignment_path]

        aligned_teacher_lm = teacher_data['landmarks'][teacher_idx]
        aligned_student_lm = student_data['landmarks'][student_idx]
        aligned_teacher_masks = teacher_data['masks'][teacher_idx]
        aligned_student_masks = student_data['masks'][student_idx]
        aligned_teacher_traj = teacher_data['trajectory'][teacher_idx]
        aligned_student_traj = student_data['trajectory'][student_idx]

        skeleton_result = SkeletonMetrics.compute_skeleton_score(
            aligned_teacher_lm, aligned_student_lm, config,
        )

        mask_result = MaskMetrics.compute_mask_score(
            aligned_teacher_masks, aligned_student_masks, config,
        )

        trajectory_result = TrajectoryMetrics.compute_trajectory_score(
            aligned_teacher_traj, aligned_student_traj,
            weight_direction=config.trajectory_weight_direction,
            weight_speed=config.trajectory_weight_speed,
        )

        overall_score = (
            config.weight_skeleton * skeleton_result['score']
            + config.weight_trajectory * trajectory_result['score']
            + config.weight_mask * mask_result['score']
        )

        return {
            'overall_score': round(overall_score, 1),
            'skeleton_score': skeleton_result['score'],
            'trajectory_score': trajectory_result['score'],
            'mask_score': mask_result['score'],
            'timing_cost': round(timing_cost, 3),
            'alignment_path': alignment_path,
            'per_joint_scores': skeleton_result['per_joint_scores'],
            'worst_frames': skeleton_result['worst_frames'],
            'per_frame_shape': mask_result['per_frame_shape'],
            'energy_details': mask_result['energy_details'],
            'direction_similarity': trajectory_result['direction_similarity'],
            'teacher_fps': teacher_data['fps'],
            'student_fps': student_data['fps'],
        }
