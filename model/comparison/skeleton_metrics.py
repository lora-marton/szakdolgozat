"""
Skeleton-based comparison metrics: joint angles and center of gravity.
"""
import numpy as np


class SkeletonMetrics:
    """Joint angle and center-of-gravity comparison between two pose sequences."""

    @staticmethod
    def compute_skeleton_score(
        teacher_landmarks: np.ndarray,
        student_landmarks: np.ndarray,
        config,
    ) -> dict:
        """Compute the combined skeleton comparison score.

        Runs joint angle comparison and center-of-gravity comparison,
        then combines them with configured weights.

        Args:
            teacher_landmarks: Array of shape (N, 33, 4), DTW-aligned.
            student_landmarks: Array of shape (N, 33, 4), DTW-aligned.
            config: ComparisonConfig instance.

        Returns:
            Dict with 'score' (0-100), 'per_joint_scores' dict,
            and 'worst_frames' list.
        """
        teacher_angles = SkeletonMetrics._compute_joint_angles(
            teacher_landmarks, config.joint_angles,
        )
        student_angles = SkeletonMetrics._compute_joint_angles(
            student_landmarks, config.joint_angles,
        )
        angle_score, per_joint_scores, worst_frames = SkeletonMetrics._compare_angles(
            teacher_angles, student_angles, config.joint_tolerances, config.angle_sigma,
        )

        teacher_cog = SkeletonMetrics._compute_cog(teacher_landmarks, config.cog_weights)
        student_cog = SkeletonMetrics._compute_cog(student_landmarks, config.cog_weights)
        cog_score = SkeletonMetrics._compare_cog(teacher_cog, student_cog, config.cog_sigma)

        skeleton_score = config.weight_angles * angle_score + config.weight_cog * cog_score

        return {
            'score': round(skeleton_score, 1),
            'per_joint_scores': per_joint_scores,
            'worst_frames': worst_frames,
        }

    @staticmethod
    def _compute_joint_angles(landmarks: np.ndarray, angle_definitions: tuple) -> np.ndarray:
        """Calculate 2D joint angles for a sequence of frames.

        For each (parent, joint, child) triplet, computes the angle at the
        vertex joint formed by vectors joint->parent and joint->child using
        the dot-product formula.

        Args:
            landmarks: Array of shape (N, 33, 4) with x, y, z, visibility per landmark.
            angle_definitions: Tuple of (parent, joint, child) index triplets.

        Returns:
            Array of shape (N, len(angle_definitions)) with angles in degrees.
        """
        positions = landmarks[:, :, :2]

        num_frames = positions.shape[0]
        num_angles = len(angle_definitions)
        angles = np.zeros((num_frames, num_angles), dtype=np.float32)

        for j, (parent, joint, child) in enumerate(angle_definitions):
            vec_a = positions[:, parent, :] - positions[:, joint, :]
            vec_b = positions[:, child, :] - positions[:, joint, :]

            dot = (vec_a * vec_b).sum(axis=-1)
            mag_a = np.linalg.norm(vec_a, axis=-1)
            mag_b = np.linalg.norm(vec_b, axis=-1)

            denom = mag_a * mag_b
            denom = np.where(denom == 0, 1e-8, denom)

            cos_angle = np.clip(dot / denom, -1.0, 1.0)
            angles[:, j] = np.degrees(np.arccos(cos_angle))

        return angles

    @staticmethod
    def _compute_cog(landmarks: np.ndarray, cog_weights: dict) -> np.ndarray:
        """Calculate the weighted center of gravity for each frame.

        Args:
            landmarks: Array of shape (N, 33, 4) with landmark data.
            cog_weights: Dict mapping joint index to body-segment weight.

        Returns:
            Array of shape (N, 2) with center of gravity (x, y) per frame.
        """
        positions = landmarks[:, :, :2]

        joint_indices = list(cog_weights.keys())
        weights = np.array([cog_weights[i] for i in joint_indices], dtype=np.float32)
        weight_sum = weights.sum()

        selected = positions[:, joint_indices, :]

        cog = np.einsum('j,njd->nd', weights, selected) / weight_sum

        return cog

    @staticmethod
    def _compare_angles(
        teacher_angles: np.ndarray,
        student_angles: np.ndarray,
        tolerances: dict,
        sigma: float,
    ) -> tuple[float, dict, list]:
        """Score the similarity of joint angles using exponential decay.

        Within tolerance the score is 100. Beyond tolerance the score decays
        as 100 * exp(-((error - tolerance) / sigma)^2).

        Args:
            teacher_angles: Array of shape (N, J) with teacher angles in degrees.
            student_angles: Array of shape (N, J) with student angles (DTW-aligned).
            tolerances: Dict of joint name to tolerance in degrees.
            sigma: Decay parameter — error beyond tolerance at which score drops to ~37%.

        Returns:
            Tuple of (overall_score, per_joint_scores dict, worst_frames list).
        """
        joint_names = list(tolerances.keys())
        num_frames = teacher_angles.shape[0]

        errors = np.abs(teacher_angles - student_angles)

        frame_scores = np.zeros_like(errors)

        for j, name in enumerate(joint_names):
            tol = tolerances[name]
            excess = np.maximum(0, errors[:, j] - tol)
            frame_scores[:, j] = 100.0 * np.exp(-((excess / sigma) ** 2))

        per_joint_scores = {}
        for j, name in enumerate(joint_names):
            per_joint_scores[name] = round(float(frame_scores[:, j].mean()), 1)

        score = float(frame_scores.mean())

        worst_frames = []
        num_worst = min(5, num_frames)

        mean_frame_scores = frame_scores.mean(axis=1)
        worst_indices = np.argsort(mean_frame_scores)[:num_worst]

        for idx in worst_indices:
            worst_joint_j = np.argmax(errors[idx])
            worst_frames.append((
                int(idx),
                joint_names[worst_joint_j],
                round(float(errors[idx, worst_joint_j]), 1),
            ))

        return round(score, 1), per_joint_scores, worst_frames

    @staticmethod
    def _compare_cog(
        teacher_cog: np.ndarray,
        student_cog: np.ndarray,
        sigma: float,
    ) -> float:
        """Score the similarity of center-of-gravity positions.

        Uses exponential decay on the Euclidean distance between teacher
        and student CoG per frame.

        Args:
            teacher_cog: Array of shape (N, 2) with teacher CoG per frame.
            student_cog: Array of shape (N, 2) with student CoG per frame (DTW-aligned).
            sigma: Distance in coordinate units at which score drops to ~37%.

        Returns:
            CoG similarity score (0-100).
        """
        distances = np.linalg.norm(teacher_cog - student_cog, axis=-1)
        frame_scores = 100.0 * np.exp(-((distances / sigma) ** 2))
        score = float(frame_scores.mean())
        return round(score, 1)
