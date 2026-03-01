"""
Skeleton-based comparison metrics: joint angles and center of gravity.
"""
import numpy as np


def compute_joint_angles(landmarks, angle_definitions):
    """
    Calculate 2D joint angles for a sequence of frames.

    For each (parent, joint, child) triplet, computes the angle at 'joint'
    formed by the vectors joint→parent and joint→child using the dot product.

    Args:
        landmarks: Array of shape (N, 33, 4) — [x, y, z, visibility] per landmark.
        angle_definitions: Tuple of (parent, joint, child) index triplets.

    Returns:
        angles: Array of shape (N, len(angle_definitions)) — angles in degrees.
    """
    # Use only x, y (2D)
    positions = landmarks[:, :, :2]  # (N, 33, 2)

    num_frames = positions.shape[0]
    num_angles = len(angle_definitions)
    angles = np.zeros((num_frames, num_angles), dtype=np.float32)

    for j, (parent, joint, child) in enumerate(angle_definitions):
        vec_a = positions[:, parent, :] - positions[:, joint, :]  # (N, 2)
        vec_b = positions[:, child, :] - positions[:, joint, :]   # (N, 2)

        # Dot product and magnitudes
        dot = (vec_a * vec_b).sum(axis=-1)  # (N,)
        mag_a = np.linalg.norm(vec_a, axis=-1)  # (N,)
        mag_b = np.linalg.norm(vec_b, axis=-1)  # (N,)

        # Avoid division by zero
        denom = mag_a * mag_b
        denom = np.where(denom == 0, 1e-8, denom)

        cos_angle = np.clip(dot / denom, -1.0, 1.0)
        angles[:, j] = np.degrees(np.arccos(cos_angle))

    return angles


def compute_cog(landmarks, cog_weights):
    """
    Calculate the weighted center of gravity for each frame.

    Args:
        landmarks: Array of shape (N, 33, 4) — landmark data.
        cog_weights: Dict mapping joint index → body segment weight.

    Returns:
        cog: Array of shape (N, 2) — center of gravity (x, y) per frame.
    """
    positions = landmarks[:, :, :2]  # (N, 33, 2)

    joint_indices = list(cog_weights.keys())
    weights = np.array([cog_weights[i] for i in joint_indices], dtype=np.float32)  # (J,)
    weight_sum = weights.sum()

    selected = positions[:, joint_indices, :]  # (N, J, 2)

    # Weighted average: sum(weight_i * pos_i) / sum(weights)
    cog = np.einsum('j,njd->nd', weights, selected) / weight_sum  # (N, 2)

    return cog


def compare_angles(teacher_angles, student_angles, tolerances, sigma=25.0):
    """
    Score the similarity of joint angles using exponential decay penalty.

    Within tolerance: score = 100.
    Beyond tolerance: score = 100 * exp(-((error - tolerance) / sigma)²).

    Args:
        teacher_angles: Array of shape (N, J) — teacher joint angles in degrees.
        student_angles: Array of shape (N, J) — student joint angles (DTW-aligned).
        tolerances: Dict of joint name → tolerance in degrees.
        sigma: Decay parameter — error beyond tolerance at which score drops to ~37%.

    Returns:
        score: Float 0-100 — overall skeleton accuracy score.
        per_joint_scores: Dict of joint name → score (0-100).
        worst_frames: List of (frame_idx, joint_name, error_degrees) for feedback.
    """
    joint_names = list(tolerances.keys())
    num_joints = len(joint_names)
    num_frames = teacher_angles.shape[0]

    errors = np.abs(teacher_angles - student_angles)  # (N, J)

    # Build per-frame, per-joint scores
    frame_scores = np.zeros_like(errors)  # (N, J)

    for j, name in enumerate(joint_names):
        tol = tolerances[name]
        excess = np.maximum(0, errors[:, j] - tol)
        frame_scores[:, j] = 100.0 * np.exp(-((excess / sigma) ** 2))

    # Per-joint average across frames
    per_joint_scores = {}
    for j, name in enumerate(joint_names):
        per_joint_scores[name] = round(float(frame_scores[:, j].mean()), 1)

    # Overall score: average across all joints
    score = float(frame_scores.mean())

    # Worst frames: find the top offenders for feedback
    worst_frames = []
    num_worst = min(5, num_frames)  # Top 5 worst moments

    # Mean score per frame across joints
    mean_frame_scores = frame_scores.mean(axis=1)  # (N,)
    worst_indices = np.argsort(mean_frame_scores)[:num_worst]

    for idx in worst_indices:
        worst_joint_j = np.argmax(errors[idx])
        worst_frames.append((
            int(idx),
            joint_names[worst_joint_j % len(joint_names)],
            round(float(errors[idx, worst_joint_j]), 1),
        ))

    return round(score, 1), per_joint_scores, worst_frames


def compare_cog(teacher_cog, student_cog, sigma=15.0):
    """
    Score the similarity of center-of-gravity positions.

    Uses exponential decay on the Euclidean distance between teacher and student CoG.

    Args:
        teacher_cog: Array of shape (N, 2) — teacher CoG per frame.
        student_cog: Array of shape (N, 2) — student CoG per frame (DTW-aligned).
        sigma: Distance (in coordinate units) at which score drops to ~37%.

    Returns:
        score: Float 0-100 — CoG similarity score.
    """
    distances = np.linalg.norm(teacher_cog - student_cog, axis=-1)  # (N,)
    frame_scores = 100.0 * np.exp(-((distances / sigma) ** 2))
    score = float(frame_scores.mean())
    return round(score, 1)
