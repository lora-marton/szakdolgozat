# Walkthrough — Preprocessor & Skeleton Metrics

## Preprocessor Module (completed earlier)

Created `preprocessing/` sub-package: audio cross-correlation for sync, motion energy for trim, intersection for shared range.

## Skeleton Metrics

### What Was Implemented

#### [skeleton_metrics.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/skeleton_metrics.py)

| Function | Purpose |
|----------|---------|
| [compute_joint_angles](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/skeleton_metrics.py#7-45) | 2D angle at each joint via dot product |
| [compute_cog](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/skeleton_metrics.py#47-70) | Weighted center of gravity per frame |
| [compare_angles](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/skeleton_metrics.py#72-129) | Exponential decay scoring with per-joint tolerances |
| [compare_cog](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/skeleton_metrics.py#131-149) | CoG distance scoring with exponential decay |

```diff:skeleton_metrics.py
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
===
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

```

#### [config.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/config.py)

Added to [ComparisonConfig](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/config.py#48-111):
- `weight_angles = 0.80`, `weight_cog = 0.20` (skeleton sub-weights)
- `angle_sigma = 25.0°` (exponential decay steepness for angles)
- `cog_sigma = 0.05` (decay steepness for CoG in 0–1 normalized coords)

```diff:config.py
from dataclasses import dataclass, field
import mediapipe as mp


@dataclass(frozen=True)
class ExtractionConfig:
    """All configuration constants for pose extraction."""

    # Model
    model_path: str = 'model/pose_landmarker_heavy.task'

    # Timing
    target_fps: float = 60.0

    # Normalization
    target_torso_px: float = 40.0           # Torso length in normalized space (small to fit raised arms)
    target_mask_size: tuple = (256, 256)
    norm_center: tuple = (128, 128)

    # Detection confidence
    min_pose_detection_confidence: float = 0.8
    min_pose_presence_confidence: float = 0.8
    min_tracking_confidence: float = 0.8

    # One Euro Filter
    filter_min_cutoff: float = 1.0
    filter_beta: float = 20.0

    # Skeleton connections for visualization
    pose_connections: tuple = (
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
        (11, 23), (12, 24), (23, 24),                       # Torso
        (23, 25), (25, 27), (24, 26), (26, 28),             # Legs
    )

    def create_landmarker_options(self):
        """Create MediaPipe PoseLandmarkerOptions from this config."""
        return mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=True,
        )


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for dance comparison scoring."""

    # Overall scoring weights (must sum to 1.0)
    weight_skeleton: float = 0.50
    weight_trajectory: float = 0.30
    weight_mask: float = 0.20

    # DTW: which joints to use for alignment (12 main joints)
    dtw_joints: tuple = (
        11, 12,  # Shoulders
        13, 14,  # Elbows
        15, 16,  # Wrists
        23, 24,  # Hips
        25, 26,  # Knees
        27, 28,  # Ankles
    )

    # Joint angle tolerance buffers (degrees) — style vs mistake
    joint_tolerances: dict = field(default_factory=lambda: {
        'hips': 5.0,
        'knees': 15.0,
        'elbows': 20.0,
        'wrists': 25.0,
        'shoulders': 10.0,
        'ankles': 15.0,
    })

    # Joint angle definitions: (parent, joint, child) triplets
    joint_angles: tuple = (
        (11, 13, 15),  # Left elbow (shoulder → elbow → wrist)
        (12, 14, 16),  # Right elbow
        (23, 25, 27),  # Left knee (hip → knee → ankle)
        (24, 26, 28),  # Right knee
        (13, 11, 23),  # Left shoulder (elbow → shoulder → hip)
        (14, 12, 24),  # Right shoulder
    )

    # Center of Gravity: segment weights (biomechanics approximation)
    cog_weights: dict = field(default_factory=lambda: {
        0: 0.08,    # Head/Nose
        11: 0.06,   # L Shoulder
        12: 0.06,   # R Shoulder
        13: 0.03,   # L Elbow
        14: 0.03,   # R Elbow
        15: 0.02,   # L Wrist
        16: 0.02,   # R Wrist
        23: 0.15,   # L Hip
        24: 0.15,   # R Hip
        25: 0.06,   # L Knee
        26: 0.06,   # R Knee
        27: 0.02,   # L Ankle
        28: 0.02,   # R Ankle
    })


# Default config instances
DEFAULT_CONFIG = ExtractionConfig()
DEFAULT_COMPARISON_CONFIG = ComparisonConfig()

===
from dataclasses import dataclass, field
import mediapipe as mp


@dataclass(frozen=True)
class ExtractionConfig:
    """All configuration constants for pose extraction."""

    # Model
    model_path: str = 'model/pose_landmarker_heavy.task'

    # Timing
    target_fps: float = 60.0

    # Normalization
    target_torso_px: float = 40.0           # Torso length in normalized space (small to fit raised arms)
    target_mask_size: tuple = (256, 256)
    norm_center: tuple = (128, 128)

    # Detection confidence
    min_pose_detection_confidence: float = 0.8
    min_pose_presence_confidence: float = 0.8
    min_tracking_confidence: float = 0.8

    # One Euro Filter
    filter_min_cutoff: float = 1.0
    filter_beta: float = 20.0

    # Skeleton connections for visualization
    pose_connections: tuple = (
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
        (11, 23), (12, 24), (23, 24),                       # Torso
        (23, 25), (25, 27), (24, 26), (26, 28),             # Legs
    )

    def create_landmarker_options(self):
        """Create MediaPipe PoseLandmarkerOptions from this config."""
        return mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=True,
        )


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for dance comparison scoring."""

    # Overall scoring weights (must sum to 1.0)
    weight_skeleton: float = 0.50
    weight_trajectory: float = 0.30
    weight_mask: float = 0.20

    # Skeleton sub-weights: angles vs CoG (must sum to 1.0)
    weight_angles: float = 0.80
    weight_cog: float = 0.20

    # Exponential decay parameters
    angle_sigma: float = 25.0   # degrees — score ≈37% at tolerance + sigma
    cog_sigma: float = 0.05    # normalized coords (0–1) — score ≈37% at this distance

    # DTW: which joints to use for alignment (12 main joints)
    dtw_joints: tuple = (
        11, 12,  # Shoulders
        13, 14,  # Elbows
        15, 16,  # Wrists
        23, 24,  # Hips
        25, 26,  # Knees
        27, 28,  # Ankles
    )

    # Joint angle tolerance buffers (degrees) — style vs mistake
    joint_tolerances: dict = field(default_factory=lambda: {
        'hips': 5.0,
        'knees': 15.0,
        'elbows': 20.0,
        'wrists': 25.0,
        'shoulders': 10.0,
        'ankles': 15.0,
    })

    # Joint angle definitions: (parent, joint, child) triplets
    joint_angles: tuple = (
        (11, 13, 15),  # Left elbow (shoulder → elbow → wrist)
        (12, 14, 16),  # Right elbow
        (23, 25, 27),  # Left knee (hip → knee → ankle)
        (24, 26, 28),  # Right knee
        (13, 11, 23),  # Left shoulder (elbow → shoulder → hip)
        (14, 12, 24),  # Right shoulder
    )

    # Center of Gravity: segment weights (biomechanics approximation)
    cog_weights: dict = field(default_factory=lambda: {
        0: 0.08,    # Head/Nose
        11: 0.06,   # L Shoulder
        12: 0.06,   # R Shoulder
        13: 0.03,   # L Elbow
        14: 0.03,   # R Elbow
        15: 0.02,   # L Wrist
        16: 0.02,   # R Wrist
        23: 0.15,   # L Hip
        24: 0.15,   # R Hip
        25: 0.06,   # L Knee
        26: 0.06,   # R Knee
        27: 0.02,   # L Ankle
        28: 0.02,   # R Ankle
    })


@dataclass(frozen=True)
class PreprocessorConfig:
    """Configuration for preprocessing (audio sync + motion trimming)."""

    # Audio cross-correlation
    audio_sample_rate: int = 22050

    # Motion energy thresholds
    motion_threshold_ratio: float = 0.15   # fraction of max energy to count as active
    min_active_duration: int = 10          # frames of sustained motion required


# Default config instances
DEFAULT_CONFIG = ExtractionConfig()
DEFAULT_PREPROCESSOR_CONFIG = PreprocessorConfig()
DEFAULT_COMPARISON_CONFIG = ComparisonConfig()

```

#### [comparator.py](file:///c:/Users/marto/Documents/egyetem/szakdoga/szakdolgozat/model/comparison/comparator.py)

Phase B now computes angle score + CoG score and combines them with configured weights.

```diff:comparator.py
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
===
"""
Dance comparison orchestrator.

Loads teacher and student HDF5 data, runs all comparison metrics,
and produces a structured result with scores and feedback.
"""
import os
import numpy as np
import h5py

from model.config import DEFAULT_COMPARISON_CONFIG
from model.preprocessing.preprocessor import preprocess
from model.comparison.dtw import align_sequences
from model.comparison.skeleton_metrics import compute_joint_angles, compute_cog, compare_angles, compare_cog
from model.comparison.mask_metrics import compute_iou
from model.comparison.trajectory_metrics import compare_trajectories


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
    angle_score, per_joint_scores, worst_frames = compare_angles(
        teacher_angles, student_angles, config.joint_tolerances, config.angle_sigma,
    )

    teacher_cog = compute_cog(aligned_teacher_lm, config.cog_weights)
    student_cog = compute_cog(aligned_student_lm, config.cog_weights)
    cog_score = compare_cog(teacher_cog, student_cog, config.cog_sigma)

    skeleton_score = config.weight_angles * angle_score + config.weight_cog * cog_score

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
```

### Verification Results

| Test | Expected | Got | Status |
|------|----------|-----|--------|
| 90° elbow angle | 90.0° | 90.0° | ✅ |
| CoG midpoint of symmetric hips | (0.5, 0.5) | (0.5, 0.5) | ✅ |
| Identical sequences | 100.0 | 100.0 | ✅ |
| 80° error sequences | near 0 | 0.2 | ✅ |
| Identical CoG | 100.0 | 100.0 | ✅ |
| CoG shift 0.03 (3%) | moderate penalty | 48.7 | ✅ |
| CoG shift 0.10 (10%) | harsh penalty | 0.0 | ✅ |

> [!IMPORTANT]
> During verification, discovered `cog_sigma` was initially set to 15.0 (pixel-space assumption) which made CoG scoring meaningless in the 0–1 normalized coordinate space. Fixed to 0.05.
