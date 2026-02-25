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

