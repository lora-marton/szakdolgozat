"""
Extraction configuration.

All constants for pose extraction: model path, target FPS,
normalization, detection confidence, filtering, and skeleton connections.
"""

from dataclasses import dataclass

import mediapipe as mp


@dataclass(frozen=True)
class ExtractionConfig:
    """All configuration constants for pose extraction."""

    model_path: str = "model/pose_landmarker_heavy.task"

    target_fps: float = 60.0
    output_fps: float = 30.0

    target_torso_px: float = 40.0
    target_mask_size: tuple = (256, 256)
    norm_center: tuple = (128, 128)

    min_pose_detection_confidence: float = 0.8
    min_pose_presence_confidence: float = 0.8
    min_tracking_confidence: float = 0.8

    filter_min_cutoff: float = 1.0
    filter_beta: float = 20.0

    pose_connections: tuple = (
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),
    )

    def create_landmarker_options(self) -> mp.tasks.vision.PoseLandmarkerOptions:
        """Create MediaPipe PoseLandmarkerOptions from this config."""
        return mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=True,
        )


DEFAULT_EXTRACTION_CONFIG = ExtractionConfig()
