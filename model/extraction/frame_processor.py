"""
Per-frame pose processing for the extraction pipeline.

Handles landmark filtering, scale calibration, follow-cam normalization,
and mask warping. Maintains filter state across frames.
"""
import cv2
import numpy as np

from model.config.extraction_config import ExtractionConfig
from model.extraction.one_euro_filter import OneEuroFilter


class FrameProcessor:
    """Filters, calibrates, and normalizes pose data frame by frame."""

    def __init__(self, config: ExtractionConfig) -> None:
        """
        Initialize the frame processor.

        Args:
            config: ExtractionConfig with filter params, normalization targets, etc.
        """
        self._config = config
        self._filters: dict = {}
        self._fixed_scale: float | None = None

    def process_frame(
        self,
        raw_landmarks: list,
        segmentation_mask: np.ndarray,
        vid_w: int,
        vid_h: int,
        timestamp_ms: float,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """
        Process a single frame's pose data through the full normalization pipeline.

        Args:
            raw_landmarks: MediaPipe landmark list for one detected pose.
            segmentation_mask: Raw float mask from MediaPipe (0-1 range).
            vid_w: Video frame width in pixels.
            vid_h: Video frame height in pixels.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            Tuple of (filtered_landmarks, normalized_mask, trajectory) where
            filtered_landmarks is shape (33, 4), normalized_mask is uint8,
            and trajectory is [hip_x, hip_y] in original pixels.
        """
        filtered = self._filter_landmarks(raw_landmarks, timestamp_ms)
        hip_center = self._get_hip_center(filtered, vid_w, vid_h)
        self._calibrate_if_needed(filtered, vid_w, vid_h)
        affine = self._build_affine(hip_center)
        norm_mask = self._warp_mask(segmentation_mask, affine)
        trajectory = [hip_center[0], hip_center[1]]

        return filtered, norm_mask, trajectory

    @property
    def fixed_scale(self) -> float | None:
        """The fixed scale factor, or None if not yet calibrated."""
        return self._fixed_scale

    def _filter_landmarks(
        self,
        raw_landmarks: list,
        timestamp_ms: float,
    ) -> np.ndarray:
        """
        Apply One Euro Filters to raw MediaPipe landmarks.

        Creates filter instances on first encounter of each landmark index.

        Args:
            raw_landmarks: MediaPipe landmark list for one detected pose.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            Filtered landmarks as array of shape (33, 4) with [x, y, z, visibility].
        """
        filtered = np.zeros((33, 4), dtype=np.float32)
        time_sec = timestamp_ms / 1000.0

        for i, lm in enumerate(raw_landmarks):
            if i not in self._filters:
                self._filters[i] = self._create_filter_triplet(time_sec, lm)

            filtered[i] = [
                self._filters[i][0](time_sec, lm.x),
                self._filters[i][1](time_sec, lm.y),
                self._filters[i][2](time_sec, lm.z),
                lm.visibility,
            ]

        return filtered

    def _create_filter_triplet(self, time_sec: float, landmark: object) -> list[OneEuroFilter]:
        """
        Create three One Euro Filters (x, y, z) for a single landmark.

        Args:
            time_sec: Initial timestamp in seconds.
            landmark: MediaPipe landmark with .x, .y, .z attributes.

        Returns:
            List of three OneEuroFilter instances.
        """
        cfg = self._config
        return [
            OneEuroFilter(time_sec, landmark.x, min_cutoff=cfg.filter_min_cutoff, beta=cfg.filter_beta),
            OneEuroFilter(time_sec, landmark.y, min_cutoff=cfg.filter_min_cutoff, beta=cfg.filter_beta),
            OneEuroFilter(time_sec, landmark.z, min_cutoff=cfg.filter_min_cutoff, beta=cfg.filter_beta),
        ]

    def _calibrate_if_needed(self, landmarks: np.ndarray, vid_w: int, vid_h: int) -> None:
        """
        Compute and store the fixed scale factor on the first frame.

        Uses the torso length (mid-shoulder to mid-hip distance) to determine
        how much to scale all frames so the torso matches the target size.

        Args:
            landmarks: Filtered landmarks of shape (33, 4).
            vid_w: Video frame width in pixels.
            vid_h: Video frame height in pixels.
        """
        if self._fixed_scale is not None:
            return

        torso_len = self._get_torso_length(landmarks, vid_w, vid_h)
        self._fixed_scale = self._config.target_torso_px / torso_len
        print(f"Calibration Complete. Fixed Scale: {self._fixed_scale:.2f}")

    def _get_hip_center(self, landmarks: np.ndarray, vid_w: int, vid_h: int) -> np.ndarray:
        """
        Compute the mid-hip position in pixel coordinates.

        Args:
            landmarks: Filtered landmarks of shape (33, 4) with normalized coords.
            vid_w: Video frame width in pixels.
            vid_h: Video frame height in pixels.

        Returns:
            Mid-hip position as array of [x, y] in pixels.
        """
        l_hip = np.array([landmarks[23, 0] * vid_w, landmarks[23, 1] * vid_h])
        r_hip = np.array([landmarks[24, 0] * vid_w, landmarks[24, 1] * vid_h])
        return (l_hip + r_hip) / 2.0

    def _get_torso_length(self, landmarks: np.ndarray, vid_w: int, vid_h: int) -> float:
        """
        Compute the torso length (mid-shoulder to mid-hip) in pixels.

        Args:
            landmarks: Filtered landmarks of shape (33, 4) with normalized coords.
            vid_w: Video frame width in pixels.
            vid_h: Video frame height in pixels.

        Returns:
            Torso length in pixels.
        """
        l_shoulder = np.array([landmarks[11, 0] * vid_w, landmarks[11, 1] * vid_h])
        r_shoulder = np.array([landmarks[12, 0] * vid_w, landmarks[12, 1] * vid_h])
        mid_shoulder = (l_shoulder + r_shoulder) / 2.0
        mid_hip = self._get_hip_center(landmarks, vid_w, vid_h)
        return float(np.linalg.norm(mid_shoulder - mid_hip))

    def _build_affine(self, hip_center: np.ndarray) -> np.ndarray:
        """
        Build a 2x3 affine matrix that centers the hip and applies fixed scale.

        Args:
            hip_center: (x, y) hip position in original pixels.

        Returns:
            Affine matrix of shape (2, 3).
        """
        norm_center = self._config.norm_center
        tx = norm_center[0] - (hip_center[0] * self._fixed_scale)
        ty = norm_center[1] - (hip_center[1] * self._fixed_scale)

        return np.float32([
            [self._fixed_scale, 0, tx],
            [0, self._fixed_scale, ty],
        ])

    def _warp_mask(self, mask: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the follow-cam affine transform to a segmentation mask.

        Args:
            mask: Raw segmentation mask from MediaPipe (float, 0-1 range).
            affine_matrix: 2x3 affine transform from _build_affine.

        Returns:
            Normalized mask as uint8 (0-255).
        """
        target_size = self._config.target_mask_size
        warped = cv2.warpAffine(mask, affine_matrix, target_size, flags=cv2.INTER_LINEAR)
        return (warped * 255).astype(np.uint8)
