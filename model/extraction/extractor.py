"""
Pose extraction orchestrator for dance videos.

Opens a video file, runs MediaPipe pose detection on each frame,
delegates per-frame processing to FrameProcessor, and saves the
results as HDF5 files.
"""

import os

import cv2
import h5py
import mediapipe as mp
import numpy as np

from model.config import DEFAULT_EXTRACTION_CONFIG
from model.config.extraction_config import ExtractionConfig
from model.extraction.frame_processor import FrameProcessor
from model.extraction.visualization import Visualization


class Extractor:
    """Extracts pose landmarks, segmentation masks, and trajectory from a dance video."""

    @staticmethod
    def data_extraction(
        video_path: str,
        output_dir: str = "data",
        label: str = "dance",
        debug: bool = False,
        status_callback: object | None = None,
        config: ExtractionConfig | None = None,
    ) -> None:
        """
        Extract pose data from a video and save to HDF5 files.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory for output HDF5 files.
            label: Prefix for output filenames (e.g., 'teacher', 'student').
            debug: If True, show OpenCV debug windows.
            status_callback: Optional callback(msg: str) for progress updates.
            config: ExtractionConfig instance (uses DEFAULT_EXTRACTION_CONFIG if None).
        """
        if config is None:
            config = DEFAULT_EXTRACTION_CONFIG

        os.makedirs(output_dir, exist_ok=True)
        processor = FrameProcessor(config)
        options = config.create_landmarker_options()

        collected_raw = []
        collected_masks = []
        collected_trajectory = []

        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_interval_ms = 1000.0 / config.target_fps
            last_processed_time = -frame_interval_ms

            print(f"Source: {vid_w}x{vid_h} @ {source_fps} FPS. Target: {config.target_fps} FPS")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                timestamp_ms = (frame_idx * 1000.0) / source_fps

                if timestamp_ms < last_processed_time + frame_interval_ms - (1000.0 / source_fps / 2):
                    continue
                last_processed_time += frame_interval_ms

                if frame_idx % 60 == 0 and status_callback:
                    status_callback(f"Processing frame {int(frame_idx)}...")

                result = Extractor._detect_pose(frame, landmarker, timestamp_ms)

                landmarks, norm_mask, trajectory = Extractor._extract_frame_data(
                    result,
                    processor,
                    vid_w,
                    vid_h,
                    timestamp_ms,
                    config,
                )

                if debug and result.pose_landmarks:
                    Extractor._show_debug(frame, result, landmarks, norm_mask, config, vid_w, vid_h)

                collected_raw.append(landmarks)
                collected_masks.append(norm_mask)
                collected_trajectory.append(trajectory)

                if debug and cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            if debug:
                cv2.destroyAllWindows()

        Extractor._save_data(
            output_dir,
            label,
            collected_raw,
            collected_masks,
            collected_trajectory,
            config.target_fps,
            processor.fixed_scale,
        )

    @staticmethod
    def _detect_pose(
        frame: np.ndarray,
        landmarker: mp.tasks.vision.PoseLandmarker,
        timestamp_ms: float,
    ) -> mp.tasks.vision.PoseLandmarkerResult:
        """
        Run MediaPipe pose detection on a single frame.

        Args:
            frame: BGR video frame.
            landmarker: MediaPipe PoseLandmarker instance.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            MediaPipe PoseLandmarkerResult.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return landmarker.detect_for_video(mp_image, int(timestamp_ms))

    @staticmethod
    def _extract_frame_data(
        result: mp.tasks.vision.PoseLandmarkerResult,
        processor: FrameProcessor,
        vid_w: int,
        vid_h: int,
        timestamp_ms: float,
        config: ExtractionConfig,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """
        Extract landmarks, mask, and trajectory from a detection result.

        If no pose was detected, returns zeroed defaults.

        Args:
            result: MediaPipe detection result.
            processor: FrameProcessor instance for filtering and normalization.
            vid_w: Video frame width in pixels.
            vid_h: Video frame height in pixels.
            timestamp_ms: Frame timestamp in milliseconds.
            config: ExtractionConfig for default sizes.

        Returns:
            Tuple of (landmarks, normalized_mask, trajectory).
        """
        if not result.pose_landmarks:
            return (
                np.zeros((33, 4), dtype=np.float32),
                np.zeros(config.target_mask_size, dtype=np.uint8),
                [0.0, 0.0],
            )

        raw_landmarks = result.pose_landmarks[0]
        segmentation_mask = result.segmentation_masks[0].numpy_view()
        processed_frame = processor.process_frame(raw_landmarks, segmentation_mask, vid_w, vid_h, timestamp_ms)
        return processed_frame

    @staticmethod
    def _show_debug(
        frame: np.ndarray,
        result: mp.tasks.vision.PoseLandmarkerResult,
        landmarks: np.ndarray,
        norm_mask: np.ndarray,
        config: ExtractionConfig,
        vid_w: int,
        vid_h: int,
    ) -> None:
        """
        Display debug visualization windows.

        Args:
            frame: Original BGR video frame.
            result: MediaPipe detection result (for raw segmentation mask).
            landmarks: Filtered landmarks of shape (33, 4).
            norm_mask: Normalized mask (uint8).
            config: ExtractionConfig with pose_connections.
            vid_w: Video frame width in pixels.
            vid_h: Video frame height in pixels.
        """
        segmentation_mask = result.segmentation_masks[0].numpy_view()
        overlay = Visualization.draw_mask_overlay(frame, segmentation_mask)
        overlay = Visualization.draw_skeleton(overlay, landmarks, config.pose_connections, vid_w, vid_h)
        cv2.imshow("Main View (Skeleton + Mask)", overlay)
        cv2.imshow("Follow-Cam View (Centered)", norm_mask)

    @staticmethod
    def _save_data(
        output_dir: str,
        label: str,
        collected_raw: list,
        collected_masks: list,
        collected_trajectory: list,
        target_fps: float,
        fixed_scale: float | None,
    ) -> None:
        """
        Save extracted data to HDF5 files.

        Args:
            output_dir: Directory for output files.
            label: Prefix for filenames ('teacher' or 'student').
            collected_raw: List of landmark arrays.
            collected_masks: List of mask arrays.
            collected_trajectory: List of [x, y] trajectory points.
            target_fps: Frame rate used during extraction.
            fixed_scale: Scale factor from calibration (1.0 if not calibrated).
        """
        output_path = os.path.join(output_dir, f"{label}_data.h5")
        output_mask_path = os.path.join(output_dir, f"{label}_masks.h5")

        print(f"Saving {len(collected_raw)} frames...")

        with h5py.File(output_path, "w") as f:
            f.create_dataset("raw", data=np.array(collected_raw, dtype=np.float32))
            dset_traj = f.create_dataset("trajectory", data=np.array(collected_trajectory, dtype=np.float32))
            dset_traj.attrs["description"] = "Hip Center (x, y) in original pixels"
            f.attrs["fps"] = target_fps
            f.attrs["fixed_scale"] = fixed_scale if fixed_scale is not None else 1.0

        print(f"Saving masks to {output_mask_path} (gzip)...")
        with h5py.File(output_mask_path, "w") as f:
            f.create_dataset(
                "masks",
                data=np.array(collected_masks, dtype=np.uint8),
                compression="gzip",
                compression_opts=4,
            )

        print("Done!")
