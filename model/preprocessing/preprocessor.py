"""
Preprocessing orchestrator for dance comparison.

Loads HDF5 data, synchronises and trims two dance sequences before DTW alignment:
1. Load HDF5 data           → read landmarks, masks, trajectory from session files
2. Audio cross-correlation   → find frame offset between the videos
3. Apply offset              → shift/trim the leading video
4. Motion energy detection   → find active range in each sequence
5. Intersection              → keep only the overlapping active region
"""

import os

import h5py
import numpy as np

from model.config import DEFAULT_PREPROCESSOR_CONFIG
from model.config.preprocessor_config import PreprocessorConfig
from model.preprocessing.audio_sync import AudioSync
from model.preprocessing.motion_energy import MotionEnergy


class Preprocessor:
    """Synchronises and trims two dance sequences to their shared active region."""

    @staticmethod
    def preprocess(
        output_dir: str,
        teacher_video: str,
        student_video: str,
        config: PreprocessorConfig | None = None,
    ) -> tuple[dict, dict, dict]:
        """
        Load, synchronise and trim teacher/student data to the shared active dance region.

        Args:
            output_dir: Path to session directory containing teacher_*.h5 and student_*.h5 files.
            teacher_video: Path to the teacher video file (for audio extraction).
            student_video: Path to the student video file (for audio extraction).
            config: PreprocessorConfig instance (uses defaults if None).

        Returns:
            Tuple of (teacher_data, student_data, preprocess_info) where
            preprocess_info contains 'audio_offset', 'teacher_offset', and
            'student_offset' (number of frames trimmed from each sequence).
        """
        if config is None:
            config = DEFAULT_PREPROCESSOR_CONFIG

        teacher_data = Preprocessor._load_session_data(output_dir, "teacher")
        student_data = Preprocessor._load_session_data(output_dir, "student")

        fps = teacher_data.get("fps", 60.0)

        offset = AudioSync.compute_audio_offset(
            teacher_video,
            student_video,
            target_fps=fps,
            sr=config.audio_sample_rate,
        )
        print(
            f"[Preprocessor] Audio offset: {offset} frames "
            f"({'teacher leads' if offset > 0 else 'student leads' if offset < 0 else 'in sync'})"
        )

        teacher_audio_trim = offset if offset > 0 else 0
        student_audio_trim = -offset if offset < 0 else 0

        teacher_data, student_data = Preprocessor._apply_offset(teacher_data, student_data, offset)

        t_energy = MotionEnergy.compute_motion_energy(teacher_data["landmarks"])
        s_energy = MotionEnergy.compute_motion_energy(student_data["landmarks"])

        t_start, t_end = MotionEnergy.find_active_range(
            t_energy,
            config.motion_threshold_ratio,
            config.min_active_duration,
            config.active_window_ratio,
        )
        s_start, s_end = MotionEnergy.find_active_range(
            s_energy,
            config.motion_threshold_ratio,
            config.min_active_duration,
            config.active_window_ratio,
        )

        print(f"[Preprocessor] Teacher active range: frames {t_start}--{t_end}")
        print(f"[Preprocessor] Student active range: frames {s_start}--{s_end}")

        shared_start = max(t_start, s_start)
        shared_end = min(t_end, s_end)

        if shared_start >= shared_end:
            print("[Preprocessor] WARNING: No overlapping active region found. " "Skipping trimming.")
            preprocess_info = {
                "audio_offset": offset,
                "teacher_offset": teacher_audio_trim,
                "student_offset": student_audio_trim,
            }
            return teacher_data, student_data, preprocess_info

        print(
            f"[Preprocessor] Shared active region: frames {shared_start}--{shared_end} "
            f"({shared_end - shared_start} frames)"
        )

        teacher_data = Preprocessor._slice_data(teacher_data, shared_start, shared_end)
        student_data = Preprocessor._slice_data(student_data, shared_start, shared_end)

        preprocess_info = {
            "audio_offset": offset,
            "teacher_offset": teacher_audio_trim + shared_start,
            "student_offset": student_audio_trim + shared_start,
        }

        return teacher_data, student_data, preprocess_info

    @staticmethod
    def _apply_offset(
        teacher_data: dict,
        student_data: dict,
        offset: int,
    ) -> tuple[dict, dict]:
        """
        Trim the leading frames from the earlier video so both start at the same
        musical moment, then truncate to the shorter length.

        Args:
            teacher_data: Teacher data dict with array entries.
            student_data: Student data dict with array entries.
            offset: Frame offset from audio cross-correlation (positive = teacher leads).

        Returns:
            Tuple of (teacher_data, student_data) trimmed and length-matched.
        """
        if offset > 0:
            teacher_data = Preprocessor._slice_data(teacher_data, offset, None)
        elif offset < 0:
            student_data = Preprocessor._slice_data(student_data, -offset, None)

        min_len = min(
            len(teacher_data["landmarks"]),
            len(student_data["landmarks"]),
        )
        teacher_data = Preprocessor._slice_data(teacher_data, 0, min_len)
        student_data = Preprocessor._slice_data(student_data, 0, min_len)

        return teacher_data, student_data

    @staticmethod
    def _load_session_data(output_dir: str, label: str) -> dict:
        """
        Load landmarks, masks, and trajectory from a session's HDF5 files.

        Args:
            output_dir: Path to the session directory.
            label: Either 'teacher' or 'student'.

        Returns:
            Dict with 'landmarks', 'masks', 'trajectory' numpy arrays
            plus 'fps' and 'fixed_scale' scalar metadata.
        """
        data_path = os.path.join(output_dir, f"{label}_data.h5")
        mask_path = os.path.join(output_dir, f"{label}_masks.h5")

        with h5py.File(data_path, "r") as f:
            landmarks = f["raw"][:]
            trajectory = f["trajectory"][:]
            fps = f.attrs.get("fps", 60.0)
            fixed_scale = f.attrs.get("fixed_scale", 1.0)

        with h5py.File(mask_path, "r") as f:
            masks = f["masks"][:]

        return {
            "landmarks": landmarks,
            "trajectory": trajectory,
            "masks": masks,
            "fps": fps,
            "fixed_scale": fixed_scale,
        }

    @staticmethod
    def _slice_data(data: dict, start: int, end: int | None) -> dict:
        """
        Return a copy of the data dict with all arrays sliced to [start:end].
        Non-array entries (e.g. 'fps', 'fixed_scale') are preserved as-is.

        Args:
            data: Data dict containing numpy arrays and scalar metadata.
            start: Start index for slicing.
            end: End index for slicing (None means slice to the end).

        Returns:
            New dict with sliced arrays and preserved scalar values.
        """
        sliced = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced
