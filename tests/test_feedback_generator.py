"""
Integration test for the FeedbackGenerator orchestrator.

Sets up synthetic teacher/student mp4 files plus the HDF5 landmark data
that VideoFeedback expects, then drives FeedbackGenerator.generate_feedback
end-to-end and verifies the returned dict structure and output artifacts.
"""

import logging
import os
import shutil
import sys
import tempfile

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests  # noqa: F401

from model.config import DEFAULT_FEEDBACK_CONFIG
from model.feedback.feedback_generator import FeedbackGenerator

logger = logging.getLogger(__name__)


def _make_synthetic_results() -> dict:
    """Build a comparison-results dict with plausible values."""
    return {
        "overall_score": 78.0,
        "skeleton_score": 80.0,
        "trajectory_score": 72.0,
        "mask_score": 77.0,
        "timing_cost": 0.25,
        "alignment_path": [(i, i) for i in range(30)],
        "per_joint_scores": {
            "left_elbow": 82.0,
            "right_elbow": 81.0,
            "left_knee": 79.0,
            "right_knee": 80.0,
            "left_shoulder": 83.0,
            "right_shoulder": 84.0,
            "left_hip": 78.0,
            "right_hip": 78.0,
            "left_wrist": 75.0,
            "right_wrist": 76.0,
            "left_ankle": 77.0,
            "right_ankle": 77.0,
            "left_inner_shoulder": 80.0,
            "right_inner_shoulder": 80.0,
            "left_inner_hip": 80.0,
            "right_inner_hip": 80.0,
        },
        "worst_frames": [(12, 60.0), (20, 65.0)],
        "per_frame_shape": np.ones(30, dtype=np.float32) * 0.8,
        "energy_details": {
            "energy_score": 0.82,
            "per_frame_ratios": np.ones(29, dtype=np.float32) * 0.8,
            "teacher_energy": np.ones(29, dtype=np.float32) * 5.0,
            "student_energy": np.ones(29, dtype=np.float32) * 4.0,
        },
        "preprocess_info": {"audio_offset": 0, "teacher_offset": 0, "student_offset": 0},
        "teacher_fps": 30.0,
        "student_fps": 30.0,
        "direction_similarity": 0.72,
    }


class TestFeedbackGenerator:
    """Test suite for FeedbackGenerator.generate_feedback."""

    def setup_method(self):
        """Create a temp directory with synthetic videos and HDF5 files."""
        self.tmpdir = tempfile.mkdtemp(prefix="test_feedback_gen_")
        self.num_frames = 30

        self.teacher_video = os.path.join(self.tmpdir, "teacher.mp4")
        self.student_video = os.path.join(self.tmpdir, "student.mp4")
        self._create_video(self.teacher_video, color=(80, 40, 40))
        self._create_video(self.student_video, color=(40, 40, 80))

        self._create_h5("teacher")
        self._create_h5("student")

    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_video(self, path: str, color: tuple):
        """Write a short solid-color mp4 for testing."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30.0, (320, 240))
        for _ in range(self.num_frames):
            frame = np.full((240, 320, 3), color, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def _create_h5(self, label: str):
        """Write fake HDF5 landmark/mask/trajectory data for one session."""
        landmarks = np.random.rand(self.num_frames, 33, 4).astype(np.float32)
        landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0.1, 0.9)
        landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0.1, 0.9)
        landmarks[:, :, 3] = 0.9

        trajectory = np.random.rand(self.num_frames, 2).astype(np.float32) * 100
        with h5py.File(os.path.join(self.tmpdir, f"{label}_data.h5"), "w") as f:
            f.create_dataset("raw", data=landmarks)
            f.create_dataset("trajectory", data=trajectory)
            f.attrs["fps"] = 60.0
            f.attrs["fixed_scale"] = 1.0

        masks = np.zeros((self.num_frames, 256, 256), dtype=np.uint8)
        with h5py.File(os.path.join(self.tmpdir, f"{label}_masks.h5"), "w") as f:
            f.create_dataset("masks", data=masks)

    def test_returns_expected_keys(self):
        """The orchestrator result should include feedback, markers, and video filename."""
        results = _make_synthetic_results()

        out = FeedbackGenerator.generate_feedback(
            results,
            self.teacher_video,
            self.student_video,
            self.tmpdir,
            DEFAULT_FEEDBACK_CONFIG,
        )

        logger.info("=== Feedback Generator Keys ===")
        logger.info("  keys: %s", sorted(out.keys()))
        assert set(out.keys()) == {"feedback", "timeline_markers", "feedback_video"}
        assert isinstance(out["feedback"], list) and len(out["feedback"]) > 0
        assert isinstance(out["timeline_markers"], list)
        logger.info("  PASSED\n")

    def test_feedback_video_file_exists(self):
        """The generated feedback video must actually exist on disk."""
        results = _make_synthetic_results()

        out = FeedbackGenerator.generate_feedback(
            results,
            self.teacher_video,
            self.student_video,
            self.tmpdir,
        )

        produced = os.path.join(self.tmpdir, out["feedback_video"])
        logger.info("=== Feedback Video File ===")
        logger.info("  produced: %s", produced)
        assert os.path.isfile(produced)
        assert out["feedback_video"].endswith(".mp4")
        logger.info("  PASSED\n")

    def test_uses_default_config_when_none(self):
        """Passing config=None should fall back to DEFAULT_FEEDBACK_CONFIG."""
        results = _make_synthetic_results()

        out = FeedbackGenerator.generate_feedback(
            results,
            self.teacher_video,
            self.student_video,
            self.tmpdir,
            None,
        )

        logger.info("=== Feedback Default Config ===")
        logger.info("  feedback lines: %d", len(out["feedback"]))
        assert len(out["feedback"]) > 0
        logger.info("  PASSED\n")
