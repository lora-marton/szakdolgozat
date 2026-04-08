"""
Integration smoke test for the VideoProcessor orchestrator.

Drives VideoProcessor.process_videos end-to-end while substituting the
heavy MediaPipe extractor and moviepy audio loader with lightweight
fakes. Preprocessing, comparison, feedback, and video rendering all run
for real against synthetic HDF5 + mp4 fixtures.
"""

import asyncio
import os
import shutil
import sys
import tempfile

import cv2
import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_for_tests
from config_for_tests import make_masks, make_stick_figure_landmarks, make_trajectory

from model import video_processor as video_processor_module
from model.preprocessing import audio_sync as audio_sync_module
from model.video_processor import VideoProcessor


def _create_video(path: str, num_frames: int, color: tuple):
    """Write a short solid-color mp4 at 30 fps."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (320, 240))
    for _ in range(num_frames):
        frame = np.full((240, 320, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_synthetic_hdf5(output_dir: str, label: str, num_frames: int):
    """Write synthetic landmark / trajectory / mask HDF5 files for one session."""
    landmarks = make_stick_figure_landmarks(num_frames=num_frames)
    for f in range(num_frames):
        landmarks[f, :, 0] += 0.002 * f
        landmarks[f, :, 1] += 0.001 * f

    trajectory = make_trajectory(num_frames=num_frames, start=(100.0, 120.0), velocity=(0.5, 0.2))

    with h5py.File(os.path.join(output_dir, f"{label}_data.h5"), "w") as f:
        f.create_dataset("raw", data=landmarks)
        f.create_dataset("trajectory", data=trajectory)
        f.attrs["fps"] = 30.0
        f.attrs["fixed_scale"] = 1.0

    masks = make_masks(num_frames=num_frames, h=256, w=256, radius=40)
    with h5py.File(os.path.join(output_dir, f"{label}_masks.h5"), "w") as f:
        f.create_dataset("masks", data=masks)


class TestVideoProcessor:
    """End-to-end smoke tests for VideoProcessor.process_videos."""

    def setup_method(self):
        """Create temp dir, synthetic videos, and install fake extractor / audio sync."""
        self.tmpdir = tempfile.mkdtemp(prefix="test_video_proc_")
        self.num_frames = 60

        self.teacher_video = os.path.join(self.tmpdir, "teacher.mp4")
        self.student_video = os.path.join(self.tmpdir, "student.mp4")
        _create_video(self.teacher_video, self.num_frames, color=(80, 40, 40))
        _create_video(self.student_video, self.num_frames, color=(40, 40, 80))

        num_frames = self.num_frames

        def fake_extract(video_path, output_dir, label, *args, **kwargs):
            """Stand in for Extractor.data_extraction by writing synthetic HDF5."""
            _write_synthetic_hdf5(output_dir, label, num_frames)

        def fake_offset(*args, **kwargs):
            """Stand in for AudioSync.compute_audio_offset."""
            return 0

        self._orig_extract = video_processor_module.Extractor.data_extraction
        self._orig_offset = audio_sync_module.AudioSync.compute_audio_offset
        video_processor_module.Extractor.data_extraction = staticmethod(fake_extract)
        audio_sync_module.AudioSync.compute_audio_offset = staticmethod(fake_offset)

    def teardown_method(self):
        """Restore originals and delete temp dir."""
        video_processor_module.Extractor.data_extraction = staticmethod(self._orig_extract)
        audio_sync_module.AudioSync.compute_audio_offset = staticmethod(self._orig_offset)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_smoke_returns_full_result_dict(self):
        """The pipeline should return a dict with scores, feedback, and video filename."""
        result = asyncio.run(
            VideoProcessor.process_videos(
                self.teacher_video,
                self.student_video,
                self.tmpdir,
            )
        )

        print("=== VideoProcessor Smoke ===")
        print(f"  result keys: {sorted(result.keys()) if result else None}")
        assert result is not None
        for key in (
            "overall_score",
            "skeleton_score",
            "trajectory_score",
            "mask_score",
            "feedback",
            "timeline_markers",
            "feedback_video",
        ):
            assert key in result, f"missing key: {key}"
        assert isinstance(result["feedback"], list) and len(result["feedback"]) > 0
        assert os.path.isfile(os.path.join(self.tmpdir, result["feedback_video"]))
        print("  PASSED\n")

    def test_missing_video_returns_none(self):
        """A missing input file should short-circuit and return None."""
        result = asyncio.run(
            VideoProcessor.process_videos(
                os.path.join(self.tmpdir, "does_not_exist.mp4"),
                self.student_video,
                self.tmpdir,
            )
        )

        print("=== VideoProcessor Missing Input ===")
        print(f"  result: {result}")
        assert result is None
        print("  PASSED\n")

    def test_event_handler_receives_status_updates(self):
        """The async event handler should be invoked with status strings."""
        messages: list = []

        async def handler(msg: str) -> None:
            messages.append(msg)

        asyncio.run(
            VideoProcessor.process_videos(
                self.teacher_video,
                self.student_video,
                self.tmpdir,
                handler,
            )
        )

        print("=== VideoProcessor Event Handler ===")
        print(f"  messages: {messages[:3]}... ({len(messages)} total)")
        assert len(messages) > 0
        assert any("Processing" in m for m in messages)
        print("  PASSED\n")
