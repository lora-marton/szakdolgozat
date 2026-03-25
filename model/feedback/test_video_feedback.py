"""
Test for the video feedback renderer.

Creates synthetic video files + HDF5 data and verifies that
generate_feedback_video produces a valid MP4.
"""
import os
import sys
import tempfile
import shutil
import numpy as np
import cv2
import h5py
import pytest

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.feedback.video_feedback import generate_feedback_video


def _create_synthetic_video(path, num_frames=30, width=320, height=240, fps=30.0,
                            color=(50, 50, 50)):
    """Create a solid-color test video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for _ in range(num_frames):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _create_synthetic_h5(output_dir, label, num_frames=30):
    """Create fake HDF5 landmark + mask data."""
    # Landmarks: 33 joints with (x, y, z, visibility) in normalized coords
    landmarks = np.random.rand(num_frames, 33, 4).astype(np.float32)
    landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0.1, 0.9)  # x in [0.1, 0.9]
    landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0.1, 0.9)  # y in [0.1, 0.9]
    landmarks[:, :, 3] = 0.9  # all visible

    trajectory = np.random.rand(num_frames, 2).astype(np.float32) * 100

    data_path = os.path.join(output_dir, f'{label}_data.h5')
    with h5py.File(data_path, 'w') as f:
        f.create_dataset('raw', data=landmarks)
        f.create_dataset('trajectory', data=trajectory)
        f.attrs['fps'] = 60.0
        f.attrs['fixed_scale'] = 1.0

    mask_path = os.path.join(output_dir, f'{label}_masks.h5')
    masks = np.zeros((num_frames, 256, 256), dtype=np.uint8)
    with h5py.File(mask_path, 'w') as f:
        f.create_dataset('masks', data=masks)


class TestVideoFeedback:
    """Test suite for generate_feedback_video."""

    def setup_method(self):
        """Create temp directory with synthetic test data."""
        self.tmpdir = tempfile.mkdtemp(prefix='test_video_feedback_')
        self.num_frames = 30

        # Create synthetic videos
        self.teacher_video = os.path.join(self.tmpdir, 'teacher.mp4')
        self.student_video = os.path.join(self.tmpdir, 'student.mp4')
        _create_synthetic_video(self.teacher_video, self.num_frames, color=(80, 40, 40))
        _create_synthetic_video(self.student_video, self.num_frames, color=(40, 40, 80))

        # Create synthetic HDF5 data
        _create_synthetic_h5(self.tmpdir, 'teacher', self.num_frames)
        _create_synthetic_h5(self.tmpdir, 'student', self.num_frames)

        # 1:1 alignment
        self.alignment_path = [(i, i) for i in range(self.num_frames)]

    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_output_file_created(self):
        """Verify that the output MP4 file is created."""
        output = generate_feedback_video(
            self.teacher_video, self.student_video, self.tmpdir,
            self.alignment_path,
        )
        assert os.path.isfile(output)
        assert output.endswith('.mp4')

    def test_output_has_correct_frames(self):
        """Verify the output video has the expected number of frames."""
        output = generate_feedback_video(
            self.teacher_video, self.student_video, self.tmpdir,
            self.alignment_path,
        )
        cap = cv2.VideoCapture(output)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frame_count == self.num_frames

    def test_output_is_side_by_side(self):
        """Verify the output frame width is the sum of both input widths."""
        output = generate_feedback_video(
            self.teacher_video, self.student_video, self.tmpdir,
            self.alignment_path,
        )
        cap = cv2.VideoCapture(output)
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        # Both are 320px wide, same height → side-by-side = 640
        assert out_w == 640

    def test_mismatched_alignment_shorter(self):
        """Verify it handles fewer alignment pairs than video frames."""
        short_path = [(i, i) for i in range(10)]
        output = generate_feedback_video(
            self.teacher_video, self.student_video, self.tmpdir,
            short_path,
        )
        cap = cv2.VideoCapture(output)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frame_count == 10
