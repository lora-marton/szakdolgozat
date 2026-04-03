"""
Side-by-side skeleton feedback video renderer.

Generates an MP4 showing teacher and student videos side-by-side with
skeleton overlays. Uses only audio offset to sync the videos (no DTW),
so the viewer can see timing differences between the dancers.
"""
import os
import subprocess

import cv2
import h5py
import numpy as np

from model.config import DEFAULT_EXTRACTION_CONFIG


class VideoFeedback:
    """Side-by-side skeleton overlay video renderer."""

    @staticmethod
    def render_video(
        teacher_video: str,
        student_video: str,
        output_dir: str,
        preprocess_info: dict | None = None,
        config=None,
    ) -> str:
        """Generate a side-by-side comparison video with skeleton overlays.

        Uses only the audio offset to sync both videos — no DTW alignment.
        The output shows teacher (left) and student (right) with skeleton
        overlays and a teacher ghost skeleton on the student side.

        Args:
            teacher_video: Path to the teacher video file.
            student_video: Path to the student video file.
            output_dir: Session directory containing HDF5 data.
            preprocess_info: Dict with 'audio_offset'. If None, no offset applied.
            config: ExtractionConfig instance (uses default if None).

        Returns:
            Path to the generated MP4 file.
        """
        if config is None:
            config = DEFAULT_EXTRACTION_CONFIG
        if preprocess_info is None:
            preprocess_info = {'audio_offset': 0}

        connections = config.pose_connections
        target_fps = config.target_fps
        output_fps = config.output_fps
        audio_offset = preprocess_info.get('audio_offset', 0)

        teacher_lm = VideoFeedback._load_landmarks(output_dir, 'teacher')
        student_lm = VideoFeedback._load_landmarks(output_dir, 'student')

        print("[VideoFeedback] Reading teacher video frames...")
        teacher_frames, teacher_src_fps = VideoFeedback._read_resampled_frames(teacher_video, target_fps)
        print(f"[VideoFeedback] Teacher: {len(teacher_frames)} resampled frames (source: {teacher_src_fps} fps)")

        print("[VideoFeedback] Reading student video frames...")
        student_frames, student_src_fps = VideoFeedback._read_resampled_frames(student_video, target_fps)
        print(f"[VideoFeedback] Student: {len(student_frames)} resampled frames (source: {student_src_fps} fps)")

        source_fps = min(teacher_src_fps, student_src_fps)
        teacher_frames, teacher_lm, student_frames, student_lm, num_frames = (
            VideoFeedback._apply_audio_offset(
                teacher_frames, teacher_lm,
                student_frames, student_lm,
                audio_offset, target_fps, source_fps,
            )
        )

        dimensions = VideoFeedback._compute_output_dimensions(
            teacher_frames[0], student_frames[0],
        )

        temp_path = os.path.join(output_dir, 'feedback_video_temp.mp4')
        VideoFeedback._render_frames(
            teacher_frames, student_frames,
            teacher_lm, student_lm,
            connections, dimensions, num_frames,
            output_fps, temp_path,
        )

        del teacher_frames
        del student_frames

        output_path = os.path.join(output_dir, 'feedback_video.mp4')
        output_path = VideoFeedback._reencode_to_h264(temp_path, output_path)

        print(f"[VideoFeedback] Done! Saved to {output_path}")
        return output_path

    @staticmethod
    def _load_landmarks(output_dir: str, label: str) -> np.ndarray:
        """Load raw landmarks array from session HDF5 file.

        Args:
            output_dir: Session directory containing HDF5 files.
            label: Either 'teacher' or 'student'.

        Returns:
            Landmarks array of shape (N, 33, 4).
        """
        data_path = os.path.join(output_dir, f'{label}_data.h5')
        with h5py.File(data_path, 'r') as f:
            return f['raw'][:]

    @staticmethod
    def _read_resampled_frames(
        video_path: str,
        target_fps: float,
    ) -> tuple[list, float]:
        """Read all frames from a video with FPS resampling matching the extractor.

        Args:
            video_path: Path to the video file.
            target_fps: Target FPS for resampling.

        Returns:
            Tuple of (list of BGR frames, source FPS).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval_ms = 1000.0 / target_fps
        last_processed_time = -frame_interval_ms
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            timestamp_ms = (frame_idx * 1000.0) / source_fps

            if timestamp_ms < last_processed_time + frame_interval_ms - (1000.0 / source_fps / 2):
                continue
            last_processed_time += frame_interval_ms
            frames.append(frame)

        cap.release()
        return frames, source_fps

    @staticmethod
    def _apply_audio_offset(
        teacher_frames: list,
        teacher_lm: np.ndarray,
        student_frames: list,
        student_lm: np.ndarray,
        audio_offset: float,
        target_fps: float,
        source_fps: float,
    ) -> tuple[list, np.ndarray, list, np.ndarray, int]:
        """Trim leading frames based on audio sync offset.

        Args:
            teacher_frames: List of teacher BGR frames.
            teacher_lm: Teacher landmarks array.
            student_frames: List of student BGR frames.
            student_lm: Student landmarks array.
            audio_offset: Offset in target FPS units (positive = teacher leads).
            target_fps: Extraction target FPS.
            source_fps: Source video FPS.

        Returns:
            Tuple of (teacher_frames, teacher_lm, student_frames, student_lm, num_frames).
        """
        scale = source_fps / target_fps
        trim_frames = int(round(abs(audio_offset) * scale))

        print(f"[VideoFeedback] Audio offset: {audio_offset} (at {target_fps}fps) "
              f"= {trim_frames} frames (at {source_fps}fps)")

        if audio_offset > 0:
            teacher_frames = teacher_frames[trim_frames:]
            teacher_lm = teacher_lm[trim_frames:]
            print(f"[VideoFeedback] Trimmed {trim_frames} leading teacher frames")
        elif audio_offset < 0:
            student_frames = student_frames[trim_frames:]
            student_lm = student_lm[trim_frames:]
            print(f"[VideoFeedback] Trimmed {trim_frames} leading student frames")

        num_frames = min(len(teacher_frames), len(student_frames))
        print(f"[VideoFeedback] Playing {num_frames} aligned frames")

        return teacher_frames, teacher_lm, student_frames, student_lm, num_frames

    @staticmethod
    def _compute_output_dimensions(
        teacher_frame: np.ndarray,
        student_frame: np.ndarray,
    ) -> dict:
        """Compute side-by-side output dimensions capped at 720p height.

        Args:
            teacher_frame: First teacher BGR frame.
            student_frame: First student BGR frame.

        Returns:
            Dict with 't_h', 't_w', 's_h', 's_w', 'new_t_w', 'new_s_w',
            'out_w', 'out_h'.
        """
        t_h, t_w = teacher_frame.shape[:2]
        s_h, s_w = student_frame.shape[:2]

        common_h = min(t_h, s_h, 720)
        t_scale = common_h / t_h
        s_scale = common_h / s_h
        new_t_w = int(t_w * t_scale)
        new_s_w = int(s_w * s_scale)

        return {
            't_h': t_h, 't_w': t_w,
            's_h': s_h, 's_w': s_w,
            'new_t_w': new_t_w, 'new_s_w': new_s_w,
            'out_w': new_t_w + new_s_w, 'out_h': common_h,
        }

    @staticmethod
    def _render_frames(
        teacher_frames: list,
        student_frames: list,
        teacher_lm: np.ndarray,
        student_lm: np.ndarray,
        connections: tuple,
        dimensions: dict,
        num_frames: int,
        output_fps: float,
        temp_path: str,
    ) -> None:
        """Render the composited side-by-side video with skeleton overlays.

        Args:
            teacher_frames: List of teacher BGR frames.
            student_frames: List of student BGR frames.
            teacher_lm: Teacher landmarks array.
            student_lm: Student landmarks array.
            connections: Skeleton connection pairs.
            dimensions: Dict from _compute_output_dimensions.
            num_frames: Number of frames to render.
            output_fps: Output video FPS.
            temp_path: Path for the temporary output file.
        """
        t_w = dimensions['t_w']
        t_h = dimensions['t_h']
        s_w = dimensions['s_w']
        s_h = dimensions['s_h']
        new_t_w = dimensions['new_t_w']
        new_s_w = dimensions['new_s_w']
        out_w = dimensions['out_w']
        out_h = dimensions['out_h']

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, output_fps, (out_w, out_h))

        if not out.isOpened():
            raise RuntimeError(f"Could not create video writer at {temp_path}")

        for i in range(num_frames):
            if i % 100 == 0:
                print(f"[VideoFeedback] Frame {i}/{num_frames}")

            teacher_frame = teacher_frames[i].copy()
            student_frame = student_frames[i].copy()

            if i < len(teacher_lm):
                VideoFeedback._draw_skeleton(
                    teacher_frame, teacher_lm[i], connections, t_w, t_h,
                    line_color=(255, 240, 0),
                    point_color=(255, 240, 0),
                )

            if i < len(student_lm):
                VideoFeedback._draw_skeleton(
                    student_frame, student_lm[i], connections, s_w, s_h,
                    line_color=(0, 255, 170),
                    point_color=(0, 255, 170),
                )

            if i < len(teacher_lm):
                VideoFeedback._draw_skeleton(
                    student_frame, teacher_lm[i], connections, s_w, s_h,
                    line_color=(255, 240, 0),
                    point_color=(255, 240, 0),
                    line_thickness=3, point_radius=5,
                )

            teacher_resized = cv2.resize(teacher_frame, (new_t_w, out_h))
            student_resized = cv2.resize(student_frame, (new_s_w, out_h))

            VideoFeedback._draw_label(teacher_resized, 'Teacher')
            VideoFeedback._draw_label(student_resized, 'Student')

            composite = np.hstack([teacher_resized, student_resized])
            out.write(composite)

        out.release()
        print("[VideoFeedback] Raw video written. Re-encoding to H.264...")

    @staticmethod
    def _draw_skeleton(
        frame: np.ndarray,
        landmarks: np.ndarray,
        connections: tuple,
        vid_w: int,
        vid_h: int,
        line_color: tuple = (0, 255, 0),
        point_color: tuple = (0, 0, 255),
        line_thickness: int = 5,
        point_radius: int = 7,
    ) -> None:
        """Draw a skeleton on a frame using normalized 0-1 landmarks.

        Args:
            frame: BGR image (modified in-place).
            landmarks: Array of shape (33, 4) with x, y, z, visibility.
            connections: Iterable of (idx1, idx2) pairs.
            vid_w: Frame width for denormalization.
            vid_h: Frame height for denormalization.
            line_color: BGR tuple for bones.
            point_color: BGR tuple for joints.
            line_thickness: Bone line width.
            point_radius: Joint circle radius.
        """
        for p1, p2 in connections:
            if landmarks[p1][3] > 0.5 and landmarks[p2][3] > 0.5:
                pt1 = (int(landmarks[p1][0] * vid_w), int(landmarks[p1][1] * vid_h))
                pt2 = (int(landmarks[p2][0] * vid_w), int(landmarks[p2][1] * vid_h))
                cv2.line(frame, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)

        for i in range(33):
            if landmarks[i][3] > 0.5:
                cx = int(landmarks[i][0] * vid_w)
                cy = int(landmarks[i][1] * vid_h)
                cv2.circle(frame, (cx, cy), point_radius, point_color, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_label(frame: np.ndarray, text: str) -> None:
        """Draw a semi-transparent label bar at the top of the frame.

        Args:
            frame: BGR image (modified in-place).
            text: Label text to display.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(
            frame, text, (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    @staticmethod
    def _reencode_to_h264(input_path: str, output_path: str) -> str:
        """Re-encode a video to H.264 for browser compatibility.

        Uses ffmpeg bundled with imageio-ffmpeg. Falls back to the mp4v
        version if ffmpeg is unavailable or encoding fails.

        Args:
            input_path: Path to the mp4v encoded temp file.
            output_path: Desired output path for the H.264 file.

        Returns:
            Path to the final output file.
        """
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            print("[VideoFeedback] imageio-ffmpeg not available, skipping H.264 re-encode.")
            return input_path

        try:
            subprocess.run(
                [
                    ffmpeg_exe,
                    '-y',
                    '-i', input_path,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    output_path,
                ],
                check=True,
                capture_output=True,
            )
            os.remove(input_path)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"[VideoFeedback] ffmpeg re-encode failed: {e.stderr.decode()}")
            os.rename(input_path, output_path)
            return output_path
