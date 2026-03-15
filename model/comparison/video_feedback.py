"""
Side-by-side skeleton feedback video renderer.

Generates an MP4 showing teacher and student videos side-by-side with skeleton
overlays. Uses only audio offset to sync the videos (no DTW), so the viewer
can see timing differences between the dancers.
"""
import os
import subprocess
import cv2
import numpy as np
import h5py

from model.config import DEFAULT_CONFIG


# ── Skeleton drawing helpers ─────────────────────────────────────────────

def _draw_skeleton(frame, landmarks, connections, vid_w, vid_h,
                   line_color=(0, 255, 0), point_color=(0, 0, 255),
                   line_thickness=4, point_radius=6):
    """
    Draw a skeleton on a frame using normalized (0-1) landmarks.

    Args:
        frame: BGR image (modified in-place).
        landmarks: Array of shape (33, 4) -- [x, y, z, visibility].
        connections: Iterable of (idx1, idx2) pairs.
        vid_w, vid_h: Frame dimensions for denormalization.
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


def _draw_label(frame, text):
    """Draw a semi-transparent label bar at the top of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(
        frame, text, (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )


# ── Sequential frame reader ─────────────────────────────────────────────

def _read_resampled_frames(video_path, target_fps):
    """
    Read all frames from a video, applying the same FPS resampling as the
    extractor. Returns a list of BGR frames indexed identically to the HDF5
    landmark data, plus the source FPS.

    Returns:
        (frames, source_fps): List of BGR frames and the video's source FPS.
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

        # Same resampling logic as extractor.py
        if timestamp_ms < last_processed_time + frame_interval_ms - (1000.0 / source_fps / 2):
            continue
        last_processed_time += frame_interval_ms
        frames.append(frame)

    cap.release()
    return frames, source_fps


# ── H.264 re-encoding ───────────────────────────────────────────────────

def _reencode_to_h264(input_path, output_path):
    """
    Re-encode a video to H.264 (browser-compatible) using ffmpeg.
    Uses the ffmpeg bundled with imageio-ffmpeg (a moviepy dependency).
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
                '-y',                       # overwrite
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-pix_fmt', 'yuv420p',       # browser compatibility
                '-movflags', '+faststart',   # streaming-friendly
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        # Remove the temp file
        os.remove(input_path)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"[VideoFeedback] ffmpeg re-encode failed: {e.stderr.decode()}")
        # Fall back to the mp4v version
        os.rename(input_path, output_path)
        return output_path


# ── Main renderer ────────────────────────────────────────────────────────

def generate_feedback_video(teacher_video, student_video, output_dir,
                            preprocess_info=None, config=None):
    """
    Generate a side-by-side comparison video with skeleton overlays.

    Uses only the audio offset to sync both videos — no DTW alignment.
    This allows natural playback where the viewer can see timing differences.

    The output video shows teacher (left) and student (right) with:
    - Green skeleton on both dancers
    - Blue ghost skeleton of the teacher overlaid on the student's frames

    Args:
        teacher_video: Path to the teacher video file.
        student_video: Path to the student video file.
        output_dir: Session directory containing HDF5 data and where the
                     output video will be saved.
        preprocess_info: Dict with 'audio_offset' (frames, positive = teacher
                         leads). If None, no offset is applied.
        config: ExtractionConfig instance (uses DEFAULT_CONFIG if None).

    Returns:
        output_path: Path to the generated MP4 file.
    """
    if config is None:
        config = DEFAULT_CONFIG
    if preprocess_info is None:
        preprocess_info = {'audio_offset': 0}

    connections = config.pose_connections
    target_fps = config.target_fps
    audio_offset = preprocess_info.get('audio_offset', 0)

    # ── Step 1: Load landmark data from HDF5 ─────────────────────────
    teacher_lm = _load_landmarks(output_dir, 'teacher')
    student_lm = _load_landmarks(output_dir, 'student')

    # ── Step 2: Read all video frames with FPS resampling ────────────
    print("[VideoFeedback] Reading teacher video frames...")
    teacher_frames, teacher_src_fps = _read_resampled_frames(teacher_video, target_fps)
    print(f"[VideoFeedback] Teacher: {len(teacher_frames)} resampled frames (source: {teacher_src_fps} fps)")

    print("[VideoFeedback] Reading student video frames...")
    student_frames, student_src_fps = _read_resampled_frames(student_video, target_fps)
    print(f"[VideoFeedback] Student: {len(student_frames)} resampled frames (source: {student_src_fps} fps)")

    # ── Step 3: Apply audio offset ───────────────────────────────────
    # audio_offset is in target_fps (60) units — convert to actual
    # resampled frame units using the source FPS
    source_fps = min(teacher_src_fps, student_src_fps)
    scale = source_fps / target_fps  # e.g. 30/60 = 0.5
    trim_frames = int(round(abs(audio_offset) * scale))

    print(f"[VideoFeedback] Audio offset: {audio_offset} (at {target_fps}fps) "
          f"= {trim_frames} frames (at {source_fps}fps)")

    # Trim the leading frames from whichever video starts earlier
    if audio_offset > 0:
        # Teacher leads — trim first frames from teacher
        teacher_frames = teacher_frames[trim_frames:]
        teacher_lm = teacher_lm[trim_frames:]
        print(f"[VideoFeedback] Trimmed {trim_frames} leading teacher frames")
    elif audio_offset < 0:
        # Student leads — trim first frames from student
        student_frames = student_frames[trim_frames:]
        student_lm = student_lm[trim_frames:]
        print(f"[VideoFeedback] Trimmed {trim_frames} leading student frames")

    # Use the shorter of the two
    num_frames = min(len(teacher_frames), len(student_frames))
    print(f"[VideoFeedback] Playing {num_frames} aligned frames")

    # ── Step 4: Determine output dimensions ──────────────────────────
    t_h, t_w = teacher_frames[0].shape[:2]
    s_h, s_w = student_frames[0].shape[:2]

    # Match heights (capped at 720p)
    common_h = min(t_h, s_h, 720)
    t_scale = common_h / t_h
    s_scale = common_h / s_h
    new_t_w = int(t_w * t_scale)
    new_s_w = int(s_w * s_scale)
    out_w = new_t_w + new_s_w
    out_h = common_h

    # ── Step 5: Write composited video ───────────────────────────────
    temp_path = os.path.join(output_dir, 'feedback_video_temp.mp4')
    output_path = os.path.join(output_dir, 'feedback_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = 30.0  # Match source video FPS (hardcoded)
    out = cv2.VideoWriter(temp_path, fourcc, output_fps, (out_w, out_h))

    if not out.isOpened():
        raise RuntimeError(f"Could not create video writer at {temp_path}")

    for i in range(num_frames):
        if i % 100 == 0:
            print(f"[VideoFeedback] Frame {i}/{num_frames}")

        teacher_frame = teacher_frames[i].copy()
        student_frame = student_frames[i].copy()

        # Draw skeleton on teacher  (colours from view/src/theme.ts)
        if i < len(teacher_lm):
            _draw_skeleton(
                teacher_frame, teacher_lm[i], connections, t_w, t_h,
                line_color=(193, 145, 63),
                point_color=(193, 145, 63),
            )

        # Draw skeleton on student  (colours from view/src/theme.ts)
        if i < len(student_lm):
            _draw_skeleton(
                student_frame, student_lm[i], connections, s_w, s_h,
                line_color=(193, 145, 63),
                point_color=(193, 145, 63),
            )

        # Draw ghost (teacher skeleton) on student's frame  (score.good #2a8c62)
        if i < len(teacher_lm):
            _draw_skeleton(
                student_frame, teacher_lm[i], connections, s_w, s_h,
                line_color=(0, 255, 170),
                point_color=(0, 255, 170),
                line_thickness=2, point_radius=4,
            )

        # Resize to common height
        teacher_resized = cv2.resize(teacher_frame, (new_t_w, out_h))
        student_resized = cv2.resize(student_frame, (new_s_w, out_h))

        # Draw labels
        _draw_label(teacher_resized, 'Teacher')
        _draw_label(student_resized, 'Student')

        # Composite side by side
        composite = np.hstack([teacher_resized, student_resized])
        out.write(composite)

    out.release()
    print("[VideoFeedback] Raw video written. Re-encoding to H.264...")

    # Free frame memory before re-encoding
    del teacher_frames
    del student_frames

    # ── Step 6: Re-encode to H.264 for browser compatibility ─────────
    output_path = _reencode_to_h264(temp_path, output_path)

    print(f"[VideoFeedback] Done! Saved to {output_path}")
    return output_path


# ── Utility functions ────────────────────────────────────────────────────

def _load_landmarks(output_dir, label):
    """Load raw landmarks array from session HDF5 file."""
    data_path = os.path.join(output_dir, f'{label}_data.h5')
    with h5py.File(data_path, 'r') as f:
        return f['raw'][:]
