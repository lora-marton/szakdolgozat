"""
Audio cross-correlation for temporal synchronisation of two dance videos.

Extracts audio from both videos and computes the lag (in frames)
between them so downstream steps can align the sequences.
"""
import numpy as np
from scipy.signal import correlate, correlation_lags, resample


def _load_audio_from_video(video_path, sr=22050):
    """
    Extract mono audio from a video file as a numpy array.

    Uses moviepy (which bundles FFmpeg via imageio-ffmpeg),
    so no system FFmpeg install is needed.

    Args:
        video_path: Path to the video file.
        sr: Desired sample rate for the output audio.

    Returns:
        audio: 1D numpy array of float32 audio samples.
    """
    from moviepy import VideoFileClip

    clip = VideoFileClip(video_path)

    if clip.audio is None:
        clip.close()
        return np.array([], dtype=np.float32)

    # Extract audio at its native sample rate
    audio_array = clip.audio.to_soundarray(fps=sr)
    clip.close()

    # Convert to mono (average channels)
    if audio_array.ndim == 2:
        mono = audio_array.mean(axis=1)
    else:
        mono = audio_array

    return mono.astype(np.float32)


def compute_audio_offset(video1_path, video2_path, target_fps=60.0, sr=22050):
    """
    Find the frame offset between two videos using audio cross-correlation.

    Args:
        video1_path: Path to the first video file (treated as reference).
        video2_path: Path to the second video file.
        target_fps: Frame rate used by the extraction pipeline.
        sr: Audio sample rate for analysis.

    Returns:
        offset_frames: Integer frame offset.
            Positive → video2 starts LATER (video1 has extra leading frames).
            Negative → video1 starts LATER (video2 has extra leading frames).
    """
    audio1 = _load_audio_from_video(video1_path, sr=sr)
    audio2 = _load_audio_from_video(video2_path, sr=sr)

    if len(audio1) == 0 or len(audio2) == 0:
        print("[AudioSync] WARNING: Could not extract audio. Assuming zero offset.")
        return 0

    # Full cross-correlation
    correlation = correlate(audio1, audio2, mode='full')
    lags = correlation_lags(len(audio1), len(audio2), mode='full')

    best_lag_samples = lags[np.argmax(correlation)]

    # Convert sample lag → seconds → frames
    offset_seconds = best_lag_samples / sr
    offset_frames = int(round(offset_seconds * target_fps))

    return offset_frames
