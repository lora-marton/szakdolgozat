"""
Audio cross-correlation for temporal synchronisation of two dance videos.

Extracts audio from both videos and computes the lag (in frames)
between them so downstream steps can align the sequences.
"""
import numpy as np
import librosa
from scipy.signal import correlate, correlation_lags


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
    audio1, _ = librosa.load(video1_path, sr=sr, mono=True)
    audio2, _ = librosa.load(video2_path, sr=sr, mono=True)

    # Full cross-correlation
    correlation = correlate(audio1, audio2, mode='full')
    lags = correlation_lags(len(audio1), len(audio2), mode='full')

    best_lag_samples = lags[np.argmax(correlation)]

    # Convert sample lag → seconds → frames
    offset_seconds = best_lag_samples / sr
    offset_frames = int(round(offset_seconds * target_fps))

    return offset_frames
