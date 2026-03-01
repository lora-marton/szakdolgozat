"""
Preprocessing orchestrator for dance comparison.

Synchronises and trims two dance sequences before DTW alignment:
1. Audio cross-correlation  → find frame offset between the videos
2. Apply offset             → shift/trim the leading video
3. Motion energy detection  → find active range in each sequence
4. Intersection             → keep only the overlapping active region
"""
import numpy as np

from model.preprocessing.audio_sync import compute_audio_offset
from model.preprocessing.motion_energy import compute_motion_energy, find_active_range


def preprocess(teacher_data, student_data, teacher_video, student_video, config=None):
    """
    Synchronise and trim teacher/student data to the shared active dance region.

    Args:
        teacher_data: Dict with 'landmarks', 'masks', 'trajectory' arrays + 'fps'.
        student_data: Same structure as teacher_data.
        teacher_video: Path to the teacher video file (for audio extraction).
        student_video: Path to the student video file (for audio extraction).
        config: PreprocessorConfig instance (uses defaults if None).

    Returns:
        (teacher_data, student_data): Trimmed copies of both data dicts,
            each containing only the frames within the shared active region.
    """
    from model.config import DEFAULT_PREPROCESSOR_CONFIG
    if config is None:
        config = DEFAULT_PREPROCESSOR_CONFIG

    fps = teacher_data.get('fps', 60.0)

    # ------------------------------------------------------------------
    # Step 1: Audio cross-correlation → frame offset
    # ------------------------------------------------------------------
    offset = compute_audio_offset(
        teacher_video, student_video,
        target_fps=fps,
        sr=config.audio_sample_rate,
    )
    print(f"[Preprocessor] Audio offset: {offset} frames "
          f"({'teacher leads' if offset > 0 else 'student leads' if offset < 0 else 'in sync'})")

    # ------------------------------------------------------------------
    # Step 2: Apply offset — trim leading frames from the earlier video
    # ------------------------------------------------------------------
    teacher_data, student_data = _apply_offset(teacher_data, student_data, offset)

    # ------------------------------------------------------------------
    # Step 3: Motion energy → active range for each dancer
    # ------------------------------------------------------------------
    t_energy = compute_motion_energy(teacher_data['landmarks'])
    s_energy = compute_motion_energy(student_data['landmarks'])

    t_start, t_end = find_active_range(
        t_energy, config.motion_threshold_ratio, config.min_active_duration,
    )
    s_start, s_end = find_active_range(
        s_energy, config.motion_threshold_ratio, config.min_active_duration,
    )

    print(f"[Preprocessor] Teacher active range: frames {t_start}–{t_end}")
    print(f"[Preprocessor] Student active range: frames {s_start}–{s_end}")

    # ------------------------------------------------------------------
    # Step 4: Intersection — keep only the overlapping active region
    # ------------------------------------------------------------------
    shared_start = max(t_start, s_start)
    shared_end = min(t_end, s_end)

    if shared_start >= shared_end:
        print("[Preprocessor] WARNING: No overlapping active region found. "
              "Skipping trimming.")
        return teacher_data, student_data

    print(f"[Preprocessor] Shared active region: frames {shared_start}–{shared_end} "
          f"({shared_end - shared_start} frames)")

    teacher_data = _slice_data(teacher_data, shared_start, shared_end)
    student_data = _slice_data(student_data, shared_start, shared_end)

    return teacher_data, student_data


# ── Helpers ──────────────────────────────────────────────────────────────


def _apply_offset(teacher_data, student_data, offset):
    """
    Trim the leading frames from the earlier video so both start at the same
    musical moment, then truncate to the shorter length.
    """
    if offset > 0:
        # Teacher leads → trim first `offset` frames from teacher
        teacher_data = _slice_data(teacher_data, offset, None)
    elif offset < 0:
        # Student leads → trim first `|offset|` frames from student
        student_data = _slice_data(student_data, -offset, None)

    # Truncate to the shorter sequence
    min_len = min(
        len(teacher_data['landmarks']),
        len(student_data['landmarks']),
    )
    teacher_data = _slice_data(teacher_data, 0, min_len)
    student_data = _slice_data(student_data, 0, min_len)

    return teacher_data, student_data


def _slice_data(data, start, end):
    """
    Return a copy of the data dict with all arrays sliced to [start:end].
    Non-array entries (e.g. 'fps', 'fixed_scale') are preserved as-is.
    """
    sliced = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced
