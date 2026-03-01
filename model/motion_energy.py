"""
Motion energy detection for finding the active dance range in a sequence.

Computes per-frame motion energy from pose landmarks and identifies
the first/last frame of sustained movement, trimming idle periods.
"""
import numpy as np


def compute_motion_energy(landmarks):
    """
    Compute per-frame motion energy from pose landmarks.

    Motion energy is the average Euclidean displacement of all joints
    between consecutive frames.

    Args:
        landmarks: Array of shape (T, 33, 4) — [x, y, z, visibility] per landmark.

    Returns:
        energy: Array of shape (T-1,) with per-frame motion energy.
    """
    positions = landmarks[:, :, :3]  # (T, 33, 3) — drop visibility
    deltas = np.diff(positions, axis=0)  # (T-1, 33, 3)
    per_joint = np.sqrt((deltas ** 2).sum(axis=-1))  # (T-1, 33)
    energy = per_joint.mean(axis=-1)  # (T-1,)
    return energy


def find_active_range(energy, threshold_ratio=0.15, min_duration_frames=10):
    """
    Find the first and last frame of sustained movement.

    Scans forward/backward through the motion energy signal to find
    the boundaries where sustained activity begins and ends.

    Args:
        energy: Array of shape (T-1,) — per-frame motion energy.
        threshold_ratio: Fraction of max energy used as the activity threshold.
        min_duration_frames: Number of consecutive frames that must be above
            threshold to count as sustained movement.

    Returns:
        (start, end): Tuple of frame indices (inclusive) marking the active range.
            These indices refer to the original landmark array, not the energy array.
    """
    threshold = energy.max() * threshold_ratio
    active = energy > threshold

    required_active = int(min_duration_frames * 0.7)

    # Scan forward for start
    start = 0
    for i in range(len(active) - min_duration_frames):
        window = active[i:i + min_duration_frames]
        if window.sum() >= required_active:
            start = i
            break

    # Scan backward for end
    end = len(energy)  # energy has T-1 elements → index T-1 maps to landmark frame T
    for i in range(len(active) - 1, min_duration_frames - 1, -1):
        window = active[i - min_duration_frames + 1:i + 1]
        if window.sum() >= required_active:
            end = i + 1  # +1 because energy[i] spans landmark frames i and i+1
            break

    return start, end
