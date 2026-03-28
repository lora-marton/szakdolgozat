"""
Motion energy detection for finding the active dance range in a sequence.

Computes per-frame motion energy from pose landmarks and identifies
the first/last frame of sustained movement, trimming idle periods.
"""
import numpy as np


class MotionEnergy:
    """Detects the active dance region in a landmark sequence using motion energy."""

    @staticmethod
    def compute_motion_energy(landmarks: np.ndarray) -> np.ndarray:
        """
        Compute per-frame motion energy from pose landmarks.

        Motion energy is the average Euclidean displacement of all joints
        between consecutive frames.

        Args:
            landmarks: Array of shape (T, 33, 4) with x, y, z, visibility per landmark.

        Returns:
            Array of shape (T-1,) with per-frame motion energy.
        """
        positions = landmarks[:, :, :3]
        deltas = np.diff(positions, axis=0)
        per_joint = np.sqrt((deltas ** 2).sum(axis=-1))
        energy = per_joint.mean(axis=-1)
        return energy

    @staticmethod
    def find_active_range(
        energy: np.ndarray,
        threshold_ratio: float = 0.15,
        min_duration_frames: int = 10,
        active_window_ratio: float = 0.7,
    ) -> tuple[int, int]:
        """
        Find the first and last frame of sustained movement.

        Scans forward and backward through the motion energy signal to find
        the boundaries where sustained activity begins and ends.

        Args:
            energy: Array of shape (T-1,) with per-frame motion energy.
            threshold_ratio: Fraction of max energy used as the activity threshold.
            min_duration_frames: Number of consecutive frames that must be above
                threshold to count as sustained movement.
            active_window_ratio: Fraction of frames within the sliding window that
                must be above threshold to count as sustained movement.

        Returns:
            Tuple of (start, end) frame indices marking the active range.
            These indices refer to the original landmark array, not the energy array.
        """
        threshold = energy.max() * threshold_ratio
        active = energy > threshold

        required_active = int(min_duration_frames * active_window_ratio)

        start = 0
        for i in range(len(active) - min_duration_frames):
            window = active[i:i + min_duration_frames]
            if window.sum() >= required_active:
                start = i
                break

        end = len(energy)
        for i in range(len(active) - 1, min_duration_frames - 1, -1):
            window = active[i - min_duration_frames + 1:i + 1]
            if window.sum() >= required_active:
                end = i + 1
                break

        return start, end
