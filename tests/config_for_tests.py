"""
Shared helpers for the model unit tests.

Imports of this module ensure the project root is on ``sys.path`` so that
``from model...`` imports work regardless of where pytest is invoked from.
Also provides small synthetic-data builders reused across test files.
"""

import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def make_stick_figure_landmarks(num_frames: int = 30, visibility: float = 0.95) -> np.ndarray:
    """Create a synthetic 33-landmark pose sequence arranged as a static stick figure.

    All frames share the same pose so any comparison of the result with itself
    should produce a near-perfect score.

    Args:
        num_frames: Number of frames to generate.
        visibility: Fixed visibility value for every landmark.

    Returns:
        Array of shape (num_frames, 33, 4) with [x, y, z, visibility].
    """
    landmarks = np.zeros((num_frames, 33, 4), dtype=np.float32)

    base = {
        0: (0.50, 0.18),
        11: (0.42, 0.30),
        12: (0.58, 0.30),
        13: (0.38, 0.42),
        14: (0.62, 0.42),
        15: (0.34, 0.54),
        16: (0.66, 0.54),
        17: (0.33, 0.56),
        18: (0.67, 0.56),
        19: (0.33, 0.57),
        20: (0.67, 0.57),
        21: (0.34, 0.55),
        22: (0.66, 0.55),
        23: (0.44, 0.58),
        24: (0.56, 0.58),
        25: (0.44, 0.74),
        26: (0.56, 0.74),
        27: (0.44, 0.90),
        28: (0.56, 0.90),
        29: (0.43, 0.92),
        30: (0.57, 0.92),
        31: (0.46, 0.93),
        32: (0.54, 0.93),
    }

    for i in range(33):
        x, y = base.get(i, (0.50, 0.50))
        landmarks[:, i, 0] = x
        landmarks[:, i, 1] = y
        landmarks[:, i, 2] = 0.0
        landmarks[:, i, 3] = visibility

    return landmarks


def make_trajectory(num_frames: int = 30, start=(100.0, 200.0), velocity=(1.0, 0.5)) -> np.ndarray:
    """Create a linear 2D trajectory.

    Args:
        num_frames: Number of frames.
        start: Starting (x, y) position.
        velocity: Per-frame (vx, vy) displacement.

    Returns:
        Array of shape (num_frames, 2).
    """
    t = np.arange(num_frames, dtype=np.float32)[:, None]
    return np.array(start, dtype=np.float32) + t * np.array(velocity, dtype=np.float32)


def make_masks(num_frames: int = 30, h: int = 64, w: int = 64, radius: int = 18) -> np.ndarray:
    """Create a sequence of identical circular masks as uint8 (0 / 255).

    Args:
        num_frames: Number of frames.
        h: Mask height.
        w: Mask width.
        radius: Radius of the filled circle at the center.

    Returns:
        Array of shape (num_frames, h, w) with dtype uint8.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
    frame = (disk.astype(np.uint8)) * 255
    return np.broadcast_to(frame, (num_frames, h, w)).copy()
