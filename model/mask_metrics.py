"""
Mask-based comparison metrics: IoU and shape analysis.
"""
import numpy as np


def compute_iou(teacher_masks, student_masks):
    """
    Calculate Intersection over Union for aligned mask pairs.

    Args:
        teacher_masks: Array of shape (N, H, W) — teacher masks (uint8, 0-255).
        student_masks: Array of shape (N, H, W) — student masks (DTW-aligned).

    Returns:
        mean_iou: Float 0-1 — average IoU across all frames.
        per_frame_iou: Array of shape (N,) — IoU per frame.
    """
    # TODO: Implement IoU calculation
    raise NotImplementedError("Mask IoU not yet implemented")
