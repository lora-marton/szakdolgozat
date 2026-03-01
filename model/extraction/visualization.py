import cv2
import numpy as np


def draw_mask_overlay(frame, segmentation_mask):
    """
    Draw a yellow semi-transparent mask overlay on the frame.

    Args:
        frame: BGR video frame.
        segmentation_mask: Raw float mask from MediaPipe.

    Returns:
        Blended frame with yellow mask overlay.
    """
    visual_mask = (segmentation_mask * 255).astype(np.uint8)
    visual_mask = cv2.cvtColor(visual_mask, cv2.COLOR_GRAY2BGR)
    visual_mask[:, :, 0] = 0  # Remove blue channel → yellow
    return cv2.addWeighted(frame, 1.0, visual_mask, 0.5, 0)


def draw_skeleton(frame, landmarks, connections, vid_w, vid_h):
    """
    Draw pose skeleton (lines + points) on a frame.

    Args:
        frame: BGR image to draw on (will be modified in-place).
        landmarks: Array of shape (33, 4) with [x, y, z, visibility].
        connections: Iterable of (idx1, idx2) pairs for skeleton edges.
        vid_w: Video width in pixels.
        vid_h: Video height in pixels.

    Returns:
        Frame with skeleton drawn on it.
    """
    # Draw lines
    for p1, p2 in connections:
        if landmarks[p1][3] > 0.5 and landmarks[p2][3] > 0.5:
            pt1 = (int(landmarks[p1][0] * vid_w), int(landmarks[p1][1] * vid_h))
            pt2 = (int(landmarks[p2][0] * vid_w), int(landmarks[p2][1] * vid_h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Draw points
    for i in range(33):
        if landmarks[i][3] > 0.5:
            cx = int(landmarks[i][0] * vid_w)
            cy = int(landmarks[i][1] * vid_h)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    return frame
