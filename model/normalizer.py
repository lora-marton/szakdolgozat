import numpy as np
import cv2


def get_torso_stats(landmarks, image_width, image_height):
    """
    Calculate mid-hip position and torso length in pixels.

    Args:
        landmarks: Array of shape (33, 3+) with normalized x, y, z coordinates.
        image_width: Video frame width in pixels.
        image_height: Video frame height in pixels.

    Returns:
        (mid_hip, torso_len): mid_hip as (x, y) in pixels, torso_len in pixels.
    """
    l_shoulder = np.array([landmarks[11, 0] * image_width, landmarks[11, 1] * image_height])
    r_shoulder = np.array([landmarks[12, 0] * image_width, landmarks[12, 1] * image_height])
    l_hip = np.array([landmarks[23, 0] * image_width, landmarks[23, 1] * image_height])
    r_hip = np.array([landmarks[24, 0] * image_width, landmarks[24, 1] * image_height])

    mid_shoulder = (l_shoulder + r_shoulder) / 2.0
    mid_hip = (l_hip + r_hip) / 2.0

    torso_len = np.linalg.norm(mid_shoulder - mid_hip)
    return mid_hip, torso_len


def calibrate_scale(torso_len, target_torso_px):
    """
    Compute a fixed scale factor to normalize torso length.

    Args:
        torso_len: Measured torso length in pixels.
        target_torso_px: Desired torso length in the normalized space.

    Returns:
        scale_factor: float
    """
    return target_torso_px / torso_len


def compute_follow_cam(hip_center, scale_factor, norm_center):
    """
    Build an affine matrix that centers the hip and applies fixed scale.

    Args:
        hip_center: (x, y) hip position in original pixels.
        scale_factor: Fixed scale factor from calibration.
        norm_center: (cx, cy) target center in normalized space.

    Returns:
        affine_matrix: np.ndarray of shape (2, 3)
    """
    tx = norm_center[0] - (hip_center[0] * scale_factor)
    ty = norm_center[1] - (hip_center[1] * scale_factor)

    return np.float32([
        [scale_factor, 0, tx],
        [0, scale_factor, ty],
    ])


def warp_mask(mask, affine_matrix, target_size):
    """
    Apply the follow-cam affine transform to a segmentation mask.

    Args:
        mask: Raw segmentation mask from MediaPipe (float, 0-1 range).
        affine_matrix: 2x3 affine transform from compute_follow_cam.
        target_size: Output size as (width, height).

    Returns:
        mask_uint8: Normalized mask as uint8 (0-255).
    """
    warped = cv2.warpAffine(mask, affine_matrix, target_size, flags=cv2.INTER_LINEAR)
    return (warped * 255).astype(np.uint8)
