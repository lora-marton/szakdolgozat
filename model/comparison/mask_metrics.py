"""
Mask-based comparison metrics: EFD contour smoothing, DTM shape scoring,
and optical flow energy analysis.

Pipeline (per your research notes):
  1. EFD smooth  →  strip clothing noise from mask contours
  2. DTM score   →  Gaussian-weighted distance transform overlap
  3. Optical flow →  compare movement "energy" inside the masks
"""
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from pyefd import elliptic_fourier_descriptors, reconstruct_contour


# ── Stage 1: EFD Contour Smoothing ───────────────────────────────────────


def smooth_mask_efd(mask, n_harmonics=8, n_points=200, threshold=128):
    """
    Smooth a segmentation mask by keeping only low-frequency contour harmonics.

    Binarizes the mask, extracts the largest contour, decomposes it with
    Elliptic Fourier Descriptors, and reconstructs a clean contour using
    only the first `n_harmonics` harmonics (discarding clothing noise).

    Args:
        mask: Single-frame mask of shape (H, W), uint8 (0–255).
        n_harmonics: Number of EFD harmonics to keep (5–8 recommended).
        n_points: Number of points in the reconstructed contour.
        threshold: Binarization threshold for the uint8 mask.

    Returns:
        smoothed: Binary mask of same shape, uint8 (0 or 255).
            Returns all-zeros if no valid contour is found.
    """
    # Binarize
    binary = (mask > threshold).astype(np.uint8) * 255

    # Find contours — keep only the largest (the dancer)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.zeros_like(mask)

    largest = max(contours, key=cv2.contourArea)

    # Need at least enough points for EFD
    if len(largest) < 10:
        return np.zeros_like(mask)

    # Reshape from (M, 1, 2) to (M, 2) for pyefd
    contour_points = largest.squeeze()

    # Compute EFD coefficients (normalized: invariant to rotation/scale/start)
    coeffs = elliptic_fourier_descriptors(contour_points, order=n_harmonics, normalize=True)

    # Compute the centroid (locus) of the original contour
    cx = contour_points[:, 0].mean()
    cy = contour_points[:, 1].mean()

    # Reconstruct a smoothed contour from the low-frequency harmonics
    smoothed_contour = reconstruct_contour(coeffs, locus=(cx, cy), num_points=n_points)

    # Fill the smoothed contour to create a binary mask
    smoothed_int = smoothed_contour.astype(np.int32).reshape(-1, 1, 2)
    result = np.zeros_like(mask)
    cv2.fillPoly(result, [smoothed_int], 255)

    return result


# ── Stage 2: Distance Transform Mapping ──────────────────────────────────


def compare_shapes_dtm(teacher_masks, student_masks, sigma=10.0,
                       n_harmonics=8, n_points=200, threshold=128):
    """
    Compare dancer silhouettes using Gaussian-weighted Distance Transform.

    For each frame:
      1. Smooth both masks via EFD.
      2. Generate a Gaussian DTM from the teacher's smoothed mask.
      3. Multiply the student's smoothed mask by the teacher's DTM.
      4. Score = mean of the student's pixel values in the DTM overlap.

    Frames where both masks are empty are skipped.

    Args:
        teacher_masks: Array of shape (N, H, W), uint8 (0–255).
        student_masks: Array of shape (N, H, W), uint8 (0–255).
        sigma: Gaussian decay width in pixels for the distance transform.
        n_harmonics: EFD harmonics for contour smoothing.
        n_points: Points in the reconstructed EFD contour.
        threshold: Binarization threshold.

    Returns:
        mean_score: Float 0–1 — average shape similarity across active frames.
        per_frame_scores: Array of shape (N,) — DTM score per frame (0–1).
    """
    n_frames = teacher_masks.shape[0]
    per_frame_scores = np.zeros(n_frames, dtype=np.float32)
    active_mask = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        # Smooth both masks
        t_smooth = smooth_mask_efd(teacher_masks[i], n_harmonics, n_points, threshold)
        s_smooth = smooth_mask_efd(student_masks[i], n_harmonics, n_points, threshold)

        t_binary = (t_smooth > 0)
        s_binary = (s_smooth > 0)

        # Skip if both are empty
        if not t_binary.any() and not s_binary.any():
            per_frame_scores[i] = 1.0  # Both empty = "match"
            continue

        active_mask[i] = True

        # If only one is empty, score = 0
        if not t_binary.any() or not s_binary.any():
            per_frame_scores[i] = 0.0
            continue

        # Distance transform on teacher's mask
        dist_map = distance_transform_edt(t_binary)

        # Gaussian weighting: center = 1.0, edges → 0
        dtm = np.exp(-(dist_map ** 2) / (2 * sigma ** 2))

        # Score: mean DTM value at student's mask pixels
        student_pixels = s_binary.astype(bool)
        score = dtm[student_pixels].mean()

        per_frame_scores[i] = score

    if not active_mask.any():
        return 1.0, per_frame_scores

    mean_score = float(per_frame_scores[active_mask].mean())

    return round(mean_score, 4), per_frame_scores


# ── Stage 3: Optical Flow Energy ─────────────────────────────────────────


def compare_mask_energy(teacher_masks, student_masks, winsize=15,
                        threshold=128, min_energy=1e-3):
    """
    Compare movement energy inside segmentation masks using optical flow.

    For each consecutive frame pair, computes Farneback dense optical flow,
    then measures the average magnitude of motion vectors inside the mask.
    Compares teacher vs student energy using min/max ratio.

    Args:
        teacher_masks: Array of shape (N, H, W), uint8 (0–255).
        student_masks: Array of shape (N, H, W), uint8 (0–255).
        winsize: Farneback averaging window size.
        threshold: Binarization threshold for mask region.
        min_energy: Minimum average magnitude to count as active.

    Returns:
        Dict with:
            'energy_score': float 0–1 (aggregate score).
            'per_frame_ratios': array of shape (N-1,) — min/max ratio per pair.
            'teacher_energy': array of shape (N-1,) — avg magnitude per frame.
            'student_energy': array of shape (N-1,) — avg magnitude per frame.
    """
    n_frames = teacher_masks.shape[0]

    if n_frames < 2:
        return {
            'energy_score': 1.0,
            'per_frame_ratios': np.array([], dtype=np.float32),
            'teacher_energy': np.array([], dtype=np.float32),
            'student_energy': np.array([], dtype=np.float32),
        }

    n_pairs = n_frames - 1
    teacher_energy = np.zeros(n_pairs, dtype=np.float32)
    student_energy = np.zeros(n_pairs, dtype=np.float32)
    per_frame_ratios = np.zeros(n_pairs, dtype=np.float32)

    for i in range(n_pairs):
        # Teacher flow
        t_energy = _compute_frame_energy(
            teacher_masks[i], teacher_masks[i + 1], winsize, threshold,
        )
        # Student flow
        s_energy = _compute_frame_energy(
            student_masks[i], student_masks[i + 1], winsize, threshold,
        )

        teacher_energy[i] = t_energy
        student_energy[i] = s_energy

        # min/max ratio (skip if both near-zero)
        max_e = max(t_energy, s_energy)
        min_e = min(t_energy, s_energy)

        if max_e < min_energy:
            per_frame_ratios[i] = 1.0  # Both stationary = match
        else:
            per_frame_ratios[i] = min_e / max_e

    # Aggregate: mean of ratios (excluding stationary-both frames)
    active_mask = (teacher_energy > min_energy) | (student_energy > min_energy)

    if not active_mask.any():
        energy_score = 1.0
    else:
        energy_score = float(per_frame_ratios[active_mask].mean())

    return {
        'energy_score': round(energy_score, 4),
        'per_frame_ratios': per_frame_ratios,
        'teacher_energy': teacher_energy,
        'student_energy': student_energy,
    }


def _compute_frame_energy(mask_prev, mask_curr, winsize, threshold):
    """
    Compute average optical flow magnitude inside the mask between two frames.

    Flow is computed on the full mask image (grayscale), then masked to
    only measure pixels inside the union of both frame masks.

    Args:
        mask_prev: Mask at frame t, uint8 (H, W).
        mask_curr: Mask at frame t+1, uint8 (H, W).
        winsize: Farneback window size.
        threshold: Binarization threshold.

    Returns:
        energy: Float — average flow magnitude inside mask pixels.
    """
    # Use the masks directly as grayscale images for flow computation
    prev_gray = mask_prev
    curr_gray = mask_curr

    # Compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=winsize,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # Magnitude of flow vectors
    magnitude = np.linalg.norm(flow, axis=-1)

    # Mask: union of both frames (movement could be in either region)
    region = (mask_prev > threshold) | (mask_curr > threshold)

    if not region.any():
        return 0.0

    return float(magnitude[region].mean())


# ── Top-Level Orchestrator ───────────────────────────────────────────────


def compute_mask_score(teacher_masks, student_masks, config):
    """
    Compute the combined mask comparison score using EFD + DTM + optical flow.

    Pipeline:
      1. EFD contour smoothing (inside compare_shapes_dtm).
      2. DTM shape comparison → shape similarity score.
      3. Optical flow energy comparison → energy similarity score.
      4. Combined: score = weight_shape * dtm + weight_energy * flow.

    Args:
        teacher_masks: Array of shape (N, H, W), uint8 (0–255).
        student_masks: Array of shape (N, H, W), uint8 (0–255).
        config: ComparisonConfig instance.

    Returns:
        Dict with:
            'score': float 0–100 — combined mask score.
            'per_frame_shape': array (N,) — DTM shape scores per frame.
            'energy_details': dict from compare_mask_energy with per-frame data.
    """
    # Stage 1+2: EFD smoothing + DTM shape comparison
    dtm_score, per_frame_shape = compare_shapes_dtm(
        teacher_masks, student_masks,
        sigma=config.dtm_sigma,
        n_harmonics=config.efd_harmonics,
        n_points=config.efd_contour_points,
        threshold=config.mask_binary_threshold,
    )

    # Stage 3: Optical flow energy comparison
    energy_details = compare_mask_energy(
        teacher_masks, student_masks,
        winsize=config.flow_winsize,
        threshold=config.mask_binary_threshold,
    )

    # Weighted combination → 0–100 scale
    combined = (
        config.weight_shape * dtm_score
        + config.weight_energy * energy_details['energy_score']
    ) * 100.0

    return {
        'score': round(combined, 1),
        'per_frame_shape': per_frame_shape,
        'energy_details': energy_details,
    }
