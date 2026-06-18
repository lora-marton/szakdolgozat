"""
Mask-based comparison metrics: EFD contour smoothing, DTM shape scoring,
and optical flow energy analysis.
"""

import cv2
import numpy as np
from pyefd import elliptic_fourier_descriptors, reconstruct_contour
from scipy.ndimage import distance_transform_edt


class MaskMetrics:
    """Silhouette shape and movement energy comparison between two pose sequences."""

    @staticmethod
    def compute_mask_score(
        teacher_masks: np.ndarray,
        student_masks: np.ndarray,
        config,
    ) -> dict:
        """Compute the combined mask comparison score.

        Runs EFD contour smoothing, DTM shape comparison, and optical flow
        energy comparison, then combines them with configured weights.

        Args:
            teacher_masks: Array of shape (N, H, W), uint8 0-255.
            student_masks: Array of shape (N, H, W), uint8 0-255.
            config: ComparisonConfig instance.

        Returns:
            Dict with 'score' (0-100), 'per_frame_shape' array,
            and 'energy_details' dict.
        """
        dtm_score, per_frame_shape = MaskMetrics._compare_shapes_dtm(
            teacher_masks,
            student_masks,
            sigma=config.dtm_sigma,
            n_harmonics=config.efd_harmonics,
            n_points=config.efd_contour_points,
            threshold=config.mask_binary_threshold,
        )

        energy_details = MaskMetrics._compare_mask_energy(
            teacher_masks,
            student_masks,
            winsize=config.flow_winsize,
            threshold=config.mask_binary_threshold,
        )

        combined = (config.weight_shape * dtm_score + config.weight_energy * energy_details["energy_score"]) * 100.0

        return {
            "score": round(combined, 1),
            "per_frame_shape": per_frame_shape,
            "energy_details": energy_details,
        }

    @staticmethod
    def _smooth_mask_efd(
        mask: np.ndarray,
        n_harmonics: int,
        n_points: int,
        threshold: int,
    ) -> np.ndarray:
        """Smooth a segmentation mask using Elliptic Fourier Descriptors.

        Binarizes the mask, extracts the largest contour, decomposes it
        into EFD harmonics, and reconstructs a clean contour from the
        low-frequency components only.

        Args:
            mask: Single-frame mask of shape (H, W), uint8 0-255.
            n_harmonics: Number of EFD harmonics to keep.
            n_points: Number of points in the reconstructed contour.
            threshold: Binarization threshold for the uint8 mask.

        Returns:
            Smoothed binary mask of same shape, uint8 0 or 255.
        """
        binary = (mask > threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return np.zeros_like(mask)

        largest = max(contours, key=cv2.contourArea)

        if len(largest) < 10:
            return np.zeros_like(mask)

        contour_points = largest.squeeze()

        coeffs = elliptic_fourier_descriptors(contour_points, order=n_harmonics, normalize=False)

        cx = contour_points[:, 0].mean()
        cy = contour_points[:, 1].mean()

        smoothed_contour = reconstruct_contour(coeffs, locus=(cx, cy), num_points=n_points)

        smoothed_int = smoothed_contour.astype(np.int32).reshape(-1, 1, 2)
        result = np.zeros_like(mask)
        cv2.fillPoly(result, [smoothed_int], 255)

        return result

    @staticmethod
    def _compare_shapes_dtm(
        teacher_masks: np.ndarray,
        student_masks: np.ndarray,
        sigma: float,
        n_harmonics: int,
        n_points: int,
        threshold: int,
    ) -> tuple[float, np.ndarray]:
        """Compare dancer silhouettes using bidirectional Gaussian DTM.

        For each frame, smooths both masks via EFD, then scores overlap
        bidirectionally: how well the student covers the teacher shape,
        and how well the teacher covers the student shape.

        Args:
            teacher_masks: Array of shape (N, H, W), uint8 0-255.
            student_masks: Array of shape (N, H, W), uint8 0-255.
            sigma: Gaussian decay width in pixels for the distance transform.
            n_harmonics: EFD harmonics for contour smoothing.
            n_points: Points in the reconstructed EFD contour.
            threshold: Binarization threshold.

        Returns:
            Tuple of (mean_score 0-1, per_frame_scores array of shape N).
        """
        n_frames = teacher_masks.shape[0]
        per_frame_scores = np.zeros(n_frames, dtype=np.float32)
        active_mask = np.zeros(n_frames, dtype=bool)

        for i in range(n_frames):
            t_smooth = MaskMetrics._smooth_mask_efd(
                teacher_masks[i],
                n_harmonics,
                n_points,
                threshold,
            )
            s_smooth = MaskMetrics._smooth_mask_efd(
                student_masks[i],
                n_harmonics,
                n_points,
                threshold,
            )

            t_binary = t_smooth > 0
            s_binary = s_smooth > 0

            if not t_binary.any() and not s_binary.any():
                per_frame_scores[i] = 1.0
                continue

            active_mask[i] = True

            if not t_binary.any() or not s_binary.any():
                per_frame_scores[i] = 0.0
                continue

            forward = MaskMetrics._dtm_one_direction(t_binary, s_binary, sigma)
            backward = MaskMetrics._dtm_one_direction(s_binary, t_binary, sigma)
            per_frame_scores[i] = (forward + backward) / 2.0

        if not active_mask.any():
            return 1.0, per_frame_scores

        mean_score = float(per_frame_scores[active_mask].mean())

        return round(mean_score, 4), per_frame_scores

    @staticmethod
    def _dtm_one_direction(
        reference_binary: np.ndarray,
        query_binary: np.ndarray,
        sigma: float,
    ) -> float:
        """Score how well query pixels fall inside the reference shape.

        Builds a Gaussian score map from the reference mask where pixels
        inside score 1.0 and pixels outside decay with distance to the
        nearest edge. Samples the map at query pixel locations.

        Args:
            reference_binary: Boolean mask of the reference shape (H, W).
            query_binary: Boolean mask of the query shape (H, W).
            sigma: Gaussian decay width in pixels.

        Returns:
            Mean score of query pixels in the reference score map (0-1).
        """
        outside_dist = distance_transform_edt(~reference_binary)
        score_map = np.exp(-(outside_dist**2) / (2 * sigma**2))
        return float(score_map[query_binary].mean())

    @staticmethod
    def _compare_mask_energy(
        teacher_masks: np.ndarray,
        student_masks: np.ndarray,
        winsize: int,
        threshold: int,
        min_energy: float = 1e-3,
    ) -> dict:
        """Compare movement energy inside segmentation masks using optical flow.

        For each consecutive frame pair, computes Farneback dense optical
        flow on the mask images, measures average flow magnitude inside
        the mask region, and scores similarity as a min/max ratio.

        Args:
            teacher_masks: Array of shape (N, H, W), uint8 0-255.
            student_masks: Array of shape (N, H, W), uint8 0-255.
            winsize: Farneback averaging window size.
            threshold: Binarization threshold for mask region.
            min_energy: Minimum average magnitude to count as active.

        Returns:
            Dict with 'energy_score' (0-1), 'per_frame_ratios' array,
            'teacher_energy' array, and 'student_energy' array.
        """
        n_frames = teacher_masks.shape[0]

        if n_frames < 2:
            return {
                "energy_score": 1.0,
                "per_frame_ratios": np.array([], dtype=np.float32),
                "teacher_energy": np.array([], dtype=np.float32),
                "student_energy": np.array([], dtype=np.float32),
            }

        n_pairs = n_frames - 1
        teacher_energy = np.zeros(n_pairs, dtype=np.float32)
        student_energy = np.zeros(n_pairs, dtype=np.float32)
        per_frame_ratios = np.zeros(n_pairs, dtype=np.float32)

        for i in range(n_pairs):
            t_energy = MaskMetrics._compute_frame_energy(
                teacher_masks[i],
                teacher_masks[i + 1],
                winsize,
                threshold,
            )
            s_energy = MaskMetrics._compute_frame_energy(
                student_masks[i],
                student_masks[i + 1],
                winsize,
                threshold,
            )

            teacher_energy[i] = t_energy
            student_energy[i] = s_energy

            max_e = max(t_energy, s_energy)
            min_e = min(t_energy, s_energy)

            if max_e < min_energy:
                per_frame_ratios[i] = 1.0
            else:
                per_frame_ratios[i] = min_e / max_e

        active = (teacher_energy > min_energy) | (student_energy > min_energy)

        if not active.any():
            energy_score = 1.0
        else:
            energy_score = float(per_frame_ratios[active].mean())

        return {
            "energy_score": round(energy_score, 4),
            "per_frame_ratios": per_frame_ratios,
            "teacher_energy": teacher_energy,
            "student_energy": student_energy,
        }

    @staticmethod
    def _compute_frame_energy(
        mask_prev: np.ndarray,
        mask_curr: np.ndarray,
        winsize: int,
        threshold: int,
    ) -> float:
        """Compute average optical flow magnitude inside the mask between two frames.

        Args:
            mask_prev: Mask at frame t, uint8 (H, W).
            mask_curr: Mask at frame t+1, uint8 (H, W).
            winsize: Farneback window size.
            threshold: Binarization threshold.

        Returns:
            Average flow magnitude inside the mask region.
        """
        flow = cv2.calcOpticalFlowFarneback(
            mask_prev,
            mask_curr,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=winsize,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        magnitude = np.linalg.norm(flow, axis=-1)

        region = (mask_prev > threshold) | (mask_curr > threshold)

        if not region.any():
            return 0.0

        return float(magnitude[region].mean())
