import os
import mediapipe as mp
import cv2
import numpy as np
import h5py

from model.config import DEFAULT_CONFIG
from model.one_euro_filter import OneEuroFilter
from model.normalizer import get_torso_stats, calibrate_scale, compute_follow_cam, warp_mask
from model.visualization import draw_skeleton, draw_mask_overlay


def data_extraction(video_path, output_dir='data', label='dance', debug=False, status_callback=None, config=None):
    """
    Extract pose landmarks, segmentation masks, and trajectory from a dance video.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory for output HDF5 files.
        label: Prefix for output filenames (e.g., 'teacher', 'student').
        debug: If True, show OpenCV debug windows.
        status_callback: Optional callback(msg: str) for progress updates.
        config: ExtractionConfig instance (uses DEFAULT_CONFIG if None).
    """
    if config is None:
        config = DEFAULT_CONFIG

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{label}_data.h5')
    output_mask_path = os.path.join(output_dir, f'{label}_masks.h5')

    options = config.create_landmarker_options()
    filters = {}

    collected_raw = []
    collected_masks = []
    collected_trajectory = []

    fixed_scale_factor = None

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval_ms = 1000.0 / config.target_fps

        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Source: {vid_w}x{vid_h} @ {source_fps} FPS. Target: {config.target_fps} FPS")

        last_processed_time = -frame_interval_ms

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            timestamp_ms = (frame_idx * 1000.0) / source_fps

            # FPS resampling
            if timestamp_ms < last_processed_time + frame_interval_ms - (1000.0 / source_fps / 2):
                continue
            last_processed_time += frame_interval_ms

            if frame_idx % 60 == 0 and status_callback:
                status_callback(f"Processing frame {int(frame_idx)}...")

            # Detect pose
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

            frame_raw = np.zeros((33, 4), dtype=np.float32)
            current_trajectory = [0.0, 0.0]
            norm_mask = np.zeros(config.target_mask_size, dtype=np.uint8)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # 1. Filter landmarks
                current_time_sec = timestamp_ms / 1000.0
                for i, lm in enumerate(landmarks):
                    if i not in filters:
                        filters[i] = [
                            OneEuroFilter(current_time_sec, lm.x, min_cutoff=config.filter_min_cutoff, beta=config.filter_beta),
                            OneEuroFilter(current_time_sec, lm.y, min_cutoff=config.filter_min_cutoff, beta=config.filter_beta),
                            OneEuroFilter(current_time_sec, lm.z, min_cutoff=config.filter_min_cutoff, beta=config.filter_beta),
                        ]
                    frame_raw[i] = [
                        filters[i][0](current_time_sec, lm.x),
                        filters[i][1](current_time_sec, lm.y),
                        filters[i][2](current_time_sec, lm.z),
                        lm.visibility,
                    ]

                # 2. Calibrate scale (first frame only)
                hip_center, torso_len = get_torso_stats(frame_raw, vid_w, vid_h)

                if fixed_scale_factor is None:
                    fixed_scale_factor = calibrate_scale(torso_len, config.target_torso_px)
                    print(f"Calibration Complete. Fixed Scale: {fixed_scale_factor:.2f}")

                # 3. Follow-cam transform
                affine_matrix = compute_follow_cam(hip_center, fixed_scale_factor, config.norm_center)
                current_trajectory = [hip_center[0], hip_center[1]]

                # 4. Normalize mask
                segmentation_mask = result.segmentation_masks[0].numpy_view()
                norm_mask = warp_mask(segmentation_mask, affine_matrix, config.target_mask_size)

                # 5. Debug visualization
                if debug:
                    overlay = draw_mask_overlay(frame, segmentation_mask)
                    overlay = draw_skeleton(overlay, frame_raw, config.pose_connections, vid_w, vid_h)
                    cv2.imshow('Main View (Skeleton + Mask)', overlay)
                    cv2.imshow('Follow-Cam View (Centered)', norm_mask)

            collected_raw.append(frame_raw)
            collected_masks.append(norm_mask)
            collected_trajectory.append(current_trajectory)

            if debug and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if debug:
            cv2.destroyAllWindows()

    # Save results
    print(f"Saving {len(collected_raw)} frames...")

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('raw', data=np.array(collected_raw, dtype=np.float32))
        dset_traj = f.create_dataset('trajectory', data=np.array(collected_trajectory, dtype=np.float32))
        dset_traj.attrs['description'] = 'Hip Center (x, y) in original pixels'
        f.attrs['fps'] = config.target_fps
        f.attrs['fixed_scale'] = fixed_scale_factor if fixed_scale_factor is not None else 1.0

    print(f"Saving masks to {output_mask_path} (gzip)...")
    with h5py.File(output_mask_path, 'w') as f:
        f.create_dataset(
            'masks',
            data=np.array(collected_masks, dtype=np.uint8),
            compression="gzip",
            compression_opts=4,
        )

    print("Done!")


if __name__ == "__main__":
    data_extraction('videos/test_dance1.mp4', output_dir='data', label='dance', debug=True)
