import mediapipe as mp
import cv2
import numpy as np
import h5py
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from one_euro_filter import OneEuroFilter

# Configuration
model_path = 'pose_landmarker_heavy.task'
video_path = 'videos/test_dance1.mp4'
output_path = 'output/dance_data_processed.h5'
output_mask_path = 'output/dance_masks.h5'

TARGET_FPS = 60.0
# Reduced target size to ensure dancer fits even with arms raised
TARGET_TORSO_PX = 40.0 
TARGET_MASK_SIZE = (256, 256)
NORM_CENTER = (128, 128)

# Initialize MediaPipe Pose Landmarker
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.8, 
    min_pose_presence_confidence=0.8,
    min_tracking_confidence=0.8,
    output_segmentation_masks=True)

def get_torso_stats(landmarks, image_width, image_height):
    """Calculates mid-hip, mid-shoulder, and torso length in pixels."""
    # Landmarks: 11-12 (Shoulders), 23-24 (Hips)
    l_shoulder = np.array([landmarks[11].x * image_width, landmarks[11].y * image_height])
    r_shoulder = np.array([landmarks[12].x * image_width, landmarks[12].y * image_height])
    l_hip = np.array([landmarks[23].x * image_width, landmarks[23].y * image_height])
    r_hip = np.array([landmarks[24].x * image_width, landmarks[24].y * image_height])
    
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0
    mid_hip = (l_hip + r_hip) / 2.0
    
    torso_len = np.linalg.norm(mid_shoulder - mid_hip)
    return mid_hip, torso_len

def process_video():
    filters = {}
    
    collected_raw = [] 
    collected_masks = []
    collected_trajectory = [] # Store hip position (GPS)
    
    # Store the FIXED SCALE from the FIRST valid frame
    fixed_scale_factor = None
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval_ms = 1000.0 / TARGET_FPS
        
        # Get video dimensions
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Source: {vid_w}x{vid_h} @ {source_fps} FPS. Target: {TARGET_FPS} FPS")
        
        last_processed_time = -frame_interval_ms
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            timestamp_ms = (frame_idx * 1000.0) / source_fps
            
            if timestamp_ms < last_processed_time + frame_interval_ms - (1000.0/source_fps/2):
                continue
                
            last_processed_time += frame_interval_ms
            
            # --- Processing ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
            
            frame_raw = np.zeros((33, 4), dtype=np.float32)
            
            # For visualization
            mask_overlay = frame.copy()
            norm_mask_vis = np.zeros((256, 256), dtype=np.uint8)
            current_trajectory = [0.0, 0.0]

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                
                # --- 1. Filter Raw Landmarks ---
                current_time_sec = timestamp_ms / 1000.0
                ENABLE_FILTER = True
                filtered_landmarks = [] # Object list for internal math
                
                for i, lm in enumerate(landmarks):
                    if ENABLE_FILTER:
                        if i not in filters:
                            filters[i] = [
                                OneEuroFilter(current_time_sec, lm.x, min_cutoff=1.0, beta=20.0),
                                OneEuroFilter(current_time_sec, lm.y, min_cutoff=1.0, beta=20.0),
                                OneEuroFilter(current_time_sec, lm.z, min_cutoff=1.0, beta=20.0)
                            ]
                        sx = filters[i][0](current_time_sec, lm.x)
                        sy = filters[i][1](current_time_sec, lm.y)
                        sz = filters[i][2](current_time_sec, lm.z)
                    else:
                        sx, sy, sz = lm.x, lm.y, lm.z
                    
                    frame_raw[i] = [sx, sy, sz, lm.visibility]
                    
                    class LmObj: pass
                    lm_o = LmObj(); lm_o.x = sx; lm_o.y = sy; lm_o.z = sz
                    filtered_landmarks.append(lm_o)

                # --- 2. Scale Calibration (Frame 0 Only) ---
                current_hip_center, current_torso_len = get_torso_stats(filtered_landmarks, vid_w, vid_h)
                
                if fixed_scale_factor is None:
                    # Calculate Scale ONCE based on Height
                    fixed_scale_factor = TARGET_TORSO_PX / current_torso_len
                    print(f"Calibration Complete. Fixed Scale: {fixed_scale_factor:.2f}")

                # --- 3. Follow-Cam Calculation (Every Frame) ---
                # We update Translation every frame to center the hips
                # But we keep Scale constant to preserve Z-depth
                
                tx = NORM_CENTER[0] - (current_hip_center[0] * fixed_scale_factor)
                ty = NORM_CENTER[1] - (current_hip_center[1] * fixed_scale_factor)
                
                # GPS Recording: where was the hip in the original video?
                current_trajectory = [current_hip_center[0], current_hip_center[1]]
                
                affine_matrix = np.float32([
                    [fixed_scale_factor, 0, tx],
                    [0, fixed_scale_factor, ty]
                ])

                # --- 4. Mask Processing ---
                segmentation_mask = result.segmentation_masks[0].numpy_view()
                
                # Apply the Follow-Cam transformation
                norm_mask_vis = cv2.warpAffine(
                    segmentation_mask, 
                    affine_matrix, 
                    TARGET_MASK_SIZE,
                    flags=cv2.INTER_LINEAR
                )
                
                mask_uint8 = (norm_mask_vis * 255).astype(np.uint8)
                collected_masks.append(mask_uint8)

                # --- 5. Visualization (Restored) ---
                # Draw mask overlay
                visual_mask = (segmentation_mask * 255).astype(np.uint8)
                visual_mask = cv2.cvtColor(visual_mask, cv2.COLOR_GRAY2BGR)
                visual_mask[:, :, 0] = 0 # B (Set Blue to 0, result is G+R = Yellow)
                mask_overlay = cv2.addWeighted(frame, 1.0, visual_mask, 0.5, 0)
                
                # Draw Skeleton on Main View (Manual OpenCV)
                # Define connections for visualization (subset of full body)
                POSE_CONNECTIONS = [
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
                    (11, 23), (12, 24), (23, 24),                     # Torso
                    (23, 25), (25, 27), (24, 26), (26, 28)            # Legs
                ]

                # Draw Lines
                for p1, p2 in POSE_CONNECTIONS:
                    if frame_raw[p1][3] > 0.5 and frame_raw[p2][3] > 0.5:
                        pt1 = (int(frame_raw[p1][0] * vid_w), int(frame_raw[p1][1] * vid_h))
                        pt2 = (int(frame_raw[p2][0] * vid_w), int(frame_raw[p2][1] * vid_h))
                        cv2.line(mask_overlay, pt1, pt2, (0, 255, 0), 2)

                # Draw Points
                for i in range(33):
                    lm = frame_raw[i]
                    if lm[3] > 0.5: # Visibility
                        cx, cy = int(lm[0] * vid_w), int(lm[1] * vid_h)
                        cv2.circle(mask_overlay, (cx, cy), 3, (0, 0, 255), -1)

            collected_raw.append(frame_raw)
            collected_trajectory.append(current_trajectory)

            # Debug Views
            cv2.imshow('Main View (Skeleton + Mask)', mask_overlay)
            cv2.imshow('Follow-Cam View (Centered)', norm_mask_vis)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # --- Save ---
        print(f"Saving {len(collected_raw)} frames...")
        
        with h5py.File(output_path, 'w') as f:
            # Raw Keypoints
            dset_raw = f.create_dataset('raw', data=np.array(collected_raw, dtype=np.float32))
            
            # Trajectory (GPS)
            dset_traj = f.create_dataset('trajectory', data=np.array(collected_trajectory, dtype=np.float32))
            dset_traj.attrs['description'] = 'Hip Center (x, y) in original pixels'
            
            f.attrs['fps'] = TARGET_FPS
            f.attrs['fixed_scale'] = fixed_scale_factor if fixed_scale_factor is not None else 1.0
            
        print(f"Saving masks to {output_mask_path} (gzip)...")
        with h5py.File(output_mask_path, 'w') as f:
            dset_masks = f.create_dataset(
                'masks', 
                data=np.array(collected_masks, dtype=np.uint8),
                compression="gzip",
                compression_opts=4
            )
            
        print("Done!")

if __name__ == "__main__":
    process_video()
