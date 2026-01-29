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
TARGET_FPS = 30.0

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
    min_tracking_confidence=0.8)

def get_hip_center(landmarks):
    """Calculates the average of the left (23) and right (24) hip coordinates."""
    # Landmark 23: Left Hip, 24: Right Hip
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    
    # We only care about x, y, z for centering
    center_x = (left_hip[0] + right_hip[0]) / 2.0
    center_y = (left_hip[1] + right_hip[1]) / 2.0
    center_z = (left_hip[2] + right_hip[2]) / 2.0
    
    return center_x, center_y, center_z

def process_video():
    filters = {}
    
    # Data containers
    # Raw: Normalized [0,1] screen coords
    collected_raw = [] 
    # Centered: Hip is at (0,0), relative coordinates
    collected_centered = []
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval_ms = 1000.0 / TARGET_FPS
        
        print(f"Source FPS: {source_fps}, Target FPS: {TARGET_FPS}")
        
        current_animation_time = 0.0
        last_processed_time = -frame_interval_ms # Ensure we process frame 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Current frame timestamp in video timeline
            # We use exact frame counts to avoid drift: frame_idx * (1000/fps)
            frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            timestamp_ms = (frame_idx * 1000.0) / source_fps
            
            # FPS RESAMPLING LOGIC
            # Simple "Nearest" Strategy:
            # Check if this frame is the closest to our next target timestamp
            # Ideally, we'd interpolate, but for stick figures, dropping/duping is often acceptable baseline.
            # Here implementation: simple sampler. 
            # If current timestamp is closer to target than prev/next, take it.
            # A simpler approach for extraction loop:
            # Just verify if timestamp_ms >= next_needed_timestamp
            
            # Better Approach: 
            # Continually increment our virtual 'target_time'. 
            # If the current frame is past that time, we process it.
            # This handles downsampling. Upsampling (duplicating) logic is omitted for simplicity 
            # unless source FPS < 30.
            
            if timestamp_ms < last_processed_time + frame_interval_ms - (1000.0/source_fps/2):
                # Skip this frame, we already have a sample close enough or it's too early
                continue
                
            last_processed_time += frame_interval_ms
            
            # --- Processing ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            pose_landmarker_result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
            
            # Shape: (33, 4) -> x, y, z, visibility
            frame_raw = np.zeros((33, 4), dtype=np.float32)
            frame_centered = np.zeros((33, 4), dtype=np.float32)
            
            if pose_landmarker_result.pose_landmarks:
                landmarks = pose_landmarker_result.pose_landmarks[0]
                current_time_sec = timestamp_ms / 1000.0
                
                # 1. One-Euro Filtering
                ENABLE_FILTER = True
                filtered_landmarks_list = []
                
                for i, lm in enumerate(landmarks):
                    if ENABLE_FILTER:
                        if i not in filters:
                            filters[i] = [
                                OneEuroFilter(current_time_sec, lm.x, min_cutoff=1.0, beta=10.0), # x
                                OneEuroFilter(current_time_sec, lm.y, min_cutoff=1.0, beta=10.0), # y
                                OneEuroFilter(current_time_sec, lm.z, min_cutoff=1.0, beta=10.0)  # z
                            ]
                        sx = filters[i][0](current_time_sec, lm.x)
                        sy = filters[i][1](current_time_sec, lm.y)
                        sz = filters[i][2](current_time_sec, lm.z) # Filter Z too!
                    else:
                        sx, sy, sz = lm.x, lm.y, lm.z
                    
                    filtered_landmarks_list.append([sx, sy, sz, lm.visibility])
                    
                    # Store Raw
                    frame_raw[i] = [sx, sy, sz, lm.visibility]
                
                # 2. Hip-Centric Calculation
                # Get hip center from the FILTERED data
                cx, cy, cz = get_hip_center(filtered_landmarks_list)
                
                for i in range(33):
                    raw_pt = filtered_landmarks_list[i]
                    # Subtract hip center
                    rel_x = raw_pt[0] - cx
                    rel_y = raw_pt[1] - cy 
                    rel_z = raw_pt[2] - cz
                    # Vis stays same
                    vis = raw_pt[3]
                    
                    frame_centered[i] = [rel_x, rel_y, rel_z, vis]
            
            collected_raw.append(frame_raw)
            collected_centered.append(frame_centered)

            # Optional: Visual Debug (Hip Centered)
            # Visualize locally to check if it "stays in place"
            vis_frame = np.zeros((h:=frame.shape[0], w:=frame.shape[1], 3), dtype=np.uint8)
            center_x_screen = w // 2
            center_y_screen = h // 2
            
            if pose_landmarker_result.pose_landmarks:
                for i in range(33):
                    lm = frame_centered[i]
                    if lm[3] > 0.5:
                        # Project back to screen center for viz
                        px = int(lm[0] * w + center_x_screen)
                        py = int(lm[1] * h + center_y_screen)
                        cv2.circle(vis_frame, (px, py), 2, (0, 255, 0), -1)
            
            cv2.imshow('Hip-Centric Debug', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # --- Save to HDF5 ---
        print(f"Saving {len(collected_raw)} frames of data to {output_path}...")
        
        with h5py.File(output_path, 'w') as f:
            # Raw Data
            dset_raw = f.create_dataset('raw', data=np.array(collected_raw, dtype=np.float32))
            dset_raw.attrs['description'] = 'Normalized [0,1] coordinates'
            dset_raw.attrs['columns'] = 'x,y,z,visibility'
            
            # Centered Data
            dset_cnt = f.create_dataset('centered', data=np.array(collected_centered, dtype=np.float32))
            dset_cnt.attrs['description'] = 'Relative to mid-hip coordinates'
            dset_cnt.attrs['columns'] = 'rel_x,rel_y,rel_z,visibility'
            
            # Global Metadata
            f.attrs['fps'] = TARGET_FPS
            f.attrs['model'] = 'mediapipe_video_resampled'
            
        print("Done!")

if __name__ == "__main__":
    process_video()
