import mediapipe as mp
import cv2
import numpy as np
import h5py
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from one_euro_filter import OneEuroFilter

# Configuration
model_path = 'pose_landmarker_heavy.task'
video_path = 'videos/test_dance2.mp4'
output_path = 'output/dance_data_processed.h5'
TARGET_FPS = 60.0

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

def process_video():
    filters = {}
    
    # Data containers
    # Raw: Normalized [0,1] screen coords
    collected_raw = [] 
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval_ms = 1000.0 / TARGET_FPS
        
        print(f"Source FPS: {source_fps}, Target FPS: {TARGET_FPS}")
        
        last_processed_time = -frame_interval_ms # Ensure we process frame 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Current frame timestamp in video timeline
            # We use exact frame counts to avoid drift: frame_idx * (1000/fps)
            frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            timestamp_ms = (frame_idx * 1000.0) / source_fps
            
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
            
            if pose_landmarker_result.pose_landmarks:
                landmarks = pose_landmarker_result.pose_landmarks[0]
                current_time_sec = timestamp_ms / 1000.0
                
                # 1. One-Euro Filtering
                ENABLE_FILTER = True
                
                for i, lm in enumerate(landmarks):
                    if ENABLE_FILTER:
                        if i not in filters:
                            filters[i] = [
                                OneEuroFilter(current_time_sec, lm.x, min_cutoff=1.0, beta=20.0), # x
                                OneEuroFilter(current_time_sec, lm.y, min_cutoff=1.0, beta=20.0), # y
                                OneEuroFilter(current_time_sec, lm.z, min_cutoff=1.0, beta=20.0)  # z
                            ]
                        sx = filters[i][0](current_time_sec, lm.x)
                        sy = filters[i][1](current_time_sec, lm.y)
                        sz = filters[i][2](current_time_sec, lm.z) # Filter Z too!
                    else:
                        sx, sy, sz = lm.x, lm.y, lm.z
                    
                    # Store Raw
                    frame_raw[i] = [sx, sy, sz, lm.visibility]
            
            collected_raw.append(frame_raw)

            # Optional: Simple Visual Debug
            # Project data back to screen for viz
            vis_frame = frame.copy()
            h, w = vis_frame.shape[:2]
            
            if pose_landmarker_result.pose_landmarks:
                for i in range(33):
                    lm = frame_raw[i]
                    if lm[3] > 0.5:
                        px = int(lm[0] * w)
                        py = int(lm[1] * h)
                        cv2.circle(vis_frame, (px, py), 2, (0, 255, 0), -1)
            
            cv2.imshow('Debug', vis_frame)
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
            
            # Global Metadata
            f.attrs['fps'] = TARGET_FPS
            f.attrs['model'] = 'mediapipe_video_resampled'
            
        print("Done!")

if __name__ == "__main__":
    process_video()
