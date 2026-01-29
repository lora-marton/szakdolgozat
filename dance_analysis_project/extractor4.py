import mediapipe as mp
import cv2
import time
import numpy as np
import h5py
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from one_euro_filter import OneEuroFilter

# Configuration
model_path = 'pose_landmarker_heavy.task'
video_path = 'videos/test_dance1.mp4'
output_path = 'output/dance_data.h5'

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
    min_tracking_confidence=0.8,)

def process_video():
    filters = {}
    all_data = [] # List to store collected frames [frame_idx, landmark_idx, x, y, z, vis, pres]
    
    # Store just the matrix of shape (N_frames, 33, 3) or similar? 
    # Usually easier to store (N_frames, 33, 4) -> x, y, z, visibility
    frames_data = [] 

    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) # Could be useful for metadata
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            # --- One-Euro Filter & Data Collection ---
            current_frame_landmarks = np.zeros((33, 4), dtype=np.float32) # x, y, z, visibility

            if pose_landmarker_result.pose_landmarks:
                current_time = frame_timestamp_ms / 1000.0
                landmarks = pose_landmarker_result.pose_landmarks[0]
                
                ENABLE_FILTER = True 
                
                for i, landmark in enumerate(landmarks):
                    if ENABLE_FILTER:
                        if i not in filters:
                            filters[i] = [
                                OneEuroFilter(current_time, landmark.x, min_cutoff=1.0, beta=10.0), 
                                OneEuroFilter(current_time, landmark.y, min_cutoff=1.0, beta=10.0)
                            ]
                        sx = filters[i][0](current_time, landmark.x)
                        sy = filters[i][1](current_time, landmark.y)
                    else:
                         sx, sy = landmark.x, landmark.y
                    
                    # Store normalized coordinates (MediaPipe default x,y are [0,1])
                    # z is relative scale, visibility is [0,1]
                    current_frame_landmarks[i] = [sx, sy, landmark.z, landmark.visibility]
                    
            frames_data.append(current_frame_landmarks)

            # Optional: Visualization (retained for verification)
            # You can comment out imshow if you want faster processing
            annotated_frame = frame.copy()
            h, w = frame.shape[:2]
            VISIBILITY_THRESHOLD = 0.5
            
            # Draw (simplified for verify)
            if pose_landmarker_result.pose_landmarks:
                for i in range(33):
                    lm = current_frame_landmarks[i]
                    if lm[3] > VISIBILITY_THRESHOLD:
                         cv2.circle(annotated_frame, (int(lm[0]*w), int(lm[1]*h)), 2, (0, 255, 0), -1)
            
            cv2.imshow('Recording...', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

        # --- Save to HDF5 ---
        print(f"Saving {len(frames_data)} frames of data to {output_path}...")
        
        # Convert to numpy array (N_Frames, 33, 4)
        np_data = np.array(frames_data, dtype=np.float32)
        
        with h5py.File(output_path, 'w') as f:
            dset = f.create_dataset('pose_data', data=np_data)
            dset.attrs['fps'] = fps if fps else 30.0
            dset.attrs['columns'] = 'x,y,z,visibility'
            dset.attrs['model'] = 'mediapipe_heavy_smoothed'
            
        print("Done!")

if __name__ == "__main__":
    process_video()