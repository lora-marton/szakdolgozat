import mediapipe as mp
import cv2
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from one_euro_filter import OneEuroFilter

# Configuration
model_path = 'pose_landmarker_heavy.task'
video_path = 'videos/test_dance1.mp4'

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
    
    # Store filters for each person ID (or just index 0 since we likely track one person)
    # { landmark_index: [filter_x, filter_y] }
    filters = {}
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        # Use OpenCV’s VideoCapture to load the input video
        cap = cv2.VideoCapture(video_path)
        
        # Load the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate timestamp in ms
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            # OpenCV loads files as BGR, MediaPipe needs RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Perform detection
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            # Print result or do something with it
            if pose_landmarker_result.pose_landmarks:
                print(f"Frame at {frame_timestamp_ms}ms: Found {len(pose_landmarker_result.pose_landmarks)} pose(s)")
            
            # --- One-Euro Filter Logic ---
            # Create dictionaries to hold filters if they don't exist yet
            # Using 'static' variables via function attributes or a global dict would work, 
            # but here we can define it outside the loop if we restructure, 
            # OR checking if it exists in a persisted dictionary.
            # Since process_video is a function, let's init filters before the loop.
            
            # Visualization
            # Visualization
            annotated_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # MediaPipe Pose Topology (Standard Blazepose)
            POSE_CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
                (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
                (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
            ]

            if pose_landmarker_result.pose_landmarks:
                # Use video timestamp for filtering, NOT wall-clock time
                # Convert ms to seconds
                current_time = frame_timestamp_ms / 1000.0
                
                # We are assuming single person tracking for simplicity (index 0)
                # If multiple people, we would need ID tracking to maintain filter state per person.
                landmarks = pose_landmarker_result.pose_landmarks[0]
                
                # Smooth the landmarks
                smoothed_landmarks = []
                
                # Filter Configuration
                ENABLE_FILTER = True 
                VISIBILITY_THRESHOLD = 0.5 # Lowered to keep more limbs visible
                
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
                    
                    # Preserve z and visibility from original landmark
                    smoothed_landmarks.append(type('Landmark', (object,), {
                        'x': sx, 'y': sy, 'z': landmark.z, 
                        'visibility': landmark.visibility,
                        'presence': landmark.presence
                    }))
                
                # Draw connections
                for connection in POSE_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start = smoothed_landmarks[start_idx]
                    end = smoothed_landmarks[end_idx]
                    
                    # Don't draw if either end is hidden
                    if start.visibility > VISIBILITY_THRESHOLD and end.visibility > VISIBILITY_THRESHOLD:
                        cv2.line(annotated_frame, 
                                 (int(start.x * w), int(start.y * h)), 
                                 (int(end.x * w), int(end.y * h)), 
                                 (245, 66, 230), 2)
                                 
                # Draw landmarks
                for landmark in smoothed_landmarks:
                    if landmark.visibility > VISIBILITY_THRESHOLD:
                        cv2.circle(annotated_frame, 
                                   (int(landmark.x * w), int(landmark.y * h)), 
                                   2, (245, 117, 66), -1)
            
            cv2.imshow('Skeleton', annotated_frame)

            # cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()