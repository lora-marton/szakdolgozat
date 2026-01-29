import mediapipe as mp
import cv2
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration
model_path = 'pose_landmarker_heavy.task'
video_path = 'videos/test_dance2.mp4'

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
                for landmarks in pose_landmarker_result.pose_landmarks:
                    # Draw connections
                    for connection in POSE_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        
                        start = landmarks[start_idx]
                        end = landmarks[end_idx]
                        
                        cv2.line(annotated_frame, 
                                 (int(start.x * w), int(start.y * h)), 
                                 (int(end.x * w), int(end.y * h)), 
                                 (245, 66, 230), 2)
                                 
                    # Draw landmarks
                    for landmark in landmarks:
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