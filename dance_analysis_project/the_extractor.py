import mediapipe as mp
import cv2
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    output_segmentation_masks=True)

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
            if pose_landmarker_result.segmentation_masks:
                segmentation_mask = pose_landmarker_result.segmentation_masks[0].numpy_view()
                
                # Convert to uint8 and standard BGR format for display
                # MediaPipe masks are float [0, 1]
                mask_uint8 = (segmentation_mask * 255).astype(np.uint8)
                visualized_mask = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
                
                cv2.imshow('Segmentation Mask', visualized_mask)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()