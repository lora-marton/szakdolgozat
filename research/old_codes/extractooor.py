import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configuration
model_path = 'pose_landmarker_heavy.task'  # Path to your downloaded model
video_path = 'videos/test_dance1.mp4'

# 2. Initialize the Pose Landmarker
BaseOptions = mp.tasks.python.BaseOptions
PoseLandmarker = mp.tasks.python.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.python.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.python.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

def process_video(path):
    # Use context manager to ensure landmarker is closed properly
    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert OpenCV BGR to MediaPipe RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Calculate timestamp in ms (required for VIDEO mode)
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # Perform detection
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # 3. Handle Results
            if pose_landmarker_result.pose_landmarks:
                for landmarks in pose_landmarker_result.pose_landmarks:
                    # Example: Get the nose landmark
                    # print(f"Nose: x={landmarks[0].x}, y={landmarks[0].y}")
                    
                    # TIP: For dance, you'll want to save these to a list or CSV 
                    # to compare them later with the second video.
                    pass

            # Optional: Visualization (using legacy drawing utils for speed)
            # You can add custom drawing logic here if needed.
            
            cv2.imshow('Dance Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(video_path)