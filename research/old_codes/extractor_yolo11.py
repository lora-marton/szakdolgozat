from ultralytics import YOLO
import cv2
import numpy as np

# Configuration
video_path = 'videos/test_dance2.mp4'
model_path = 'yolo11m-pose.pt'  # Will automatically download if not present

# Initialize YOLO model
model = YOLO(model_path)

def process_video():
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # COCO Keypoint Topology (Indices 0-16)
    # 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
    # 5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow
    # 9: L-Wrist, 10: R-Wrist, 11: L-Hip, 12: R-Hip
    # 13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle
    POSE_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # Face
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),               # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
    ]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection
        # stream=True for generator, verbose=False to reduce clutter
        results = model(frame, verbose=False)
        
        annotated_frame = frame.copy()
        
        # Process results
        for result in results:
            # Check if any keypoints were detected
            if result.keypoints is not None and result.keypoints.data is not None:
                # Keypoints are shape (N, 17, 3) -> (num_persons, num_keypoints, [x, y, conf])
                keypoints_data = result.keypoints.data.cpu().numpy()
                
                for person_kpts in keypoints_data:
                    # Filter out low confidence points if needed, but for drawing we often draw all valid ones
                    
                    # Draw connections
                    for connection in POSE_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        
                        kp_start = person_kpts[start_idx]
                        kp_end = person_kpts[end_idx]
                        
                        # Check confidence (index 2) usually > 0.5 is good
                        if kp_start[2] > 0.5 and kp_end[2] > 0.5:
                            start_pt = (int(kp_start[0]), int(kp_start[1]))
                            end_pt = (int(kp_end[0]), int(kp_end[1]))
                            
                            cv2.line(annotated_frame, start_pt, end_pt, (245, 66, 230), 2)
                    
                    # Draw Keypoints
                    for kp in person_kpts:
                         if kp[2] > 0.5: # Use confidence threshold
                            x, y = int(kp[0]), int(kp[1])
                            cv2.circle(annotated_frame, (x, y), 3, (245, 117, 66), -1)

        cv2.imshow('YOLO11 Skeleton', annotated_frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
