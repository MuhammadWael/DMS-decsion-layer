# main.py
import cv2
import time
from headpose import HeadPoseEstimator
from eye_closure import EyeClosureDetector
from eye_gaze import EyeGazeEstimator
from yawn_detection import YawnDetector  # Ensure this module is saved and imported correctly
from logging_system import setup_logging  

def main():
    # Initialize logging at the start of the program
    setup_logging()

    # Paths to models
    yolo_model_path = r"best.pt"  
    shape_predictor_path = r"shape_predictor_68_face_landmarks.dat"  

    # Initialize all components
    head_pose_estimator = HeadPoseEstimator(yolo_model_path)
    eye_closure_detector = EyeClosureDetector(yolo_model_path)
    eye_gaze_estimator = EyeGazeEstimator(yolo_model_path, shape_predictor_path)
    yawn_detector = YawnDetector(yolo_model_path)

    # Open webcam
    cap = cv2.VideoCapture(0)

    fps_counter = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()  # Start time for this frame
        
        # Head Pose Estimation
        frame = head_pose_estimator.estimate_head_pose(frame)
        
        # Eye Closure Detection
        frame, eyes_closed = eye_closure_detector.detect_eye_closure(frame)
        
        # Eye Gaze Estimation
        frame = eye_gaze_estimator.estimate_gaze(frame)

        # Yawn Detection
        frame, is_yawn = yawn_detector.detect_yawn(frame)

        # Measure FPS
        fps_counter += 1
        elapsed_time = time.time() - start_time
        fps = fps_counter / elapsed_time if elapsed_time > 0 else 0

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the final frame
        cv2.imshow("Driver Monitoring System", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()