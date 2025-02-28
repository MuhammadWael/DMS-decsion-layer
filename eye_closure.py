# eye_closure.py
import cv2
import dlib
import numpy as np
import time
from ultralytics import YOLO
from scipy.spatial import distance as dist
from headpose import HeadPoseEstimator
from logging_system import setup_logging, log_abnormal_state  

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

LEFT_EYE_START, LEFT_EYE_END = 36, 42
RIGHT_EYE_START, RIGHT_EYE_END = 42, 48

class EyeClosureDetector(HeadPoseEstimator):
    def __init__(self, yolo_model_path, camera_matrix=None, distortion_coeffs=None, ear_threshold=0.3, close_duration=2.0):
        super().__init__(yolo_model_path, camera_matrix, distortion_coeffs)
        self.ear_threshold = ear_threshold
        self.close_duration = close_duration
        self.eye_closed_start_time = None

        # Initialize logging
        setup_logging()

    def _get_eye_landmarks(self, landmarks, eye_start, eye_end):
        return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(eye_start, eye_end)])

    def detect_eye_closure(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector(rgb_frame, verbose=False)
        is_eyes_closed = False

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = self.predictor(gray, dlib_rect)

                left_eye = self._get_eye_landmarks(landmarks, LEFT_EYE_START, LEFT_EYE_END)
                right_eye = self._get_eye_landmarks(landmarks, RIGHT_EYE_START, RIGHT_EYE_END)

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < self.ear_threshold:
                    if self.eye_closed_start_time is None:
                        self.eye_closed_start_time = time.time()
                    elif time.time() - self.eye_closed_start_time >= self.close_duration:
                        is_eyes_closed = True
                        cv2.putText(frame, "Eyes Closed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        log_abnormal_state("DROWSINESS", "Eye Closure")  # Added logging
                else:
                    self.eye_closed_start_time = None
                    cv2.putText(frame, "Eyes Open", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame, is_eyes_closed