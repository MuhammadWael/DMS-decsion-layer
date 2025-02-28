# eye_gaze.py
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from ultralytics import YOLO
import time
from logging_system import setup_logging, log_abnormal_state  

class Pupil:
    def __init__(self, eye_frame):
        self.eye_frame = eye_frame
        self.pupil_position = None

    def detect_pupil(self):
        # Convert to grayscale and apply adaptive threshold
        gray_eye = cv2.cvtColor(self.eye_frame, cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)  # Reduce noise

        # Adaptive Thresholding
        _, binary_eye = cv2.threshold(gray_eye, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        if contours:
            iris_contour = contours[0]
            (cx, cy), radius = cv2.minEnclosingCircle(iris_contour)
            self.pupil_position = (int(cx), int(cy))

        return self.pupil_position


class Eye:
    def __init__(self, landmarks, eye_points):
        self.landmarks = landmarks
        self.eye_points = eye_points
        self.blinking_ratio = None
        self.pupil = None

    def isolate_eye(self, frame):
        eye_region = np.array([(self.landmarks.part(point).x, self.landmarks.part(point).y) for point in self.eye_points])
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [eye_region], 255)
        eye_frame = cv2.bitwise_and(frame, frame, mask=mask)

        min_x, max_x = min(eye_region[:, 0]), max(eye_region[:, 0])
        min_y, max_y = min(eye_region[:, 1]), max(eye_region[:, 1])
        eye_frame = eye_frame[min_y:max_y, min_x:max_x]

        return eye_frame

    def calculate_blinking_ratio(self):
        horizontal_dist = dist.euclidean(
            (self.landmarks.part(self.eye_points[0]).x, self.landmarks.part(self.eye_points[0]).y),
            (self.landmarks.part(self.eye_points[3]).x, self.landmarks.part(self.eye_points[3]).y)
        )
        vertical_dist = dist.euclidean(
            (self.landmarks.part(self.eye_points[1]).x, self.landmarks.part(self.eye_points[1]).y),
            (self.landmarks.part(self.eye_points[2]).x, self.landmarks.part(self.eye_points[2]).y)
        )
        self.blinking_ratio = horizontal_dist / vertical_dist
        return self.blinking_ratio

    def detect_pupil(self, eye_frame):
        self.pupil = Pupil(eye_frame)
        return self.pupil.detect_pupil()


class EyeGazeEstimator:
    def __init__(self, yolo_model_path, shape_predictor_path):
        self.face_detector = YOLO(yolo_model_path)
        self.landmark_predictor = dlib.shape_predictor(shape_predictor_path)
        self.left_eye_points = [36, 37, 38, 39, 40, 41]
        self.right_eye_points = [42, 43, 44, 45, 46, 47]
        self.gaze_start_time = None
        self.last_gaze_direction = "Center"
        self.threshold_duration = 2  # seconds

        # Initialize logging
        setup_logging()

    def estimate_gaze(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector(rgb_frame, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = self.landmark_predictor(gray, dlib_rect)

                left_eye = Eye(landmarks, self.left_eye_points)
                right_eye = Eye(landmarks, self.right_eye_points)

                left_eye_frame = left_eye.isolate_eye(frame)
                right_eye_frame = right_eye.isolate_eye(frame)

                left_pupil = left_eye.detect_pupil(left_eye_frame)
                right_pupil = right_eye.detect_pupil(right_eye_frame)

                if left_pupil and right_pupil:
                    gaze_direction = self._calculate_gaze_direction(left_pupil, right_pupil, left_eye_frame, right_eye_frame)
                    
                    # Log distraction if gaze is not "Center" and stable for threshold duration
                    if gaze_direction in ["Left", "Right"]:
                        log_abnormal_state("DISTRACTION", gaze_direction)

                    # Display text on right of screen
                    frame_height, frame_width = frame.shape[:2]
                    text_x = frame_width - 180  # Adjust X to position text on the right
                    text_y = 50  

                    cv2.rectangle(frame, (text_x - 10, text_y - 30), (text_x + 150, text_y + 10), (0, 0, 0), -1)
                    cv2.putText(frame, gaze_direction, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def _calculate_gaze_direction(self, left_pupil, right_pupil, left_eye_frame, right_eye_frame):
            left_eye_center = (left_eye_frame.shape[1] // 2, left_eye_frame.shape[0] // 2)
            right_eye_center = (right_eye_frame.shape[1] // 2, right_eye_frame.shape[0] // 2)

            left_norm = left_pupil[0] / left_eye_frame.shape[1]
            right_norm = right_pupil[0] / right_eye_frame.shape[1]

            if left_norm < 0.4 and right_norm < 0.4:
                current_gaze = "Left"
            elif left_norm > 0.6 and right_norm > 0.6:
                current_gaze = "Right"
            else:
                current_gaze = "Center"

            # Check if gaze direction remains unchanged for threshold duration
            if current_gaze == self.last_gaze_direction:
                if self.gaze_start_time is None:
                    self.gaze_start_time = time.time()
                elif time.time() - self.gaze_start_time >= self.threshold_duration:
                    return current_gaze
            else:
                self.gaze_start_time = time.time()
                self.last_gaze_direction = current_gaze

            return "Center"  # Default return when gaze isn't stable for 2 seconds