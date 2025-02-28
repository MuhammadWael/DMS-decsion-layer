# yawn_detection.py
import cv2
import numpy as np
from ultralytics import YOLO
import dlib
from headpose import ComprehensiveFacialAnalysis
from logging_system import setup_logging, log_abnormal_state  # Added import

class YawnDetector(ComprehensiveFacialAnalysis):
    def __init__(self, yolo_model_path, camera_matrix=None, distortion_coeffs=None, yawn_threshold=0.3):
        """
        Initialize Yawn Detector
        
        Args:
            yolo_model_path: Path to trained YOLOv8 model weights
            camera_matrix: Optional camera calibration matrix
            distortion_coeffs: Optional lens distortion coefficients
            yawn_threshold: Threshold for yawn detection (default: 0.7)
        """
        super().__init__(yolo_model_path, camera_matrix, distortion_coeffs)
        self.yawn_threshold = yawn_threshold

        # Initialize logging
        setup_logging()

    def _calculate_mean_lip_positions(self, landmarks):
        """
        Calculate the mean positions of the upper and lower lips
        
        Args:
            landmarks: dlib facial landmarks
        
        Returns:
            upper_mean: Mean position of the upper lip (x, y)
            lower_mean: Mean position of the lower lip (x, y)
        """
        upper_lip_points = [50, 51, 52, 61, 62, 63]
        lower_lip_points = [56, 57, 58, 65, 66, 67]
        
        upper_mean = np.mean([[landmarks.part(i).x, landmarks.part(i).y] for i in upper_lip_points], axis=0)
        lower_mean = np.mean([[landmarks.part(i).x, landmarks.part(i).y] for i in lower_lip_points], axis=0)
        
        return upper_mean, lower_mean

    def _calculate_vertical_distance(self, upper_mean, lower_mean):
        """
        Calculate the vertical distance between the upper and lower lips
        
        Args:
            upper_mean: Mean position of the upper lip (x, y)
            lower_mean: Mean position of the lower lip (x, y)
        
        Returns:
            distance: Normalized vertical distance between the upper and lower lips
        """
        return abs(upper_mean[1] - lower_mean[1]) / 100

    def detect_yawn(self, frame):
        """
        Detect yawning in the frame
        
        Args:
            frame: Input frame
        
        Returns:
            frame: Frame with yawn detection text
            is_yawn: Boolean indicating if a yawn is detected
        """
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using YOLO
        results = self.face_detector(rgb_frame, verbose=False)
        
        is_yawn = False

        for result in results:
            for box in result.boxes:
                # Convert to dlib format
                dlib_rect = self._convert_yolo_to_dlib_rect(box.xyxy[0], frame.shape)

                # Get facial landmarks
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = self.predictor(gray, dlib_rect)

                # Calculate lip positions and distance
                upper_mean, lower_mean = self._calculate_mean_lip_positions(landmarks)
                distance = self._calculate_vertical_distance(upper_mean, lower_mean)

                # Define text position (bottom left)
                text_x, text_y = 20, frame.shape[0] - 30
                
                # Determine yawn status
                if distance > self.yawn_threshold:
                    is_yawn = True
                    yawn_text = "Yawning Detected!"
                    text_color = (0, 0, 255)  # Red for warning
                    log_abnormal_state("DROWSINESS", "Yawn")  # Added logging
                else:
                    yawn_text = "Mouth Closed"
                    text_color = (0, 255, 0)  # Green for normal

                # Display yawn text on screen
                cv2.putText(frame, yawn_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        return frame, is_yawn