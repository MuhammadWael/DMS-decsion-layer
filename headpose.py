# headpose.py
import cv2
import dlib
import numpy as np
from ultralytics import YOLO
from logging_system import setup_logging, log_abnormal_state  # Added import

class HeadPoseEstimator:
    def __init__(self, yolo_model_path, camera_matrix=None, distortion_coeffs=None):
        self.detector = YOLO(yolo_model_path)
        self.predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')
        self.model_points = self._get_3d_model_points()

        if camera_matrix is None:
            self.camera_matrix = np.array([
                [1000, 0, 640],
                [0, 1000, 360],
                [0, 0, 1]
            ], dtype=np.float64)
        else:
            self.camera_matrix = camera_matrix

        if distortion_coeffs is None:
            self.distortion_coeffs = np.zeros((4, 1), dtype=np.float64)
        else:
            self.distortion_coeffs = distortion_coeffs

        # Initialize logging
        setup_logging()

    def _get_3d_model_points(self):
        return np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

    def _get_2d_image_points(self, landmarks):
        return np.array([
            (landmarks.part(30).x, landmarks.part(30).y),
            (landmarks.part(8).x, landmarks.part(8).y),
            (landmarks.part(36).x, landmarks.part(36).y),
            (landmarks.part(45).x, landmarks.part(45).y),
            (landmarks.part(48).x, landmarks.part(48).y),
            (landmarks.part(54).x, landmarks.part(54).y)
        ], dtype=np.float64)

    def _calculate_euler_angles(self, rotation_matrix):
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        return np.degrees([x, y, z])

    def _determine_head_position(self, angles):
        yaw, pitch, roll = angles[2], angles[1], angles[0]
        
        if yaw > 15:
            return "Looking Right"
        elif yaw < -15:
            return "Looking Left"
        elif pitch > 10:
            return "Looking Down"
        elif pitch < -10:
            return "Looking Up"
        elif roll > 20 or roll < -20:
            return "Head Dropped"
        else:
            return "Neutral"

    def estimate_head_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector(rgb_frame, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = self.predictor(gray, dlib_rect)
                image_points = self._get_2d_image_points(landmarks)

                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.model_points, image_points, self.camera_matrix, self.distortion_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                angles = self._calculate_euler_angles(rotation_matrix)
                head_position = self._determine_head_position(angles)
                
                # Log distraction if head position is not "Neutral"
                if head_position != "Neutral":
                    log_abnormal_state("DISTRACTION", head_position)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, head_position, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame
    
class ComprehensiveFacialAnalysis:
    def __init__(self, yolo_model_path, camera_matrix=None, distortion_coeffs=None):
        """
        Initialize Comprehensive Facial Analysis
        
        Args:
            yolo_model_path: Path to trained YOLOv8n model weights
            camera_matrix: Optional camera calibration matrix
            distortion_coeffs: Optional lens distortion coefficients
        """
        # Load YOLO model for face detection
        self.face_detector = YOLO(yolo_model_path)
        
        # Load landmark predictor (still using dlib for landmarks)
        self.predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')
        
        # Define 3D model points (facial landmarks in 3D space)
        self.model_points = self._get_3d_model_points()
        
        # Camera matrix and distortion coefficients
        if camera_matrix is None:
            # Default camera matrix (approximate values)
            self.camera_matrix = np.array([
                [1000, 0, 640],    # fx, 0, cx
                [0, 1000, 360],    # 0, fy, cy
                [0, 0, 1]          # 0, 0, 1
            ], dtype=np.float64)
        else:
            self.camera_matrix = camera_matrix
        
        if distortion_coeffs is None:
            self.distortion_coeffs = np.zeros((4, 1), dtype=np.float64)
        else:
            self.distortion_coeffs = distortion_coeffs
        
        # Landmark color palette
        self.landmark_colors = self._create_landmark_color_palette()

        # Initialize logging
        setup_logging()

    def _create_landmark_color_palette(self):
        """Create a color palette for different facial regions"""
        colors = {
            'jaw': (255, 0, 0),
            'left_eyebrow': (0, 255, 0),
            'right_eyebrow': (0, 255, 0),
            'left_eye': (0, 0, 255),
            'right_eye': (0, 0, 255),
            'nose_bridge': (255, 255, 0),
            'nose_tip': (255, 0, 255),
            'mouth_outer': (0, 255, 255),
            'mouth_inner': (128, 128, 255),
            'default': (255, 255, 255)
        }
        return colors

    def _get_3d_model_points(self):
        """Define 3D points for head pose estimation"""
        model_points = np.array([
            (0.0, 0.0, 0.0),              # Nose tip
            (0.0, -330.0, -65.0),         # Chin
            (-225.0, 170.0, -135.0),      # Left eye left corner
            (225.0, 170.0, -135.0),       # Right eye right corner
            (-150.0, -150.0, -125.0),     # Left Mouth corner
            (150.0, -150.0, -125.0)       # Right mouth corner
        ], dtype=np.float64)
        return model_points

    def _convert_yolo_to_dlib_rect(self, yolo_box, image_shape):
        """
        Convert YOLO format bounding box to dlib rectangle
        
        Args:
            yolo_box: YOLO format bounding box (x1, y1, x2, y2)
            image_shape: Shape of the image (height, width)
        
        Returns:
            dlib.rectangle object
        """
        x1, y1, x2, y2 = map(int, yolo_box)
        return dlib.rectangle(x1, y1, x2, y2)

    def _draw_landmarks(self, frame, landmarks):
        """Draw all 68 facial landmarks with color-coded regions"""
        regions = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_tip': list(range(31, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'mouth_outer': list(range(48, 60)),
            'mouth_inner': list(range(60, 68))
        }
        
        for region, point_indices in regions.items():
            color = self.landmark_colors.get(region, self.landmark_colors['default'])
            for i in point_indices:
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(frame, (x, y), 2, color, -1)
                cv2.putText(frame, str(i), (x+3, y+3), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)

    def _get_2d_image_points(self, landmarks):
        """Extract 2D landmark points for pose estimation"""
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
        ], dtype=np.float64)
        return image_points

    def _calculate_euler_angles(self, rotation_matrix):
        """Calculate Euler angles from rotation matrix"""
        sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  
                    rotation_matrix[1,0] * rotation_matrix[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = 0
        
        return np.array([np.degrees(x), np.degrees(y), np.degrees(z)])

    def estimate_head_pose(self, frame):
        """Estimate head pose and draw landmarks"""
        # Convert to RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using YOLO
        results = self.face_detector(rgb_frame, verbose=False)
        
        for result in results:
            # Process each detected face
            for box in result.boxes:
                # Convert box to dlib rectangle format
                dlib_rect = self._convert_yolo_to_dlib_rect(box.xyxy[0], frame.shape)
                
                # Get facial landmarks
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = self.predictor(gray, dlib_rect)
                
                # Draw landmarks
                self._draw_landmarks(frame, landmarks)
                
                # Get 2D image points for pose estimation
                image_points = self._get_2d_image_points(landmarks)
                
                # Solve PnP
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.model_points, 
                    image_points, 
                    self.camera_matrix, 
                    self.distortion_coeffs
                )
                
                # Convert rotation vector to matrix and calculate angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                angles = self._calculate_euler_angles(rotation_matrix)
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Annotate head pose angles
                pose_text = f"Yaw: {angles[2]:.2f}, Pitch: {angles[1]:.2f}, Roll: {angles[0]:.2f}"
                cv2.putText(frame, pose_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame