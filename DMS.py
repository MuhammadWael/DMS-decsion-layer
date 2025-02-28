# dms.py
import cv2
import dlib
import numpy as np
from ultralytics import YOLO
import pygame
import time
from scipy.spatial import distance as dist
from logging_system import setup_logging, log_abnormal_state

class Pupil:
    def __init__(self):
        """Initialize Pupil detection parameters."""
        self.bilateral_filter_d = 9
        self.bilateral_filter_sigma_color = 75
        self.bilateral_filter_sigma_space = 75
        self.erosion_kernel_size = 3
        self.erosion_kernel = np.ones((self.erosion_kernel_size, self.erosion_kernel_size), np.uint8)
        self.binarization_threshold = 30

    def preprocess_eye_frame(self, eye_frame):
        """Preprocess the eye frame."""
        if len(eye_frame.shape) == 3:
            eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        filtered_frame = cv2.bilateralFilter(
            eye_frame, self.bilateral_filter_d,
            self.bilateral_filter_sigma_color,
            self.bilateral_filter_sigma_space
        )
        eroded_frame = cv2.erode(filtered_frame, self.erosion_kernel, iterations=1)
        _, binary_frame = cv2.threshold(eroded_frame, self.binarization_threshold, 255, cv2.THRESH_BINARY_INV)
        return binary_frame

    def detect_pupil(self, eye_frame):
        """Detect the pupil in the eye frame."""
        binary_frame = self.preprocess_eye_frame(eye_frame)
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                pupil_x = int(moments["m10"] / moments["m00"])
                pupil_y = int(moments["m01"] / moments["m00"])
                return (pupil_x, pupil_y), largest_contour
        return None, None

class ComprehensiveFacialAnalysis:
    def __init__(self, yolo_model_path, camera_matrix=None, distortion_coeffs=None):
        """Initialize Comprehensive Facial Analysis."""
        self.face_detector = YOLO(yolo_model_path)
        self.predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')  # Path
        self.model_points = self._get_3d_model_points()
        self.camera_matrix = camera_matrix if camera_matrix is not None else np.array(
            [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float64
        )
        self.distortion_coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros((4, 1), dtype=np.float64)
        self.landmark_colors = self._create_landmark_color_palette()

        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound(r'echo-alert-177739.mp3')  # Path
        self.last_alert_time = 0
        self.alert_cooldown = 2.0

        self.threshold_yaw = 30
        self.threshold_pitch = 20
        self.distraction_frame_threshold = 30
        self.distraction_frame_count = 0

        self.EAR_THRESHOLD = 0.25
        self.MAR_THRESHOLD = 0.7
        self.drowsiness_frame_threshold = 50
        self.eye_closure_frame_count = 0
        self.yawn_frame_count = 0

        self.frame_rate = 30  # Initial, updated dynamically
        self.distracted_time = 0
        self.drowsiness_time = 0
        self.pupil_detector = Pupil()

        self.stats = {
            "total_distractions": 0, "left_distractions": 0, "right_distractions": 0,
            "up_distractions": 0, "down_distractions": 0, "total_driving_time": 0,
            "distracted_time": 0, "total_drowsiness_alerts": 0, "drowsiness_time": 0,
            "eye_closure_drowsiness": 0, "yawn_drowsiness": 0
        }
        self.session_start_time = time.time()
        self.prev_frame_time = 0

        # Initialize logging
        setup_logging()

    def _create_landmark_color_palette(self):
        """Create a color palette for facial regions."""
        return {
            'jaw': (255, 0, 0), 'left_eyebrow': (0, 255, 0), 'right_eyebrow': (0, 255, 0),
            'left_eye': (0, 0, 255), 'right_eye': (0, 0, 255), 'nose_bridge': (255, 255, 0),
            'nose_tip': (255, 0, 255), 'mouth_outer': (0, 255, 255), 'mouth_inner': (128, 128, 255),
            'default': (255, 255, 255)
        }

    def _get_3d_model_points(self):
        """Define 3D points for head pose estimation."""
        return np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ], dtype=np.float64)

    def _convert_yolo_to_dlib_rect(self, yolo_box, image_shape):
        """Convert YOLO bounding box to dlib rectangle."""
        x1, y1, x2, y2 = map(int, yolo_box)
        return dlib.rectangle(x1, y1, x2, y2)

    def _draw_landmarks(self, frame, landmarks):
        """Draw facial landmarks."""
        regions = {
            'jaw': list(range(0, 17)), 'right_eyebrow': list(range(17, 22)), 'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)), 'nose_tip': list(range(31, 36)), 'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)), 'mouth_outer': list(range(48, 60)), 'mouth_inner': list(range(60, 68))
        }
        for region, point_indices in regions.items():
            color = self.landmark_colors.get(region, self.landmark_colors['default'])
            for i in point_indices:
                x, y = landmarks.part(i).x, landmarks.part(i).y
                cv2.circle(frame, (x, y), 2, color, -1)
                cv2.putText(frame, str(i), (x+3, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)

    def _get_2d_image_points(self, landmarks):
        """Extract 2D landmark points."""
        return np.array([
            (landmarks.part(30).x, landmarks.part(30).y),
            (landmarks.part(8).x, landmarks.part(8).y),
            (landmarks.part(36).x, landmarks.part(36).y),
            (landmarks.part(45).x, landmarks.part(45).y),
            (landmarks.part(48).x, landmarks.part(48).y),
            (landmarks.part(54).x, landmarks.part(54).y)
        ], dtype=np.float64)

    def _calculate_euler_angles(self, rotation_matrix):
        """Calculate Euler angles."""
        sy = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = 0
        return np.degrees([x, y, z])

    def _analyze_head_pose(self, angles):
        """Analyze head pose for distraction."""
        roll, pitch, yaw = angles
        if abs(yaw) > self.threshold_yaw:
            return True, "RIGHT" if yaw > 0 else "LEFT"
        if abs(pitch) > self.threshold_pitch:
            return True, "UP" if pitch > 0 else "DOWN"
        return False, "ATTENTIVE"

    def _calculate_ear(self, landmarks):
        """Calculate Eye Aspect Ratio."""
        def eye_aspect_ratio(eye_points):
            p2_p6 = np.linalg.norm(np.array([landmarks.part(eye_points[1]).x, landmarks.part(eye_points[1]).y]) -
                                   np.array([landmarks.part(eye_points[5]).x, landmarks.part(eye_points[5]).y]))
            p3_p5 = np.linalg.norm(np.array([landmarks.part(eye_points[2]).x, landmarks.part(eye_points[2]).y]) -
                                   np.array([landmarks.part(eye_points[4]).x, landmarks.part(eye_points[4]).y]))
            p1_p4 = np.linalg.norm(np.array([landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y]) -
                                   np.array([landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y]))
            return (p2_p6 + p3_p5) / (2.0 * p1_p4)

        left_eye_points = [36, 37, 38, 39, 40, 41]
        right_eye_points = [42, 43, 44, 45, 46, 47]
        return (eye_aspect_ratio(left_eye_points) + eye_aspect_ratio(right_eye_points)) / 2.0

    def _calculate_mar(self, landmarks):
        """Calculate Mouth Aspect Ratio."""
        p2_p8 = np.linalg.norm(np.array([landmarks.part(50).x, landmarks.part(50).y]) -
                               np.array([landmarks.part(58).x, landmarks.part(58).y]))
        p3_p7 = np.linalg.norm(np.array([landmarks.part(51).x, landmarks.part(51).y]) -
                               np.array([landmarks.part(57).x, landmarks.part(57).y]))
        p4_p6 = np.linalg.norm(np.array([landmarks.part(52).x, landmarks.part(52).y]) -
                               np.array([landmarks.part(56).x, landmarks.part(56).y]))
        p4_p0 = np.linalg.norm(np.array([landmarks.part(54).x, landmarks.part(54).y]) -
                               np.array([landmarks.part(48).x, landmarks.part(48).y]))
        return (p2_p8 + p3_p7 + p4_p6) / (2.0 * p4_p0)

    def _analyze_drowsiness(self, landmarks, angles):
        """Analyze drowsiness."""
        ear = self._calculate_ear(landmarks)
        mar = self._calculate_mar(landmarks)
        roll, pitch, yaw = angles

        is_drowsy = False
        drowsiness_type = "ATTENTIVE"

        if ear < self.EAR_THRESHOLD:
            is_drowsy = True
            drowsiness_type = "EYE_CLOSED"
        elif mar > self.MAR_THRESHOLD:
            is_drowsy = True
            drowsiness_type = "YAWN"

        if is_drowsy and (abs(yaw) > self.threshold_yaw or abs(pitch) > self.threshold_pitch):
            drowsiness_type += "_POSE"
        return is_drowsy, drowsiness_type, ear, mar

    def _trigger_alert(self, alert_type):
        """Trigger audio alert."""
        current_time = time.time()
        if current_time - self.last_alert_time >= self.alert_cooldown:
            self.alert_sound.play()
            self.last_alert_time = current_time
            return True
        return False

    def _update_statistics(self, alert_type):
        """Update statistics."""
        if alert_type == "DROWSINESS":
            self.stats["total_drowsiness_alerts"] += 1
        elif alert_type == "DISTRACTION":
            self.stats["total_distractions"] += 1

    def _detect_eye_distraction(self, landmarks):
        """Detect eye distraction."""
        left_eye_center = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0)
        right_eye_center = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0)
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
        left_eye_dist = left_eye_center[0] - nose_tip[0]
        right_eye_dist = right_eye_center[0] - nose_tip[0]
        if left_eye_dist > 20:
            return "LEFT"
        elif right_eye_dist < -20:
            return "RIGHT"
        return "ATTENTIVE"

    def _extract_eye_frame(self, frame, landmarks, eye_type):
        """Extract eye frame."""
        eye_points = list(range(36, 42)) if eye_type == "left" else list(range(42, 48))
        eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points])
        x, y, w, h = cv2.boundingRect(eye_region)
        return frame[y:y+h, x:x+w]

    def _local_to_global_coords(self, pupil_coords, landmarks, eye_type):
        """Convert pupil coordinates to global frame."""
        eye_points = list(range(36, 42)) if eye_type == "left" else list(range(42, 48))
        eye_region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points])
        x, y, _, _ = cv2.boundingRect(eye_region)
        pupil_x, pupil_y = pupil_coords
        return (x + pupil_x, y + pupil_y)

    def estimate_head_pose(self, frame):
        """Main analysis function."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector(rgb_frame, verbose=False)
        is_distracted, distraction_type, is_drowsy, drowsiness_type = False, "ATTENTIVE", False, "ATTENTIVE"
        ear, mar = 1.0, 0.0

        for result in results:
            for box in result.boxes:
                dlib_rect = self._convert_yolo_to_dlib_rect(box.xyxy[0], frame.shape)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = self.predictor(gray, dlib_rect)
                self._draw_landmarks(frame, landmarks)
                image_points = self._get_2d_image_points(landmarks)

                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.model_points, image_points, self.camera_matrix, self.distortion_coeffs
                )
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                angles = self._calculate_euler_angles(rotation_matrix)

                is_distracted, distraction_type = self._analyze_head_pose(angles)
                is_drowsy, drowsiness_type, ear, mar = self._analyze_drowsiness(landmarks, angles)

                eye_distraction_type = self._detect_eye_distraction(landmarks)
                if eye_distraction_type != "ATTENTIVE":
                    distraction_type, is_distracted = eye_distraction_type, True

                left_eye_frame = self._extract_eye_frame(frame, landmarks, "left")
                right_eye_frame = self._extract_eye_frame(frame, landmarks, "right")
                left_pupil, _ = self.pupil_detector.detect_pupil(left_eye_frame)
                right_pupil, _ = self.pupil_detector.detect_pupil(right_eye_frame)

                gaze_direction = "CENTER"
                if left_pupil and right_pupil:
                    left_pupil_global = self._local_to_global_coords(left_pupil, landmarks, "left")
                    right_pupil_global = self._local_to_global_coords(right_pupil, landmarks, "right")
                    cv2.circle(frame, left_pupil_global, 3, (0, 255, 0), -1)
                    cv2.circle(frame, right_pupil_global, 3, (0, 255, 0), -1)
                    if left_pupil_global[0] < right_pupil_global[0] - 10:
                        gaze_direction = "LEFT"
                    elif left_pupil_global[0] > right_pupil_global[0] + 10:
                        gaze_direction = "RIGHT"

                    cv2.putText(frame, f"Looking {gaze_direction}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Left pupil: {left_pupil_global}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"Right pupil: {right_pupil_global}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color_distraction = (0, 0, 255) if is_distracted else (0, 255, 0)
                color_drowsiness = (255, 0, 255) if is_drowsy else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_distraction, 2)
                pose_text = f"Yaw:{angles[2]:.1f} Pitch:{angles[1]:.1f}"
                cv2.putText(frame, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_distraction, 1)
                cv2.putText(frame, f"Distr:{distraction_type}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_distraction, 1)
                cv2.putText(frame, f"Drowsy:{drowsiness_type}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_drowsiness, 1)
                cv2.putText(frame, f"EAR:{ear:.2f} MAR:{mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_drowsiness, 1)

        if is_distracted:
            self.distraction_frame_count += 1
            if self.distraction_frame_count >= self.distraction_frame_threshold:
                if self._trigger_alert("DISTRACTION"):
                    self._update_statistics("DISTRACTION")
                    self.stats["distracted_time"] += (1.0 / self.frame_rate)
                    log_abnormal_state("DISTRACTION", distraction_type)
        else:
            self.distraction_frame_count = 0
            distraction_type = "ATTENTIVE"

        if is_drowsy:
            if drowsiness_type == "EYE_CLOSED":
                self.eye_closure_frame_count += 1
                self.yawn_frame_count = 0
                if self.eye_closure_frame_count >= self.drowsiness_frame_threshold:
                    if self._trigger_alert("DROWSINESS"):
                        self._update_statistics("DROWSINESS")
                        self.stats["drowsiness_time"] += (1.0 / self.frame_rate)
                        self.stats["eye_closure_drowsiness"] += (1.0 / self.frame_rate)
                        log_abnormal_state("DROWSINESS", "Eye Closure")
            elif drowsiness_type == "YAWN":
                self.yawn_frame_count += 1
                self.eye_closure_frame_count = 0
                if self.yawn_frame_count >= self.drowsiness_frame_threshold:
                    if self._trigger_alert("DROWSINESS"):
                        self._update_statistics("DROWSINESS")
                        self.stats["drowsiness_time"] += (1.0 / self.frame_rate)
                        self.stats["yawn_drowsiness"] += (1.0 / self.frame_rate)
                        log_abnormal_state("DROWSINESS", "Yawn")
        else:
            self.eye_closure_frame_count = 0
            self.yawn_frame_count = 0
            drowsiness_type = "ATTENTIVE"

        return frame, is_distracted, distraction_type, is_drowsy, drowsiness_type, ear, mar

    def process_video(self, video_source=0):
        """Process video and perform analysis."""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - self.prev_frame_time)
            self.prev_frame_time = new_frame_time
            self.frame_rate = int(fps)

            processed_frame, is_distracted, distraction_type, is_drowsy, drowsiness_type, ear, mar = self.estimate_head_pose(frame)

            total_time = time.time() - self.session_start_time
            distraction_rate = (self.stats["distracted_time"] / total_time) * 100 if total_time > 0 else 0
            drowsiness_rate = (self.stats["drowsiness_time"] / total_time) * 100 if total_time > 0 else 0

            cv2.putText(processed_frame, f"Distr Rate: {distraction_rate:.1f}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(processed_frame, f"Drowsy Rate: {drowsiness_rate:.1f}%", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(processed_frame, f"FPS: {self.frame_rate}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Facial Analysis - Distraction & Drowsiness', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.stats["total_driving_time"] = time.time() - self.session_start_time
        return self.stats