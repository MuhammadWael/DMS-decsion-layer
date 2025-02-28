# maindms.py
from DMS import ComprehensiveFacialAnalysis
from logging_system import setup_logging  # Added import

if __name__ == "__main__":
    # Initialize logging at the start of the program
    setup_logging()

    # Replace with your YOLO model path and landmark predictor path
    yolo_model_path = r"best.pt"
    
    facial_analyzer = ComprehensiveFacialAnalysis(yolo_model_path)
    stats = facial_analyzer.process_video()

    print("\nDriver Monitoring Session Statistics:")
    print(f"Total Driving Time: {stats['total_driving_time']:.1f} seconds")
    print(f"Total Distractions: {stats['total_distractions']}")
    print(f"Distraction Rate: {(stats['distracted_time'] / stats['total_driving_time'] * 100):.1f}%")
    print(f"Total Drowsiness Alerts: {stats['total_drowsiness_alerts']}")
    print(f"Drowsiness Time: {stats['drowsiness_time']:.1f} seconds")
    print(f"Eye Closure Drowsiness Time: {stats['eye_closure_drowsiness']:.1f} seconds")
    print(f"Yawn Drowsiness Time: {stats['yawn_drowsiness']:.1f} seconds")
    print(f"Drowsiness Rate: {(stats['drowsiness_time'] / stats['total_driving_time'] * 100):.1f}%")
    # maindms.py
from DMS import ComprehensiveFacialAnalysis
from logging_system import setup_logging, log_abnormal_state

if __name__ == "__main__":
    setup_logging()

    yolo_model_path = r"E:\Graduation Project\decsion layer\best.pt"
    
    facial_analyzer = ComprehensiveFacialAnalysis(yolo_model_path)
    stats = facial_analyzer.process_video()

    print("\n=== Driver Monitoring Session Summary ===")
    print(f"Total Driving Time: {stats['total_driving_time']:.1f} seconds")
    print(f"Total Distractions: {stats['total_distractions']}")
    print(f"Distraction Rate: {(stats['distracted_time'] / stats['total_driving_time'] * 100):.1f}%")
    print(f"Total Drowsiness Alerts: {stats['total_drowsiness_alerts']}")
    print(f"Drowsiness Time: {stats['drowsiness_time']:.1f} seconds")
    print(f"Eye Closure Drowsiness Time: {stats['eye_closure_drowsiness']:.1f} seconds")
    print(f"Yawn Drowsiness Time: {stats['yawn_drowsiness']:.1f} seconds")
    print(f"Drowsiness Rate: {(stats['drowsiness_time'] / stats['total_driving_time'] * 100):.1f}%")
    print("========================================")

