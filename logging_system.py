# logging_system.py
import logging
from datetime import datetime
import os

def setup_logging(log_file=f"driver_monitoring_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"):
    """Set up logging to a file with a timestamped name."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # Create logs directory if it doesn't exist
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging initialized")

def log_abnormal_state(state_type, details):
    """Log an abnormal state with type and details."""
    logging.info(f"{state_type}: {details}")