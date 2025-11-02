"""
UDP Server Configuration
Centralized configuration for the face detection UDP server
"""

import os


class ServerConfig:
    """Server configuration with defaults"""

    # Network Settings
    HOST = "0.0.0.0"  # Listen on all interfaces
    PORT_RECEIVE = 5555  # Receive frames from Godot
    PORT_SEND = 5556  # Send detection results to Godot

    # Frame Processing Settings
    MAX_QUEUE_SIZE = 2  # Max frames in processing queue (drop old frames)
    PROCESS_TIMEOUT_SEC = 1.0  # Max time to process one frame

    # Model Paths (relative to backend directory)
    SVM_MODEL_PATH = "models/svm_model.pkl"
    BOVW_ENCODER_PATH = "models/bovw_encoder.pkl"
    SCALER_PATH = "models/scaler.pkl"

    # Mask path (not used by UDP server, but required by FaceDetectionSystem)
    # Point to any mask in your assets folder
    MASK_PATH = "assets/mask.png"  # Relative to backend/

    # Cascade Paths (will auto-detect from OpenCV)
    CASCADE_FACE = "haarcascade_frontalface_default.xml"
    CASCADE_EYE = None  # Use custom ORB eye detector

    # Detection Settings
    MAX_KEYPOINTS = 500
    ENABLE_ROTATION = True
    USE_CUSTOM_EYE_DETECTOR = True
    CUSTOM_EYE_METHOD = "orb"

    # Performance Settings
    WORKER_THREADS = 2  # Number of processing threads

    # Logging
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_FILE = None  # Set to path for file logging

    @classmethod
    def validate(cls):
        """Validate configuration and check required files"""
        errors = []

        # Check model files
        if not os.path.exists(cls.SVM_MODEL_PATH):
            errors.append(f"SVM model not found: {cls.SVM_MODEL_PATH}")
        if not os.path.exists(cls.BOVW_ENCODER_PATH):
            errors.append(f"BoVW encoder not found: {cls.BOVW_ENCODER_PATH}")
        if not os.path.exists(cls.SCALER_PATH):
            errors.append(f"Scaler not found: {cls.SCALER_PATH}")

        return errors

    @classmethod
    def from_env(cls):
        """Override config from environment variables"""
        cls.HOST = os.getenv("UDP_HOST", cls.HOST)
        cls.PORT_RECEIVE = int(os.getenv("UDP_PORT_RECEIVE", cls.PORT_RECEIVE))
        cls.PORT_SEND = int(os.getenv("UDP_PORT_SEND", cls.PORT_SEND))
        cls.LOG_LEVEL = os.getenv("LOG_LEVEL", cls.LOG_LEVEL)

        # Model paths
        cls.SVM_MODEL_PATH = os.getenv("SVM_MODEL_PATH", cls.SVM_MODEL_PATH)
        cls.BOVW_ENCODER_PATH = os.getenv("BOVW_ENCODER_PATH", cls.BOVW_ENCODER_PATH)
        cls.SCALER_PATH = os.getenv("SCALER_PATH", cls.SCALER_PATH)

        return cls


# Development vs Production presets
class DevelopmentConfig(ServerConfig):
    """Development configuration (localhost only)"""

    HOST = "127.0.0.1"
    LOG_LEVEL = "DEBUG"


class ProductionConfig(ServerConfig):
    """Production configuration (LAN accessible)"""

    HOST = "0.0.0.0"
    LOG_LEVEL = "INFO"
    WORKER_THREADS = 4
