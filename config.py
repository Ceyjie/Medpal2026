# ============================================================
# MedPal config
# ============================================================

# ------------------------------------------------------------
# Frame settings
# ------------------------------------------------------------
FRAME_W, FRAME_H = 640, 480
CONF_THRES = 0.35
COSINE_THRES = 0.5

# ------------------------------------------------------------
# Camera calibration (Astra Pro)
# ------------------------------------------------------------
# These are best-effort defaults. The planner self-calibrates
# against the depth image at runtime, so exact values here matter
# less than the ratio between them.
CAMERA_HEIGHT_MM = 200      # camera height above floor
CAMERA_TILT_DEG = 15        # downward tilt
CAMERA_HFOV_DEG = 60        # horizontal FOV (Astra Pro RGB)
CAMERA_VFOV_DEG = 49.5      # vertical FOV

MIN_VALID_DEPTH_MM = 55     # depths below this are treated as invalid

# ------------------------------------------------------------
# Distance zones (mm)
# ------------------------------------------------------------
# The Astra Pro's reliable range starts around 300 mm. Faces are
# detected up to ~1500 mm. These thresholds define the follow
# behavior zones.
REVERSE_DISTANCE_MM = 65   # emergency back-off below this
FOLLOW_MIN_MM = 95         # stop-and-turn zone starts here
FOLLOW_MAX_MM = 900         # full forward speed at or above
OBSTACLE_STOP_MM = 80      # stop if center sector closer than this
FACE_RECOGNIZE_MAX_MM = 1200  # only scan faces closer than this

# Kept for legacy references (unused by new follow logic).
FORWARD_DISTANCE_MM = 100
STOP_ZONE_MIN_MM = 60
STOP_ZONE_MAX_MM = 100
DISTANCE_THRESHOLD_MM = 100
MIN_DISTANCE_MM = 85

# ------------------------------------------------------------
# Follow speed ramp (%)
# ------------------------------------------------------------
FOLLOW_BASE_SPEED = 40       # legacy default; used as fallback
REVERSE_SPEED = 30           # speed when backing up
SPEED_INCREASE = 10
MAX_SPEED = 100

MIN_FOLLOW_SPEED = 30        # % at FOLLOW_MIN_MM
MAX_FOLLOW_SPEED = 55        # % at FOLLOW_MAX_MM
TURN_ONLY_SPEED = 40         # % pivot speed when off-axis and stopped
TURN_GAIN = 0.6              # differential fraction of forward speed
DEADBAND_PX = 40             # ±px of frame center considered "on axis"

# ------------------------------------------------------------
# Obstacle avoidance
# ------------------------------------------------------------
OBSTACLE_SLOW_MM = 350
OBSTACLE_SIDE_BIAS = 25

# ------------------------------------------------------------
# Path planner
# ------------------------------------------------------------
OCCUPANCY_GRID_SIZE = 60
OCCUPANCY_GRID_RES_MM = 50
ROBOT_RADIUS_MM = 200       # half the widest dimension of the chassis
PLANNER_DISABLED = False

# ------------------------------------------------------------
# Detection thresholds
# ------------------------------------------------------------
BODY_CONF_THRES = 0.60
MIN_BODY_BOX_W = 60
MIN_BODY_BOX_H = 120
MIN_BODY_BOX_AREA = 10000   # pixels

STRICT_THRES = 0.60

# ------------------------------------------------------------
# Identity lock
# ------------------------------------------------------------
IDENTITY_LOCK_MIN_CONF = 0.80
IDENTITY_LOCK_RELEASE_S = 3.0
SKIP_SCANS_WHEN_LOCKED = True
ASSUMED_FPS = 8.0

FACE_SCAN_COOLDOWN_S = 2.0

# ------------------------------------------------------------
# Re-identification
# ------------------------------------------------------------
REID_LOCK_FRAMES = 3
REID_LOSS_FRAMES = 7
REID_PATH = "/home/medpal/tracking_person/models/reid.onnx"

# ------------------------------------------------------------
# Face recognition
# ------------------------------------------------------------
FACE_SVM_CONFIDENCE_THRES = 0.75

# ------------------------------------------------------------
# Enrollment quality
# ------------------------------------------------------------
ENROLL_MIN_FACE_PX = 50         # minimum face width/height in pixels
ENROLL_BLUR_THRES = 40.0        # Laplacian variance; below = too blurry
ENROLL_DIVERSITY_PX = 25        # skip frames too similar to last
ENROLL_DEFAULT_COUNT = 30       # default crops per person

AUTO_SAVE_RECOGNIZED_CROPS = True

FACE_TRAINING_DIR = "/home/medpal/tracking_person/face_training"

# ------------------------------------------------------------
# Model paths
# ------------------------------------------------------------
# SCRFD + ArcFace (CPU, ONNX)
SCRFD_MODEL_PATH = "/home/medpal/2026medpal/models/scrfd_arcface/det_2.5g.onnx"
ARCFACE_MODEL_PATH = "/home/medpal/2026medpal/models/scrfd_arcface/w600k_mbf.onnx"
SCRFD_INPUT_SIZE = (640, 640)
SCRFD_CONF_THRES = 0.5
SCRFD_IOU_THRES = 0.4
ARCFACE_EMBEDDING_SIZE = 512

# Classifier written by train_face_svm.py, read by FaceRecognizer
SVM_MODEL_PATH = "/home/medpal/tracking_person/models/arcface_classifier.pkl"

# Coral detection models
CORAL_DETECTION_MODEL = "/home/medpal/tracking_person/models/tflite/ssd_mobilenet_v2_edgetpu.tflite"
CORAL_DETECTION_MODEL_CPU = "/home/medpal/tracking_person/models/tflite/ssd_mobilenet_v2.tflite"
CORAL_LABELS = "/home/medpal/tracking_person/models/tflite/coco_labels.txt"

CORAL_FACE_DETECTION_MODEL = (
    "/home/medpal/tracking_person/models/tflite/"
    "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
CORAL_FACE_LABELS = (
    "/home/medpal/tracking_person/models/tflite/face_labels.txt")

# ------------------------------------------------------------
# RFID (MFRC522 on ESP32; files on Pi)
# ------------------------------------------------------------
AUTH_UID_FILE = "/home/medpal/tracking_person/authorized_uids.txt"
RFID_PERSONS_FILE = "/home/medpal/tracking_person/rfid_persons.json"
