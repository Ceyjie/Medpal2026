# Motor pin configuration (BCM)
LEFT_RPWM = 13
LEFT_LPWM = 12
LEFT_REN  = 5
LEFT_LEN  = 6
RIGHT_RPWM = 18
RIGHT_LPWM = 19
RIGHT_REN  = 20
RIGHT_LEN  = 21

# Identity lock release: if a locked track hasn't been matched by the
# tracker for this many seconds, clear the lock and reset the name to
# "Unknown". A fresh scan then runs when the person reappears.
# Brief occlusions (walking behind a table) do not trigger this.
IDENTITY_LOCK_RELEASE_S = 3.0
SKIP_SCANS_WHEN_LOCKED = True
# Assumed tracker frame rate. Used to convert seconds to frames.
# Matches the frame_interval of 0.12 s in the main loop.
ASSUMED_FPS = 8.0

FACE_SCAN_COOLDOWN_S = 2.0
# Identity locking: a scan with this confidence or higher locks the
# track, and locked tracks are never re-scanned until the track dies.
IDENTITY_LOCK_MIN_CONF = 0.80


# Add to config.py
CAMERA_HEIGHT_MM = 200      # measure: how high the camera sits above the floor
CAMERA_TILT_DEG = 15        # measure: how much the camera points down
CAMERA_HFOV_DEG = 60        # Astra Pro RGB horizontal FOV
CAMERA_VFOV_DEG = 49.5      # Astra Pro RGB vertical FOV
OCCUPANCY_GRID_SIZE = 60    # cells per side
OCCUPANCY_GRID_RES_MM = 50  # mm per cell (5 cm)
PLANNER_DISABLED = False      # set True to default planner off

ROBOT_RADIUS_MM = 200
# half the widest dimension of the chassis

# Speed config (percentages)
FOLLOW_BASE_SPEED = 60
REVERSE_SPEED = 30
SPEED_INCREASE = 10
MAX_SPEED = 100



# Depth sensor floor
MIN_VALID_DEPTH_MM = 55

YOLO_FACE_MODEL = "/home/medpal/2026medpal/models/yolov8n-face_full_integer_quant_edgetpu.tflite"
YOLO_FACE_CONF_THRES = 0.4
YOLO_FACE_IOU_THRES = 0.5

# Distance thresholds (mm)
REVERSE_DISTANCE_MM = 60
FORWARD_DISTANCE_MM = 100
STOP_ZONE_MIN_MM = 60
STOP_ZONE_MAX_MM = 100
DISTANCE_THRESHOLD_MM = 100
MIN_DISTANCE_MM = 85

STRICT_THRES = 0.60
# Try 0.55 for variable lighting, or 0.60 for stricter matching

# Obstacle avoidance 
OBSTACLE_STOP_MM = 150
OBSTACLE_SLOW_MM = 350
OBSTACLE_SIDE_BIAS = 25

# Frame settings
FRAME_W, FRAME_H = 640, 480
CONF_THRES = 0.35
COSINE_THRES = 0.5

# Re-identification stability. 
REID_LOCK_FRAMES = 3
REID_LOSS_FRAMES = 7

# Person (body) recognition
PERSON_TRAINING_DIR = "/home/medpal/tracking_person/person_training"
PERSON_CLASSIFIER_PATH = "/home/medpal/tracking_person/models/person_classifier.pkl"
PERSON_SVM_CONFIDENCE_THRES = 0.60
PERSON_IDENTITY_TIMEOUT_FRAMES = 20   # ~2.5s at 8 FPS

# Model paths
YOLO_PATH = "/home/medpal/Documents/test/cla/MedPalRobot/yolov8n.onnx"
REID_PATH = "/home/medpal/tracking_person/models/reid.onnx"
TARGET_PATH = "/home/medpal/tracking_person/target_embedding.pkl"
TARGETS_DIR = "/home/medpal/tracking_person/targets"

# Coral TPU paths
CORAL_DETECTION_MODEL = "/home/medpal/tracking_person/models/tflite/ssd_mobilenet_v2_edgetpu.tflite"
CORAL_DETECTION_MODEL_CPU = "/home/medpal/tracking_person/models/tflite/ssd_mobilenet_v2.tflite"
CORAL_LABELS = "/home/medpal/tracking_person/models/tflite/coco_labels.txt"

# Touch sensor + servo
SERVO_PIN = 24
TOUCH_PIN = 23
SERVO_OPEN_PULSE = 2000
SERVO_CLOSE_PULSE = 1000
SERVO_NEUTRAL_PULSE = 1500
SERVO_MOVE_TIME = 1.3
LONG_PRESS_S = 1.5


# NEW: body detector needs a stricter threshold
BODY_CONF_THRES = 0.60

# NEW: ignore small body boxes (false positives are usually tiny)
MIN_BODY_BOX_W = 60
MIN_BODY_BOX_H = 120
MIN_BODY_BOX_AREA = 10000      # pixels


# only run face embedding/recognition when the face is closer than this
FACE_RECOGNIZE_MAX_MM = 1200


# Coral face detection
CORAL_FACE_DETECTION_MODEL = "/home/medpal/tracking_person/models/tflite/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
CORAL_FACE_LABELS = "/home/medpal/tracking_person/models/tflite/face_labels.txt"

# MobileFaceNet embedding
MOBILEFACENET_MODEL = "/home/medpal/tracking_person/models/tflite/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite"
FACE_EMBEDDING_SIZE = 1024
FACE_INPUT_SIZE = 224

# SVM classifier
SVM_MODEL_PATH = "/home/medpal/tracking_person/models/svm_face_classifier.pkl"
FACE_TRAINING_DIR = "/home/medpal/tracking_person/face_training"

FACE_SVM_CONFIDENCE_THRES = 0.75
# Lower = more permissive, Higher = stricter


# Enrollment quality
ENROLL_MIN_FACE_PX = 50         # minimum face width/height in pixels
ENROLL_BLUR_THRES = 40.0        # Laplacian variance; below = too blurry
ENROLL_DIVERSITY_PX = 25        # skip frames too similar to the last capture
ENROLL_DEFAULT_COUNT = 30       # default number of crops per person


AUTO_SAVE_RECOGNIZED_CROPS = True

REID_PATH = "/home/medpal/tracking_person/models/reid.onnx"

# SCRFD + ArcFace
SCRFD_MODEL_PATH = "/home/medpal/2026medpal/models/scrfd_arcface/det_2.5g.onnx"
ARCFACE_MODEL_PATH = "/home/medpal/2026medpal/models/scrfd_arcface/w600k_mbf.onnx"
SCRFD_INPUT_SIZE = (640, 640)
SCRFD_CONF_THRES = 0.5
SCRFD_IOU_THRES = 0.4
ARCFACE_EMBEDDING_SIZE = 512
SVM_MODEL_PATH = "/home/medpal/tracking_person/models/arcface_classifier.pkl"

# RFID (MFRC522 via SPI)
RFID_RST_PIN = 22
AUTH_UID_FILE = "/home/medpal/tracking_person/authorized_uids.txt"
RFID_PERSONS_FILE = "/home/medpal/tracking_person/rfid_persons.json"
