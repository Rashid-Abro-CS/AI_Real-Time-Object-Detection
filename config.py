"""
Central configuration for the AI Object Detection, Tracking & Scene
Understanding System. Edit the values below to match your environment
before running main.py.
"""

import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))  # Project-local environment file
# --------------------------------------------------------------------------
# General
# --------------------------------------------------------------------------
PROJECT_NAME = "AI Real-Time Object Detection, Tracking & Scene Understanding System"

# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1250
FRAME_HEIGHT = 700

# --------------------------------------------------------------------------
# YOLO object detection (Open Images primary, COCO supplemental)
# --------------------------------------------------------------------------

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-oiv7.pt")
YOLO_CONFIDENCE_THRESHOLD = 0.08  # raw floor; session confirmation is stricter
YOLO_IOU_THRESHOLD = 0.45
YOLO_IMAGE_SIZE = 800  # higher resolution helps small objects; reduce to 640 for slower CPUs
YOLO_TRACKER_CONFIG = os.path.join(BASE_DIR, "bytetrack_custom.yaml")
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "auto")  # auto, cpu, cuda, cuda:0, mps

# Supplemental COCO detector increases coverage of common daily objects.
# It auto-downloads yolov8n.pt via Ultralytics when enabled and absent.
ENABLE_SUPPLEMENTAL_MODEL = os.getenv("ENABLE_SUPPLEMENTAL_MODEL", "1").lower() not in ("0", "false", "no")
SUPPLEMENTAL_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
SUPPLEMENTAL_EVERY_N_FRAMES = 2
SUPPLEMENTAL_IMAGE_SIZE = 640
SUPPLEMENTAL_CONFIDENCE = 0.12

# Multi-scale refinement: tiled crops improve small/distant-object recall.
# This adds periodic inference work; disable on slow CPUs if FPS is too low.
ENABLE_TILED_DETECTION = os.getenv("ENABLE_TILED_DETECTION", "1").lower() not in ("0", "false", "no")
TILED_EVERY_N_FRAMES = 6
TILED_IMAGE_SIZE = 640
TILED_CONFIDENCE = 0.12

# Eyewear-specific high-resolution head crops. Open Images contains glasses
# classes, but COCO does not; the supplemental model alone cannot detect them.
ENABLE_EYEWEAR_CROP_DETECTION = os.getenv("ENABLE_EYEWEAR_CROP_DETECTION", "1").lower() not in ("0", "false", "no")
EYEWEAR_CROP_EVERY_N_FRAMES = 4
EYEWEAR_CROP_IMAGE_SIZE = 800
EYEWEAR_CROP_CONFIDENCE = 0.10

# Human body keypoints (pose) and hand landmarks are specialized features.
# Pose model auto-downloads on first use; MediaPipe hands is optional.
ENABLE_BODY_POSE = os.getenv("ENABLE_BODY_POSE", "1").lower() not in ("0", "false", "no")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-pose.pt")
POSE_EVERY_N_FRAMES = 3
POSE_CONFIDENCE = 0.25
POSE_KEYPOINT_CONFIDENCE = 0.25
ENABLE_MEDIAPIPE_HANDS = os.getenv("ENABLE_MEDIAPIPE_HANDS", "1").lower() not in ("0", "false", "no")

# Detection display/debugging and performance
RAW_DETECTION_DEBUG_DEFAULT = False
LIVE_STATE_UPDATE_SECONDS = 1.0
SHOW_CAMERA_WINDOW = True
TARGET_CLASSES = None  # None means allow every class supported by the loaded model

# --------------------------------------------------------------------------
# Object Tracking / Session Logic
# --------------------------------------------------------------------------
# A session is closed (and saved as ONE MongoDB record) only when its track
# has been missing for at least MAX_MISSED_FRAMES frames AND at least
# SESSION_GRACE_SECONDS seconds. Frame count alone depended on camera FPS and
# was shorter (20) than ByteTrack's own lost-track buffer, so a session could
# be closed and a duplicate one opened for the very same tracking id.
MAX_MISSED_FRAMES = 60                # legacy safety limit
SESSION_MIN_MISSED_FRAMES = 3
SESSION_GRACE_SECONDS = 2.0
MIN_SESSION_DURATION_SECONDS = 1.0    # ignore extremely short false-positive blips

# --- Confirmation: a detection becomes a "real" object only after temporal evidence
# Defaults apply to every class unless overridden in CLASS_RULES.
DETECTION_MIN_CONFIDENCE = 0.35       # generic class threshold; body/small-object overrides are lower
KEEP_CONFIDENCE_RATIO = 0.6           # confirmed sessions stay alive on frames >= ratio * min confidence
CONFIRM_MIN_HITS = 4                  # evidence frames needed
CONFIRM_MIN_SECONDS = 0.40             # first -> last evidence frame
CONFIRM_MIN_HIT_RATIO = 0.40           # evidence frames / frames since first evidence (rejects flicker)

# Per-class overrides (lower-case Open Images class names).
#  - "mobile phone" has a LOWER threshold so small/distant phones are not lost.
#  - eyewear is the noisiest Open Images class on a nano model, so it needs
#    stronger, steadier evidence. Eyewear that is not on a detected face
#    ("standalone", e.g. lying in front of the camera) needs even more.
# These are starting values: tune them with diagnose.py on your own camera.
_EYEWEAR_RULES = {
    "min_confidence": 0.32, "min_hits": 6, "min_seconds": 0.75,
    "min_hit_ratio": 0.45, "min_avg_confidence": 0.38,
    "standalone_min_hits": 15, "standalone_min_seconds": 2.0,
    "standalone_min_avg_confidence": 0.45,
}
# Specialized thresholds for small objects and body-part landmarks.
_BODY_PART_RULES = {
    "min_confidence": 0.20, "min_hits": 3, "min_seconds": 0.35,
    "min_hit_ratio": 0.25, "min_avg_confidence": 0.20,
}
_SMALL_OBJECT_RULES = {
    "min_confidence": 0.25, "min_hits": 3, "min_seconds": 0.30,
    "min_hit_ratio": 0.30, "min_avg_confidence": 0.25,
}
CLASS_RULES = {
    "mobile phone": dict(_SMALL_OBJECT_RULES),
    "cell phone": dict(_SMALL_OBJECT_RULES),
    "phone": dict(_SMALL_OBJECT_RULES),
    "hand": dict(_BODY_PART_RULES),
    "left hand": dict(_BODY_PART_RULES),
    "right hand": dict(_BODY_PART_RULES),
    "human hand": dict(_BODY_PART_RULES),
    "left wrist": dict(_BODY_PART_RULES),
    "right wrist": dict(_BODY_PART_RULES),
    "left elbow": dict(_BODY_PART_RULES),
    "right elbow": dict(_BODY_PART_RULES),
    "left shoulder": dict(_BODY_PART_RULES),
    "right shoulder": dict(_BODY_PART_RULES),
    "left knee": dict(_BODY_PART_RULES),
    "right knee": dict(_BODY_PART_RULES),
    "left ankle": dict(_BODY_PART_RULES),
    "right ankle": dict(_BODY_PART_RULES),
    "nose": dict(_BODY_PART_RULES),
    "left eye": dict(_BODY_PART_RULES),
    "right eye": dict(_BODY_PART_RULES),
    "left ear": dict(_BODY_PART_RULES),
    "right ear": dict(_BODY_PART_RULES),
    "glasses": dict(_EYEWEAR_RULES),
    "sunglasses": dict(_EYEWEAR_RULES),
    "eyeglasses": dict(_EYEWEAR_RULES),
    "spectacles": dict(_EYEWEAR_RULES),
}

# --- Re-association of a lost track that comes back under a NEW tracker id.
# Conservative on purpose: same class, confirmed session lost <= REID_MAX_GAP_SECONDS
# ago, new box overlaps its last box (IoU >= REID_MIN_IOU), and exactly one
# candidate matches. Anything ambiguous is NOT merged (two people never merge).
REID_MAX_GAP_SECONDS = 2.0
REID_MIN_IOU = 0.30

# --- Duplicate boxes for the same person. Open Images has Person/Man/Woman/Boy/Girl,
# so one body can be reported as two classes with near-identical boxes.
PERSON_DUPLICATE_IOU = 0.70

# --- Eyewear plausibility relative to a detected face/head box
EYEWEAR_FACE_MIN_COVERAGE = 0.50      # share of the glasses box inside the face box
EYEWEAR_STANDALONE_MAX_COVERAGE = 0.15  # below this the glasses are "not on a face"
EYEWEAR_CENTER_Y_RANGE = (0.12, 0.65)   # glasses centre, as fraction of face height from its top
EYEWEAR_WIDTH_RATIO_RANGE = (0.45, 1.5) # glasses width / face width
EYEWEAR_MIN_ASPECT = 1.0                # glasses width / height

# --------------------------------------------------------------------------
# Object Co-occurrence / Relationships
# --------------------------------------------------------------------------
MIN_OBJECTS_FOR_RELATIONSHIP = 2
MAX_MISSED_FRAMES_RELATIONSHIP = 20  # legacy safety
RELATIONSHIP_GRACE_SECONDS = 2.0
RELATIONSHIP_MIN_MISSED_FRAMES = 3
MIN_RELATIONSHIP_DURATION_SECONDS = 2.0

# --------------------------------------------------------------------------
# Scene Classification (Places365 -- 365 indoor/outdoor scene categories)
# --------------------------------------------------------------------------
SCENE_MODEL_ARCH = "resnet18"
SCENE_MODEL_WEIGHTS = os.path.join(BASE_DIR, "models", "scene_classification_model", "resnet18_places365.pth.tar")
SCENE_CATEGORIES_FILE = os.path.join(BASE_DIR, "models", "scene_classification_model", "categories_places365.txt")
SCENE_CLASSIFY_EVERY_N_FRAMES = 15    # legacy frame interval
SCENE_CLASSIFY_INTERVAL_SECONDS = 2.0  # wall-clock interval
SCENE_MIN_CONFIDENCE = 0.15
SCENE_MIN_DURATION_SECONDS = 2.0
SCENE_MAX_GAP_SECONDS = 3.0

# Maps raw Places365 labels to the simplified categories requested in the spec.
# Any Places365 label not listed here falls back to a readable title-cased version
# of the raw label (see scene_classifier.SceneClassifier._fallback_label).
SCENE_CATEGORY_MAP = {
    "living_room": "Living Room",
    "bedroom": "Bedroom",
    "kitchen": "Kitchen",
    "office": "Office",
    "office_cubicles": "Office",
    "computer_room": "Office",
    "home_office": "Office",
    "conference_room": "Office",
    "classroom": "Classroom",
    "shop": "Store",
    "clothing_store": "Store",
    "general_store_indoor": "Store",
    "supermarket": "Store",
    "market_indoor": "Store",
    "street": "Street",
    "highway": "Road",
    "crosswalk": "Road",
    "parking_lot": "Parking Area",
    "park": "Park",
    "courtyard": "Park",
    "train_station_platform": "Station",
    "subway_station_platform": "Station",
    "bus_station_indoor": "Station",
    "corridor": "Room",
    "dining_room": "Room",
}
DEFAULT_SCENE_LABEL = "Room"

# --------------------------------------------------------------------------
# MongoDB
# --------------------------------------------------------------------------
# MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "object_detection_db")

COLLECTION_DETECTIONS = "detections"
COLLECTION_SCENES = "scene_sessions"
COLLECTION_RELATIONSHIPS = "object_relationships"
COLLECTION_LIVE_STATE = "live_state"   # single-document collection the dashboard polls

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
WINDOW_NAME = "REAL-TIME AI CAMERA"
