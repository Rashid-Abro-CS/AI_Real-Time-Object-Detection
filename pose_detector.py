"""Optional body-part detector using YOLO pose, rendered as clean boxes only.

No skeleton lines or landmark dots are drawn. Keypoints are used internally to
estimate boxes for head, torso, arms and legs. Boxes are temporally smoothed and
cached between inference frames to reduce flicker.
"""
import logging
import time

logger = logging.getLogger("pose_detector")

# COCO pose keypoint indices
KEYPOINT_NAMES = [
    "nose", "left eye", "right eye", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle",
]

# The camera preview is horizontally mirrored in main.py. Swap the pose model's
# anatomical side indices so labels correspond to the user's actual left/right.
# Facial keypoints are deliberately excluded: no Head/Eye/Ear/Nose boxes.
REGIONS = {
    "Left arm": (6, 8, 10), "Right arm": (5, 7, 9),
    "Left leg": (12, 14, 16), "Right leg": (11, 13, 15),
    "Torso": (5, 6, 11, 12),
}

class BodyPoseDetector:
    def __init__(self, model_path, device, conf=0.25, keypoint_conf=0.25, tracker="bytetrack.yaml",
                 smoothing=0.65, cache_seconds=0.8):
        self.available = False
        self.device = device
        self.conf = conf
        self.keypoint_conf = keypoint_conf
        self.model = None
        self.smoothing = float(smoothing)
        self.cache_seconds = float(cache_seconds)
        self._smoothed = {}
        self._cached_detections = []
        self._cached_wrist_hints = []
        self._cache_time = 0.0
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.available = True
            logger.info("Body-part box detector loaded: %s", model_path)
        except Exception as exc:
            logger.warning("Body-part detector disabled: %s", exc)

    @staticmethod
    def _box_from_points(points, pad, width, height):
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        return (float(max(0, x1-pad)), float(max(0, y1-pad)),
                float(min(width, x2+pad)), float(min(height, y2+pad)))

    def _smooth_box(self, key, box):
        old = self._smoothed.get(key)
        if old is None:
            value = tuple(float(v) for v in box)
        else:
            a = self.smoothing
            value = tuple(a*old[i] + (1-a)*float(box[i]) for i in range(4))
        self._smoothed[key] = value
        return value

    def get_wrist_hints(self):
        """Cached wrist keypoint coordinates for hand-detector ROI hints."""
        return list(self._cached_wrist_hints)

    def process(self, frame, frame_idx=0, tracker="bytetrack.yaml", run_inference=True):
        if not self.available:
            return []
        now = time.monotonic()
        if not run_inference:
            if now - self._cache_time <= self.cache_seconds:
                # Keep the cache alive until the next scheduled inference frame.
                self._cache_time = now
                return [dict(d) for d in self._cached_detections]
            return []
        try:
            results = self.model.track(frame, persist=True, conf=self.conf, imgsz=640,
                                     tracker=tracker, device=self.device, verbose=False)
        except Exception as exc:
            logger.warning("Pose inference failed: %s", exc)
            return [dict(d) for d in self._cached_detections] if now-self._cache_time <= self.cache_seconds else []
        if not results:
            return []
        result = results[0]
        kp = getattr(result, "keypoints", None)
        if kp is None or kp.xy is None:
            self._cached_detections = []
            self._cached_wrist_hints = []
            self._cache_time = now
            return []
        xy = kp.xy.cpu().numpy()
        confs = kp.conf.cpu().numpy() if kp.conf is not None else None
        ids = None
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy()
        height, width = frame.shape[:2]
        detections = []
        wrist_hints = []
        for person_i, points in enumerate(xy):
            person_id = int(ids[person_i]) if ids is not None else person_i + 1000
            point_conf = confs[person_i] if confs is not None else [1.0] * len(points)
            for wrist_idx in (9, 10):
                if wrist_idx < len(points) and point_conf[wrist_idx] >= max(0.12, self.keypoint_conf * 0.6):
                    wrist_hints.append((float(points[wrist_idx][0]), float(points[wrist_idx][1])))
            for region_i, (region_name, joint_ids) in enumerate(REGIONS.items()):
                if any(j >= len(points) or point_conf[j] < self.keypoint_conf for j in joint_ids):
                    continue
                region_points = points[list(joint_ids)]
                pad = 12 if region_name == "Torso" else 9
                region_box = self._box_from_points(region_points, pad, width, height)
                region_conf = float(sum(point_conf[j] for j in joint_ids)/len(joint_ids))
                detections.append({"track_id": 2_500_000 + person_id*100 + region_i,
                    "class_name": region_name,
                    "confidence": region_conf,
                    "box": self._smooth_box((person_id, region_name), region_box)})
        self._cached_detections = [dict(d) for d in detections]
        self._cached_wrist_hints = wrist_hints
        self._cache_time = now
        return detections
