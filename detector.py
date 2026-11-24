"""YOLO object detection + ByteTrack tracking with optional multi-scale refinement.

The primary Open Images model remains the source of normal tracked detections.
A periodic tiled pass improves recall for small objects, and a focused high-res
head crop pass improves eyewear detection when a person box is available. Crop
results are temporally associated with lightweight IoU tracking and deduplicated
against primary/supplemental detections.
"""
import logging
import math
import os
import time

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

import config

logger = logging.getLogger("detector")


def resolve_device(requested="auto"):
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_class_name(names, cls_id):
    if isinstance(names, dict):
        return names.get(int(cls_id), str(int(cls_id)))
    try:
        return names[int(cls_id)]
    except (IndexError, KeyError, TypeError):
        return str(int(cls_id))


def canonical_class(name):
    n = name.lower().strip()
    synonyms = {
        "cell phone": "mobile phone", "phone": "mobile phone",
        "smartphone": "mobile phone", "mobile phone": "mobile phone",
        "person": "person", "man": "person", "woman": "person",
        "boy": "person", "girl": "person", "human": "person",
        "tv": "television", "television": "television",
        "couch": "sofa", "sofa": "sofa", "dining table": "table",
        "table": "table", "remote": "remote control",
        "remote control": "remote control", "cup": "cup", "mug": "cup",
        "potted plant": "plant", "plant": "plant",
        "eyeglasses": "glasses", "spectacles": "glasses",
    }
    return synonyms.get(n, n)


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def merge_supplemental(primary, supplemental, iou_threshold=0.60):
    merged = list(primary)
    for extra in supplemental:
        duplicate = any(
            canonical_class(extra["class_name"]) == canonical_class(base["class_name"])
            and _box_iou(extra["box"], base["box"]) >= iou_threshold
            for base in primary
        )
        if not duplicate:
            merged.append(extra)
    return merged


class ObjectDetector:
    def __init__(self, model_path=config.YOLO_MODEL_PATH, device=config.YOLO_DEVICE):
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. Run setup.bat or pip install -r requirements.txt.")
        self.device = resolve_device(device)
        self.model_path = model_path
        logger.info("Loading primary YOLO model: %s on %s", model_path, self.device)
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.last_raw_detections = []
        self.last_tracked_detections = []
        self.supplemental = None
        self.supplemental_error = None
        self.refinement_error = None
        self._refinement_tracks = {}
        self._next_refinement_id = 1_500_000
        self._last_refinement_time = 0.0
        self._frame_idx = 0
        self.eyewear_class_ids = self._find_class_ids({"glasses", "sunglasses", "eyeglasses", "spectacles"})
        if config.ENABLE_SUPPLEMENTAL_MODEL:
            self._load_supplemental()
        logger.info("Eyewear classes in primary model: %s", self.eyewear_class_ids)

    def _find_class_ids(self, target_names):
        names = self.class_names.items() if isinstance(self.class_names, dict) else enumerate(self.class_names)
        return [int(i) for i, n in names if str(n).strip().lower() in target_names]

    def _load_supplemental(self):
        try:
            logger.info("Loading supplemental COCO detector: %s", config.SUPPLEMENTAL_MODEL_PATH)
            self.supplemental = YOLO(config.SUPPLEMENTAL_MODEL_PATH)
            self.supplemental_names = self.supplemental.names
            logger.info("Supplemental detector ready")
        except Exception as exc:
            self.supplemental = None
            self.supplemental_error = str(exc)
            logger.warning("Supplemental detector unavailable: %s", exc)

    def _extract(self, result, names, id_offset=0, target_classes=None):
        raw, tracked = [], []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return raw, tracked
        try:
            cls_values = boxes.cls.cpu().numpy()
            conf_values = boxes.conf.cpu().numpy()
            xyxy_values = boxes.xyxy.cpu().numpy()
            ids_obj = boxes.id
            ids_values = ids_obj.cpu().numpy() if ids_obj is not None else None
        except Exception:
            return raw, tracked
        for idx, (cls_id, conf, box) in enumerate(zip(cls_values, conf_values, xyxy_values)):
            cls_name = get_class_name(names, cls_id)
            if target_classes and cls_name.lower() not in target_classes:
                continue
            track_id = int(ids_values[idx]) + id_offset if ids_values is not None else None
            rec = {"track_id": track_id, "class_name": cls_name,
                   "confidence": float(conf), "box": tuple(float(v) for v in box)}
            raw.append(rec)
            if track_id is not None:
                tracked.append(rec)
        return raw, tracked

    def _extract_crop_result(self, result, names, offset=(0, 0), scale=1.0, target_classes=None):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        try:
            cls_values = boxes.cls.cpu().numpy()
            conf_values = boxes.conf.cpu().numpy()
            xyxy_values = boxes.xyxy.cpu().numpy()
        except Exception:
            return []
        ox, oy = offset
        out = []
        for cls_id, conf, box in zip(cls_values, conf_values, xyxy_values):
            name = get_class_name(names, cls_id)
            if target_classes and name.lower() not in target_classes:
                continue
            mapped = (ox + float(box[0]) / scale, oy + float(box[1]) / scale,
                      ox + float(box[2]) / scale, oy + float(box[3]) / scale)
            out.append({"track_id": None, "class_name": name, "confidence": float(conf), "box": mapped})
        return out

    @staticmethod
    def _tiles(frame, overlap=0.18):
        h, w = frame.shape[:2]
        # Two-by-two overlapping tiles. Small images are not tiled.
        if w < 500 or h < 400:
            return [(0, 0, w, h)]
        tile_w, tile_h = int(w * 0.58), int(h * 0.58)
        xs = [0, max(0, w-tile_w)]
        ys = [0, max(0, h-tile_h)]
        return [(x, y, min(w, x+tile_w), min(h, y+tile_h)) for y in ys for x in xs]

    def _head_crops(self, frame, person_detections, max_people=2):
        h, w = frame.shape[:2]
        persons = [d for d in person_detections if canonical_class(d["class_name"]) == "person"]
        persons.sort(key=lambda d: d["confidence"], reverse=True)
        crops = []
        for person in persons[:max_people]:
            x1, y1, x2, y2 = person["box"]
            pw, ph = x2-x1, y2-y1
            if pw < 50 or ph < 100:
                continue
            # Face/eyewear search region: upper 32% of person with lateral padding.
            cx1 = max(0, int(x1-pw*0.12)); cx2 = min(w, int(x2+pw*0.12))
            cy1 = max(0, int(y1-ph*0.03)); cy2 = min(h, int(y1+ph*0.34))
            if cx2-cx1 >= 40 and cy2-cy1 >= 40:
                crops.append((cx1, cy1, cx2, cy2))
        return crops

    def _assign_refinement_id(self, det, now, frame_idx):
        cls = canonical_class(det["class_name"])
        box = det["box"]
        candidates = []
        for key, prev in list(self._refinement_tracks.items()):
            if now-prev["time"] > 1.8:
                self._refinement_tracks.pop(key, None)
                continue
            if canonical_class(prev["class_name"]) != cls or prev.get("used_frame") == frame_idx:
                continue
            overlap = _box_iou(box, prev["box"])
            pc = ((prev["box"][0]+prev["box"][2])/2, (prev["box"][1]+prev["box"][3])/2)
            dc = ((box[0]+box[2])/2, (box[1]+box[3])/2)
            dist = math.hypot(pc[0]-dc[0], pc[1]-dc[1])
            if overlap >= 0.12 or dist <= max(60, 0.16*max(box[2]-box[0], box[3]-box[1])):
                candidates.append((1-overlap + dist/1000, key, prev))
        if candidates:
            _, key, prev = min(candidates, key=lambda x: x[0])
            prev.update(box=box, time=now, class_name=det["class_name"], used_frame=frame_idx)
            return key
        self._next_refinement_id += 1
        key = self._next_refinement_id
        self._refinement_tracks[key] = {"class_name": det["class_name"], "box": box, "time": now, "used_frame": frame_idx}
        return key

    def _refine_small_objects(self, frame, primary_tracked, frame_idx):
        """Periodic tiled inference for objects that are small in the full frame."""
        if not getattr(config, "ENABLE_TILED_DETECTION", True):
            return []
        every = max(1, int(getattr(config, "TILED_EVERY_N_FRAMES", 6)))
        if frame_idx % every:
            return []
        if not hasattr(self.model, "predict"):
            return []
        import cv2
        h, w = frame.shape[:2]
        tile_candidates = []
        try:
            for x1, y1, x2, y2 in self._tiles(frame):
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                results = self.model.predict(crop, conf=config.TILED_CONFIDENCE,
                    imgsz=config.TILED_IMAGE_SIZE, device=self.device, verbose=False)
                if results:
                    tile_candidates.extend(self._extract_crop_result(results[0], self.class_names,
                        offset=(x1, y1), target_classes=config.TARGET_CLASSES))
            self.refinement_error = None
        except Exception as exc:
            self.refinement_error = f"Tiled inference: {type(exc).__name__}: {exc}"
            logger.warning(self.refinement_error)
            return []
        return self._track_and_dedupe_refinements(tile_candidates, primary_tracked, time.monotonic(), frame_idx)

    def _refine_eyewear(self, frame, primary_raw, primary_tracked, frame_idx):
        """High-resolution eyewear-only inference in person upper-head crops."""
        if not getattr(config, "ENABLE_EYEWEAR_CROP_DETECTION", True) or not self.eyewear_class_ids:
            return []
        every = max(1, int(getattr(config, "EYEWEAR_CROP_EVERY_N_FRAMES", 4)))
        if frame_idx % every or not hasattr(self.model, "predict"):
            return []
        import cv2
        candidates = []
        target_names = {"glasses", "sunglasses", "eyeglasses", "spectacles"}
        try:
            for x1, y1, x2, y2 in self._head_crops(frame, primary_raw):
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                scale = min(2.0, max(1.0, 700.0/max(crop.shape[:2])))
                up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                results = self.model.predict(up, conf=config.EYEWEAR_CROP_CONFIDENCE,
                    imgsz=config.EYEWEAR_CROP_IMAGE_SIZE, device=self.device,
                    classes=self.eyewear_class_ids, verbose=False)
                if results:
                    candidates.extend(self._extract_crop_result(results[0], self.class_names,
                        offset=(x1, y1), scale=scale, target_classes=target_names))
        except Exception as exc:
            self.refinement_error = f"Eyewear crop inference: {type(exc).__name__}: {exc}"
            logger.warning(self.refinement_error)
            return []
        # Person-head crop detections are contextually on-face; avoid strict
        # face-box geometry filters that previously rejected valid glasses.
        for c in candidates:
            c["context_hint"] = "crop_on_face"
        return self._track_and_dedupe_refinements(candidates, primary_tracked, time.monotonic(), frame_idx)

    def _track_and_dedupe_refinements(self, candidates, primary_tracked, now, frame_idx):
        # Remove duplicates within the crop passes first.
        clean = []
        candidates.sort(key=lambda d: d["confidence"], reverse=True)
        for det in candidates:
            if any(canonical_class(det["class_name"]) == canonical_class(other["class_name"])
                   and _box_iou(det["box"], other["box"]) >= 0.45 for other in clean):
                continue
            if any(canonical_class(det["class_name"]) == canonical_class(base["class_name"])
                   and _box_iou(det["box"], base["box"]) >= 0.40 for base in primary_tracked):
                continue
            det["track_id"] = self._assign_refinement_id(det, now, frame_idx)
            clean.append(det)
        return clean

    def track(self, frame, frame_idx=0):
        self._frame_idx = frame_idx
        primary_results = self.model.track(frame, persist=True, conf=config.YOLO_CONFIDENCE_THRESHOLD,
            iou=config.YOLO_IOU_THRESHOLD, imgsz=config.YOLO_IMAGE_SIZE,
            tracker=config.YOLO_TRACKER_CONFIG, device=self.device, verbose=False)
        primary_raw, primary_tracked = [], []
        if primary_results:
            primary_raw, primary_tracked = self._extract(primary_results[0], self.class_names,
                target_classes=config.TARGET_CLASSES)
        self.last_raw_detections = list(primary_raw)
        merged_tracked = list(primary_tracked)

        if self.supplemental is not None and frame_idx % config.SUPPLEMENTAL_EVERY_N_FRAMES == 0:
            try:
                results = self.supplemental.track(frame, persist=True, conf=config.SUPPLEMENTAL_CONFIDENCE,
                    iou=config.YOLO_IOU_THRESHOLD, imgsz=config.SUPPLEMENTAL_IMAGE_SIZE,
                    tracker=config.YOLO_TRACKER_CONFIG, device=self.device, verbose=False)
                if results:
                    raw2, tracked2 = self._extract(results[0], self.supplemental_names,
                        id_offset=1_000_000, target_classes=config.TARGET_CLASSES)
                    self.last_raw_detections.extend(raw2)
                    merged_tracked = merge_supplemental(merged_tracked, tracked2)
            except Exception as exc:
                logger.warning("Supplemental detection failed this frame: %s", exc)

        # Periodic multi-scale tiles improve recall for small and distant objects.
        tiled = self._refine_small_objects(frame, merged_tracked, frame_idx)
        merged_tracked = merge_supplemental(merged_tracked, tiled, iou_threshold=0.45)
        # Crop-based eyewear pass specifically targets a common missed small object.
        eyewear = self._refine_eyewear(frame, primary_raw, merged_tracked, frame_idx)
        merged_tracked = merge_supplemental(merged_tracked, eyewear, iou_threshold=0.40)
        self.last_raw_detections.extend(tiled)
        self.last_raw_detections.extend(eyewear)
        self.last_tracked_detections = merged_tracked
        return merged_tracked

    @property
    def available_classes(self):
        names = self.class_names
        return list(names.values()) if isinstance(names, dict) else list(names)
