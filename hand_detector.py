"""Dedicated MediaPipe hand detection with full-frame + wrist-region passes.

The pose model supplies wrist locations only as search hints. Hands are detected
independently by MediaPipe and never inferred from wrist keypoints alone.
"""
import logging
import math
import time

logger = logging.getLogger("hand_detector")


class HandLandmarkDetector:
    def __init__(self, enabled=True, detection_confidence=0.25, tracking_confidence=0.25):
        self.available = False
        self.error = None
        self.inference_error = None
        self.last_count = 0
        self.hands = None
        self.mp = None
        self._next_id = 3_000_000
        self._previous = {}
        self._last_time = 0.0
        self.detection_confidence = float(detection_confidence)
        self.tracking_confidence = float(tracking_confidence)
        if not enabled:
            self.error = "Hand detection disabled by configuration"
            return
        try:
            import mediapipe as mp
            self.mp = mp
            # Legacy Solutions API is available in MediaPipe 0.10.x. Keep the
            # detector permissive; wrist-region passes help with small hands.
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2, model_complexity=0,
                min_detection_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence,
            )
            self.available = True
            logger.info("MediaPipe hand landmarks enabled (detection=%.2f tracking=%.2f)",
                        self.detection_confidence, self.tracking_confidence)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("MediaPipe hands unavailable")

    @staticmethod
    def correct_handedness_label(label):
        """Return label for the mirrored webcam preview convention."""
        return "Right" if str(label).lower() == "left" else "Left"

    def _assign_id(self, side, center):
        now = time.time()
        for stale_key in [k for k, v in self._previous.items() if now-v["time"] > 2.0]:
            self._previous.pop(stale_key, None)
        candidates = []
        for track_key, previous in self._previous.items():
            age = now - previous["time"]
            if age > 1.5:
                continue
            px, py = previous["center"]
            distance = math.hypot(center[0]-px, center[1]-py)
            if distance <= 240:
                side_penalty = 0 if previous.get("side") == side else 25
                candidates.append((distance + side_penalty, track_key, previous))
        if candidates:
            _, track_key, previous = min(candidates, key=lambda item: item[0])
            if previous.get("used_frame") != self._frame_marker:
                previous.update(center=center, time=now, side=side, used_frame=self._frame_marker)
                return previous["id"]
        self._next_id += 1
        track_key = self._next_id
        self._previous[track_key] = {"id": self._next_id, "center": center, "time": now,
                                    "side": side, "used_frame": self._frame_marker}
        return self._next_id

    def _infer_on_image(self, image, offset=(0, 0), scale=1.0):
        """Run MP on image; return normalized landmarks transformed to source frame."""
        import cv2
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        found = []
        if not result.multi_hand_landmarks:
            return found
        for i, landmarks in enumerate(result.multi_hand_landmarks):
            side = "Hand"
            confidence = 0.55
            if result.multi_handedness and i < len(result.multi_handedness):
                classification = result.multi_handedness[i].classification[0]
                side = f"{self.correct_handedness_label(classification.label)} hand"
                confidence = float(classification.score)
            pts = []
            iw, ih = image.shape[1], image.shape[0]
            ox, oy = offset
            for lm in landmarks.landmark:
                # ROI detections are mapped back into original frame coordinates.
                pts.append((ox + lm.x * iw / scale, oy + lm.y * ih / scale))
            found.append({"side": side, "confidence": confidence, "points": pts})
        return found

    def process(self, frame, wrist_hints=None):
        if not self.available:
            return []
        import cv2
        h, w = frame.shape[:2]
        self._frame_marker = getattr(self, "_frame_marker", 0) + 1
        raw = []
        try:
            # Primary full-frame pass. This catches hands even if pose misses wrists.
            raw.extend(self._infer_on_image(frame))
            # Targeted passes around pose wrist keypoints improve small/distant hand
            # recall. Crop is upscaled before inference and results mapped back.
            for hint in (wrist_hints or [])[:4]:
                try:
                    cx, cy = float(hint[0]), float(hint[1])
                    radius = max(120, int(min(w, h) * 0.20))
                    x1, y1 = max(0, int(cx-radius)), max(0, int(cy-radius))
                    x2, y2 = min(w, int(cx+radius)), min(h, int(cy+radius))
                    if x2-x1 < 80 or y2-y1 < 80:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    # Upscale ROI (hands are often tiny in a 720p/1080p frame).
                    factor = 1.5
                    resized = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
                    raw.extend(self._infer_on_image(resized, offset=(x1, y1), scale=factor))
                except Exception as roi_exc:
                    logger.debug("Wrist ROI hand pass skipped: %s", roi_exc)
            self.inference_error = None
        except Exception as exc:
            self.inference_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Hand inference failed")
            return []

        # Merge duplicate detections from full frame and overlapping wrist crops.
        merged = []
        for item in raw:
            xs = [p[0] for p in item["points"]]
            ys = [p[1] for p in item["points"]]
            box = (max(0.0, min(xs)-10), max(0.0, min(ys)-10),
                   min(float(w), max(xs)+10), min(float(h), max(ys)+10))
            center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
            duplicate = None
            for existing in merged:
                ex = existing["center"]
                if math.hypot(center[0]-ex[0], center[1]-ex[1]) < 80:
                    duplicate = existing
                    break
            if duplicate is None:
                merged.append({**item, "box": box, "center": center})
            elif item["confidence"] > duplicate["confidence"]:
                duplicate.update({**item, "box": box, "center": center})

        self.last_count = len(merged)
        detections = []
        for item in merged[:2]:
            side, confidence, points, box = item["side"], item["confidence"], item["points"], item["box"]
            track_id = self._assign_id(side, item["center"])
            # Landmark dots only; intentionally no connected skeleton lines.
            for px, py in points:
                cv2.circle(frame, (int(px), int(py)), 2, (255, 0, 255), -1)
            label = f"{side} {confidence:.2f}"
            cv2.putText(frame, label, (int(box[0]), max(18, int(box[1])-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                         (255, 0, 255), 2)
            detections.append({"track_id": track_id, "class_name": side.title(),
                              "confidence": confidence, "box": box})
        return detections


    def close(self):
        if self.hands:
            try:
                self.hands.close()
            except Exception:
                pass
