"""
Camera/video diagnostic for raw YOLO detections, tracked detections, and
confirmed sessions. No MongoDB connection is made.

Examples:
    python diagnose.py --seconds 60 --show --raw
    python diagnose.py --source clip.mp4 --show
"""
import argparse
import collections
import datetime
import time

import cv2
import config
from detector import ObjectDetector
from tracker import ObjectSessionManager
from pose_detector import BodyPoseDetector
from hand_detector import HandLandmarkDetector


class _MemoryDB:
    def __init__(self):
        self.docs = []
    def save_detection_session(self, doc):
        self.docs.append(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(config.CAMERA_INDEX))
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--raw", action="store_true", help="Show raw detector boxes")
    ap.add_argument("--list-classes", action="store_true", help="Print model class names and exit")
    args = ap.parse_args()
    detector = ObjectDetector()
    if args.list_classes:
        print("Primary Open Images model classes:")
        for cls in detector.available_classes:
            print(cls)
        if detector.supplemental is not None:
            names = detector.supplemental_names
            classes = list(names.values()) if isinstance(names, dict) else list(names)
            print("\\nSupplemental COCO classes:")
            for cls in classes:
                print(cls)
        return
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source}")
    pose = BodyPoseDetector(config.POSE_MODEL_PATH, detector.device,
                            conf=config.POSE_CONFIDENCE,
                            keypoint_conf=config.POSE_KEYPOINT_CONFIDENCE,
                            tracker=config.YOLO_TRACKER_CONFIG) if config.ENABLE_BODY_POSE else None
    hands = HandLandmarkDetector(config.ENABLE_MEDIAPIPE_HANDS)
    db = _MemoryDB()
    mgr = ObjectSessionManager(db, run_id="diagnose")
    raw_counts = collections.Counter()
    raw_conf = collections.defaultdict(float)
    tracked_ids = collections.defaultdict(set)
    frames, start = 0, time.time()
    try:
        while time.time() - start < args.seconds:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frames += 1
            timestamp = datetime.datetime.now()
            dets = detector.track(frame, frame_idx=frames)
            for raw in detector.last_raw_detections:
                raw_counts[raw["class_name"]] += 1
                raw_conf[raw["class_name"]] += raw["confidence"]
            if pose and pose.available:
                run_pose = frames % max(1, config.POSE_EVERY_N_FRAMES) == 0
                dets.extend(pose.process(frame, frames, config.YOLO_TRACKER_CONFIG,
                                        run_inference=run_pose))
            if hands.available:
                dets.extend(hands.process(frame))
            for d in dets:
                tracked_ids[d["class_name"]].add(d["track_id"])
            shown = mgr.update(dets, timestamp=timestamp)
            if args.show:
                for d in shown:
                    x1, y1, x2, y2 = map(int, d["box"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(frame, f'{d["class_name"]} {d["confidence"]:.2f} ID:{d["track_id"]}',
                                (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
                if args.raw:
                    for d in detector.last_raw_detections:
                        x1, y1, x2, y2 = map(int, d["box"])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
                        cv2.putText(frame, f'RAW {d["class_name"]} {d["confidence"]:.2f}',
                                    (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                cv2.imshow("diagnose", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        mgr.close_all()
        cap.release()
        hands.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    elapsed = time.time() - start
    print(f"\n{frames} frames in {elapsed:.1f}s ({frames / max(elapsed, 1e-9):.1f} FPS)")
    print("\nRAW YOLO detections (before tracking/session filters):")
    for cls in sorted(raw_counts):
        avg = raw_conf[cls] / raw_counts[cls]
        print(f"  {cls:<24} {raw_counts[cls]:>5} detections  avg confidence={avg:.3f}")
    print("\nTracked IDs -> confirmed sessions saved:")
    sessions = collections.Counter(d["object_name"] for d in db.docs)
    for cls in sorted(set(tracked_ids) | set(sessions)):
        print(f"  {cls:<24} {len(tracked_ids.get(cls, ())):>4} ids  ->  {sessions.get(cls, 0):>3} sessions")
    print("\nSessions:")
    for d in db.docs:
        print(f"  {d['object_name']:<16} id={d['tracking_id']:<4} ids={d['track_ids']} "
              f"{d['first_seen']}-{d['last_seen']} {d['duration_seconds']}s "
              f"avg={d['avg_confidence']:.2f} max={d['max_confidence']:.2f}")
    print("\nRejected sessions:", dict(mgr.session_rejections))
    print("Rejected detections:", dict(mgr.frame_rejections))


if __name__ == "__main__":
    main()
