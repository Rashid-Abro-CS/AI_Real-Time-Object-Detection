"""
Entry point: opens the webcam, runs YOLO detection + ByteTrack tracking on
every frame, runs Places365 scene classification periodically, manages
object/scene/relationship sessions, writes finished sessions to MongoDB,
and shows a live annotated preview window.

    START CAMERA        -> run:  python main.py
    STOP CAMERA / EXIT   -> press "q" or ESC in the preview window, or Ctrl+C
    CLEAR HISTORY        -> press "c" in the preview window (asks to confirm
                             in the terminal)
    VIEW HISTORY /
    SCENE HISTORY /
    RELATIONSHIPS / SEARCH -> run:  streamlit run dashboard.py  (separate terminal)
"""

import argparse
import datetime
import logging
import time
from pdf_exporter import generate_objects_pdf

import cv2

import config
from database import Database
from detector import ObjectDetector
from pose_detector import BodyPoseDetector
from hand_detector import HandLandmarkDetector
from tracker import ObjectSessionManager
from relationship import RelationshipManager
from scene_classifier import SceneClassifier, SceneSessionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("main")


def open_camera():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {config.CAMERA_INDEX}. "
            "Check that a webcam is connected and not already in use by another app."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def draw_overlay(frame, detections, scene_name, scene_conf, current_group, fps, raw_detections=None, show_raw=False, pose_enabled=False, hands_enabled=False, hand_count=0, hand_error=None):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = f'{det["class_name"]} {det["confidence"] * 100:.0f}% ID:{det["track_id"]}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    if show_raw and raw_detections:
        for det in raw_detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            label = f'RAW {det["class_name"]} {det["confidence"] * 100:.0f}%'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
            cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)
    y = 25
    if scene_name:
        cv2.putText(frame, f"Scene: {scene_name} ({scene_conf * 100:.0f}%)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y += 30
    if current_group:
        cv2.putText(frame, "Together: " + " + ".join(current_group), (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 25
    hand_status = (f"ACTIVE {hand_count}" if hands_enabled and hand_count > 0 else ("SEARCHING" if hands_enabled else "UNAVAILABLE"))
    status = f"BODY BOXES: {'ON' if pose_enabled else 'OFF'} | HANDS: {hand_status}"
    cv2.putText(frame, status, (10, frame.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if hand_error:
        short_error = str(hand_error).replace('\\n', ' ')[:100]
        cv2.putText(frame, f"Hand setup error: {short_error}", (10, frame.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Real-time object, body-pose and hand detection")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV preview window")
    parser.add_argument("--raw", action="store_true", help="Show raw YOLO detections in preview")
    parser.add_argument("--imgsz", type=int, default=config.YOLO_IMAGE_SIZE, help="Primary model image size")
    args = parser.parse_args()
    config.CAMERA_INDEX = args.camera
    config.YOLO_IMAGE_SIZE = args.imgsz
    logger.info("Starting %s", config.PROJECT_NAME)

    db = Database()
    try:
        detector = ObjectDetector()
        scene_classifier = SceneClassifier()
        pose_detector = BodyPoseDetector(
            config.POSE_MODEL_PATH, detector.device, conf=config.POSE_CONFIDENCE,
            keypoint_conf=config.POSE_KEYPOINT_CONFIDENCE, tracker=config.YOLO_TRACKER_CONFIG,
        ) if config.ENABLE_BODY_POSE else None
        hand_detector = HandLandmarkDetector(config.ENABLE_MEDIAPIPE_HANDS)
    except Exception as exc:
        logger.error("Startup model initialization failed: %s", exc)
        db.close()
        return

    object_sessions = ObjectSessionManager(db)
    relationship_manager = RelationshipManager(db)
    scene_sessions = SceneSessionManager(db)

    try:
        cap = open_camera()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    except RuntimeError as exc:
        logger.error(str(exc))
        hand_detector.close()
        db.close()
        return

    frame_count = 0
    prev_time = time.time()
    fps = 0.0
    last_live_state_update = 0.0
    last_scene_classify_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Failed to read frame from camera. Stopping.")
                break

            # Mirror webcam preview (selfie-style); pose and hand side labels are corrected for this convention.
            frame = cv2.flip(frame, 1)

            frame_count += 1
            now = datetime.datetime.now()

            detections = detector.track(frame, frame_idx=frame_count)
            # Pose keypoints supplement general object classes with body-part observations.
            if pose_detector and pose_detector.available:
                run_pose = frame_count % max(1, config.POSE_EVERY_N_FRAMES) == 0
                # Reuse the last pose boxes between inference frames to prevent
                # body-part overlays from blinking with the inference cadence.
                detections.extend(pose_detector.process(
                    frame, frame_idx=frame_count, tracker=config.YOLO_TRACKER_CONFIG,
                    run_inference=run_pose))
            # MediaPipe hands is optional; when installed, hand boxes/landmarks are
            # included in the same confirmation/session pipeline.
            if hand_detector.available:
                wrist_hints = pose_detector.get_wrist_hints() if pose_detector and pose_detector.available else []
                detections.extend(hand_detector.process(frame, wrist_hints=wrist_hints))
            # Returns only detections of CONFIRMED objects, with stabilised ids/classes.
            detections = object_sessions.update(detections, timestamp=now)

            current_objects = object_sessions.current_objects()
            relationship_manager.update(current_objects, timestamp=now)

            if time.monotonic() - last_scene_classify_time >= config.SCENE_CLASSIFY_INTERVAL_SECONDS:
                scene_name, scene_conf = scene_classifier.predict(frame)
                scene_sessions.update(scene_name, scene_conf, timestamp=now)
                last_scene_classify_time = time.monotonic()

            elapsed = time.time() - prev_time
            prev_time = time.time()
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed)

            current_group = relationship_manager.current_group()
            show_window = config.SHOW_CAMERA_WINDOW and not args.headless
            frame = draw_overlay(
                frame, detections, scene_sessions.current_scene,
                scene_sessions.current_confidence, current_group, fps,
                raw_detections=detector.last_raw_detections,
                show_raw=args.raw,
                pose_enabled=bool(pose_detector and pose_detector.available),
                hands_enabled=hand_detector.available,
                hand_count=hand_detector.last_count,
                hand_error=hand_detector.error or hand_detector.inference_error,
            )
            if show_window:
                try:
                    cv2.imshow(config.WINDOW_NAME, frame)
                except cv2.error as exc:
                    logger.warning("OpenCV preview unavailable; continuing headless: %s", exc)
                    show_window = False

            if time.time() - last_live_state_update >= config.LIVE_STATE_UPDATE_SECONDS:
                db.update_live_state({
                    "scene": scene_sessions.current_scene,
                    "scene_confidence": scene_sessions.current_confidence,
                    "objects": [o["class_name"] for o in current_objects],
                    "objects_together": current_group,
                    "fps": round(fps, 1),
                    "timestamp": now.isoformat(),
                    "raw_detection_count": len(detector.last_raw_detections),
                    "raw_classes": sorted({d["class_name"] for d in detector.last_raw_detections}),
                    "device": detector.device,
                    "pose_enabled": bool(pose_detector and pose_detector.available),
                    "hands_enabled": hand_detector.available,
                    "pending_database_writes": db.pending_count,
                })
                last_live_state_update = time.time()

            if show_window:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
            if key in (ord("q"), 27):  # q or ESC
                logger.info("Stop requested by user.")
                break
            elif key == ord("d"):
                args.raw = not args.raw
                logger.info("Raw detection overlay: %s", args.raw)
            elif key == ord("c"):
                confirm = input("Clear ALL history from MongoDB? (yes/no): ")
                if confirm.strip().lower() == "yes":
                    db.clear_history()
                    logger.info("History cleared.")

    # except KeyboardInterrupt:
    #     logger.info("Interrupted by user (Ctrl+C).")
    # finally:
    #     object_sessions.close_all()
    #     relationship_manager.close_all()
    #     scene_sessions.close_all()
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     db.close()
    #     logger.info("Camera stopped safely. All open sessions were saved.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    finally:
        object_sessions.close_all()
        relationship_manager.close_all()
        scene_sessions.close_all()
        
        # --- ADD THIS LINE TO EXPORT PDF AUTOMATICALLY ON EXIT ---
        try:
            pdf_path = generate_objects_pdf(db, run_id=object_sessions.run_id)
        except Exception as exc:  # never let report problems skip camera cleanup
            pdf_path = None
            logger.error("PDF report failed: %s", exc)

        cap.release()
        hand_detector.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        db.close()
        if pdf_path:
            logger.info("Camera stopped safely. PDF report: %s", pdf_path)
        else:
            logger.info("Camera stopped safely. No confirmed detections for this run -> no PDF.")
        logger.info("Run %s: %d sessions saved, rejected sessions: %s, rejected detections: %s",
                    object_sessions.run_id, object_sessions.saved_sessions,
                    dict(object_sessions.session_rejections), dict(object_sessions.frame_rejections))
    # finally:
    #     object_sessions.close_all()
    #     relationship_manager.close_all()
    #     scene_sessions.close_all()
        
    #     # --- ADD THIS LINE TO EXPORT PDF AUTOMATICALLY ON EXIT ---
    #     generate_objects_pdf(db)
        
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     db.close()
    #     logger.info("Camera stopped safely. PDF report generated in 'results/' folder.")


if __name__ == "__main__":
    main()


    